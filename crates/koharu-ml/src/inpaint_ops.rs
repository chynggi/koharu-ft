//! Shared inpainting preprocessing, crop orchestration, and postprocessing
//! helpers, used by LaMa and MI-GAN.

use anyhow::{Context, Result, anyhow};
use fast_image_resize::{FilterType, ResizeAlg, ResizeOptions, Resizer};
use image::{GrayImage, RgbImage};
use imageproc::contours::{BorderType, find_contours_with_threshold};
use koharu_torch::{Device, Kind, Tensor};

use crate::lama::{HDStrategy, InpaintRequest};

pub(crate) fn resize_dimensions(width: u32, height: u32, size_limit: u32) -> (u32, u32) {
    let ratio = size_limit as f64 / width.max(height) as f64;
    (
        (width as f64 * ratio + 0.5) as u32,
        (height as f64 * ratio + 0.5) as u32,
    )
}

pub(crate) fn resize_rgb(image: &RgbImage, width: u32, height: u32) -> Result<RgbImage> {
    let mut output = RgbImage::new(width, height);
    Resizer::new()
        .resize(
            image,
            &mut output,
            &ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::CatmullRom)),
        )
        .map_err(|error| anyhow!("failed to resize LaMa RGB image: {error}"))?;
    Ok(output)
}

pub(crate) fn resize_gray(image: &GrayImage, width: u32, height: u32) -> Result<GrayImage> {
    let mut output = GrayImage::new(width, height);
    Resizer::new()
        .resize(
            image,
            &mut output,
            &ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::CatmullRom)),
        )
        .map_err(|error| anyhow!("failed to resize LaMa mask: {error}"))?;
    Ok(output)
}

pub(crate) fn boxes_from_mask(mask: &GrayImage) -> Vec<[u32; 4]> {
    let width = mask.width();
    let mut left = width;
    let mut top = mask.height();
    let mut right = 0;
    let mut bottom = 0;
    for y in 0..mask.height() {
        let row = &mask.as_raw()[y as usize * width as usize..(y + 1) as usize * width as usize];
        let Some(row_left) = row.iter().position(|value| *value > 127) else {
            continue;
        };
        let row_right = row
            .iter()
            .rposition(|value| *value > 127)
            .expect("masked row must have a right edge");
        left = left.min(row_left as u32);
        top = top.min(y);
        right = right.max(row_right as u32 + 1);
        bottom = y + 1;
    }
    if right <= left || bottom <= top {
        return Vec::new();
    }

    let cropped_width = right - left;
    let cropped_height = bottom - top;
    let padded_width = cropped_width + 2;
    let mut padded = GrayImage::new(padded_width, cropped_height + 2);
    for y in 0..cropped_height as usize {
        let source_start = (top as usize + y) * width as usize + left as usize;
        let target_start = (y + 1) * padded_width as usize + 1;
        padded.as_mut()[target_start..target_start + cropped_width as usize]
            .copy_from_slice(&mask.as_raw()[source_start..source_start + cropped_width as usize]);
    }

    find_contours_with_threshold::<u32>(&padded, 127)
        .into_iter()
        .filter(|contour| contour.border_type == BorderType::Outer && contour.parent.is_none())
        .filter_map(|contour| {
            let mut points = contour.points.into_iter();
            let first = points.next()?;
            let mut contour_left = first.x;
            let mut contour_top = first.y;
            let mut contour_right = first.x;
            let mut contour_bottom = first.y;
            for point in points {
                contour_left = contour_left.min(point.x);
                contour_top = contour_top.min(point.y);
                contour_right = contour_right.max(point.x);
                contour_bottom = contour_bottom.max(point.y);
            }
            Some([
                left + contour_left.saturating_sub(1),
                top + contour_top.saturating_sub(1),
                (left + contour_right).min(mask.width()),
                (top + contour_bottom).min(mask.height()),
            ])
        })
        .filter(|[left, top, right, bottom]| right > left && bottom > top)
        .collect()
}

pub(crate) fn crop_box(
    image_width: u32,
    image_height: u32,
    [left, top, right, bottom]: [u32; 4],
    margin: u32,
) -> [u32; 4] {
    let image_width = i64::from(image_width);
    let image_height = i64::from(image_height);
    let crop_width = i64::from(right - left) + i64::from(margin) * 2;
    let crop_height = i64::from(bottom - top) + i64::from(margin) * 2;
    let center_x = (i64::from(left) + i64::from(right)) / 2;
    let center_y = (i64::from(top) + i64::from(bottom)) / 2;

    let raw_left = center_x - crop_width / 2;
    let raw_right = center_x + crop_width / 2;
    let raw_top = center_y - crop_height / 2;
    let raw_bottom = center_y + crop_height / 2;

    let mut left = raw_left.max(0);
    let mut right = raw_right.min(image_width);
    let mut top = raw_top.max(0);
    let mut bottom = raw_bottom.min(image_height);

    if raw_left < 0 {
        right += -raw_left;
    }
    if raw_right > image_width {
        left -= raw_right - image_width;
    }
    if raw_top < 0 {
        bottom += -raw_top;
    }
    if raw_bottom > image_height {
        top -= raw_bottom - image_height;
    }

    [
        left.clamp(0, image_width) as u32,
        top.clamp(0, image_height) as u32,
        right.clamp(0, image_width) as u32,
        bottom.clamp(0, image_height) as u32,
    ]
}

pub(crate) fn post_process(tensor: &Tensor, width: u32, height: u32) -> Result<RgbImage> {
    let tensor = tensor
        .squeeze_dim(0)
        .permute([1, 2, 0])
        .contiguous()
        .to_device(Device::Cpu)
        .view([-1]);
    let rgb = Vec::<u8>::try_from(&tensor)?;
    RgbImage::from_raw(width, height, rgb).context("failed to convert output tensor to RGB image")
}

pub(crate) fn pad_img_to_modulo(tensor: Tensor, modulo: u32) -> Tensor {
    let height = tensor.size()[2] as u32;
    let width = tensor.size()[3] as u32;
    let out_height = ceil_modulo(height, modulo);
    let out_width = ceil_modulo(width, modulo);
    let height_indices = symmetric_indices(height, out_height, tensor.device());
    let width_indices = symmetric_indices(width, out_width, tensor.device());
    tensor
        .index_select(2, &height_indices)
        .index_select(3, &width_indices)
}

pub(crate) fn ceil_modulo(value: u32, modulo: u32) -> u32 {
    if value.is_multiple_of(modulo) {
        value
    } else {
        (value / modulo + 1) * modulo
    }
}

/// IOPaint's base-model orchestration (`base.py`'s `__call__`): pick the crop,
/// resize, or original strategy from the request, run `pad_forward` on every
/// region the strategy produces, and paste the results back. Shared by every
/// model that follows the base class — LaMa and the Manga inpainter. Models
/// with their own `__call__`, like MI-GAN, orchestrate themselves.
pub(crate) fn dispatch_hd_strategy(
    image: &RgbImage,
    mask: &GrayImage,
    config: &InpaintRequest,
    pad_forward: &dyn Fn(&RgbImage, &GrayImage) -> Result<RgbImage>,
) -> Result<RgbImage> {
    match config.hd_strategy {
        HDStrategy::Crop
            if image.width().max(image.height()) > config.hd_strategy_crop_trigger_size =>
        {
            let boxes = boxes_from_mask(mask);
            let mut crop_results = Vec::with_capacity(boxes.len());
            for bounding_box in boxes {
                let crop = crop_box(
                    image.width(),
                    image.height(),
                    bounding_box,
                    config.hd_strategy_crop_margin,
                );
                let [left, top, right, bottom] = crop;
                let crop_image =
                    image::imageops::crop_imm(image, left, top, right - left, bottom - top)
                        .to_image();
                let crop_mask =
                    image::imageops::crop_imm(mask, left, top, right - left, bottom - top)
                        .to_image();
                crop_results.push((pad_forward(&crop_image, &crop_mask)?, crop));
            }

            let mut result = image.clone();
            for (crop_result, [left, top, _, _]) in crop_results {
                image::imageops::replace(&mut result, &crop_result, left.into(), top.into());
            }
            Ok(result)
        }
        HDStrategy::Resize
            if image.width().max(image.height()) > config.hd_strategy_resize_limit =>
        {
            let (width, height) = resize_dimensions(
                image.width(),
                image.height(),
                config.hd_strategy_resize_limit,
            );
            let resized_image = resize_rgb(image, width, height)?;
            let resized_mask = resize_gray(mask, width, height)?;
            let resized_result = pad_forward(&resized_image, &resized_mask)?;
            let mut result = resize_rgb(&resized_result, image.width(), image.height())?;
            for (index, value) in mask.as_raw().iter().enumerate() {
                if *value < 127 {
                    let offset = index * 3;
                    result.as_mut()[offset..offset + 3]
                        .copy_from_slice(&image.as_raw()[offset..offset + 3]);
                }
            }
            Ok(result)
        }
        _ => pad_forward(image, mask),
    }
}

pub(crate) fn symmetric_indices(length: u32, output_length: u32, device: Device) -> Tensor {
    if !device.is_cuda() {
        let indices = (0..output_length)
            .map(|index| i64::from(symmetric_index(index, length)))
            .collect::<Vec<_>>();
        return Tensor::from_slice(&indices).to_device(device);
    }
    let length = i64::from(length);
    let period = length * 2;
    let indices = Tensor::arange(i64::from(output_length), (Kind::Int64, device));
    let indices = &indices - indices.floor_divide_scalar(period) * period;
    indices.where_self(&indices.lt(length), &(period - &indices - 1))
}

fn symmetric_index(index: u32, length: u32) -> u32 {
    let index = index % (length * 2);
    if index < length {
        index
    } else {
        length * 2 - index - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use image::Luma;

    #[test]
    fn an_in_range_index_passes_through_unchanged() {
        assert_eq!(symmetric_index(0, 4), 0);
        assert_eq!(symmetric_index(2, 4), 2);
    }

    #[test]
    fn the_last_in_range_index_is_its_own_reflection() {
        // length - 1
        assert_eq!(symmetric_index(3, 4), 3);
    }

    #[test]
    fn one_past_the_end_duplicates_the_last_in_range_index() {
        // length
        assert_eq!(symmetric_index(4, 4), 3);
    }

    #[test]
    fn the_far_end_of_the_period_reflects_back_to_the_start() {
        // 2 * length - 1
        assert_eq!(symmetric_index(7, 4), 0);
    }

    #[test]
    fn a_full_period_wraps_back_to_the_start() {
        // 2 * length
        assert_eq!(symmetric_index(8, 4), 0);
    }

    #[test]
    fn symmetric_indices_has_one_entry_per_output_position() {
        let indices = symmetric_indices(4, 6, Device::Cpu);
        assert_eq!(indices.size(), vec![6]);
        assert_eq!(indices.kind(), Kind::Int64);
    }

    #[test]
    fn symmetric_indices_pads_the_far_side_by_mirroring_inward() {
        let indices = symmetric_indices(4, 6, Device::Cpu);
        let values = Vec::<i64>::try_from(&indices).unwrap();
        assert_eq!(values, vec![0, 1, 2, 3, 3, 2]);
    }

    #[test]
    fn an_already_aligned_size_is_left_untouched() {
        assert_eq!(ceil_modulo(16, 8), 16);
    }

    #[test]
    fn a_misaligned_size_rounds_up_to_the_next_multiple() {
        assert_eq!(ceil_modulo(9, 8), 16);
        assert_eq!(ceil_modulo(1, 8), 8);
    }

    #[test]
    fn an_already_aligned_tensor_is_returned_unpadded() {
        let tensor = Tensor::from_slice(&[0f32, 1., 2., 3., 4., 5., 6., 7.]).view([1, 1, 2, 4]);
        let expected = Vec::<f32>::try_from(&tensor.shallow_clone().view([-1])).unwrap();

        let padded = pad_img_to_modulo(tensor, 2);

        assert_eq!(padded.size(), vec![1, 1, 2, 4]);
        let actual = Vec::<f32>::try_from(&padded.view([-1])).unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn a_misaligned_tensor_is_padded_on_the_far_side_by_reflection() {
        // 3x5, padded up to modulo 4 becomes 4x8. The bottom row and the
        // trailing columns are mirrored back from the last real row/column,
        // and the original 3x5 content is left untouched in the top-left.
        let tensor = Tensor::from_slice(&[
            0f32, 1., 2., 3., 4., //
            10., 11., 12., 13., 14., //
            20., 21., 22., 23., 24.,
        ])
        .view([1, 1, 3, 5]);

        let padded = pad_img_to_modulo(tensor, 4);

        assert_eq!(padded.size(), vec![1, 1, 4, 8]);
        let actual = Vec::<f32>::try_from(&padded.view([-1])).unwrap();
        #[rustfmt::skip]
        let expected = vec![
            0f32, 1., 2., 3., 4., 4., 3., 2.,
            10., 11., 12., 13., 14., 14., 13., 12.,
            20., 21., 22., 23., 24., 24., 23., 22.,
            20., 21., 22., 23., 24., 24., 23., 22.,
        ];
        assert_eq!(actual, expected);
    }

    #[test]
    fn a_box_in_the_middle_expands_by_the_margin_on_every_side() {
        let box_ = crop_box(100, 100, [40, 40, 60, 60], 10);
        assert_eq!(box_, [30, 30, 70, 70]);
    }

    #[test]
    fn a_box_touching_the_left_edge_clamps_instead_of_going_negative() {
        let box_ = crop_box(100, 100, [0, 40, 20, 60], 10);
        // The full margin width is preserved by extending the opposite side.
        assert_eq!(box_, [0, 30, 40, 70]);
    }

    #[test]
    fn a_box_touching_the_right_edge_clamps_instead_of_overshooting() {
        let box_ = crop_box(100, 100, [80, 40, 100, 60], 10);
        assert_eq!(box_, [60, 30, 100, 70]);
    }

    #[test]
    fn a_box_that_fills_the_image_stays_within_bounds_after_margin() {
        let box_ = crop_box(50, 50, [0, 0, 50, 50], 10);
        assert_eq!(box_, [0, 0, 50, 50]);
    }

    #[test]
    fn an_all_black_mask_yields_no_boxes() {
        let mask = GrayImage::new(10, 10);
        assert_eq!(boxes_from_mask(&mask), Vec::<[u32; 4]>::new());
    }

    #[test]
    fn a_single_white_rectangle_yields_one_box_that_contains_it() {
        let mut mask = GrayImage::new(20, 20);
        for y in 5..15 {
            for x in 5..15 {
                mask.put_pixel(x, y, Luma([255]));
            }
        }

        let boxes = boxes_from_mask(&mask);

        assert_eq!(boxes.len(), 1);
        let [left, top, right, bottom] = boxes[0];
        assert!(left <= 5 && top <= 5 && right >= 15 && bottom >= 15);
        assert!(right <= mask.width() && bottom <= mask.height());
    }

    #[test]
    fn a_mask_region_touching_the_image_edge_is_not_clipped() {
        let mut mask = GrayImage::new(20, 20);
        for y in 5..15 {
            for x in 0..10 {
                mask.put_pixel(x, y, Luma([255]));
            }
        }

        let boxes = boxes_from_mask(&mask);

        assert_eq!(boxes.len(), 1);
        let [left, top, right, bottom] = boxes[0];
        assert_eq!(left, 0);
        assert!(top <= 5 && right >= 10 && bottom >= 15);
    }
}
