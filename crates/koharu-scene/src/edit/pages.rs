//! Page ops.

use super::Run;
use crate::{
    Background, Content, Created, Error, Hash, Layer, LayerId, Page, PageId, Ref, Result, Size,
};

impl Run<'_, '_> {
    pub(super) fn merge_pages(
        &mut self,
        op: usize,
        pages: &[Ref],
        rtl: bool,
        keep_originals: bool,
    ) -> Result<()> {
        let requested: std::collections::BTreeSet<_> = pages
            .iter()
            .map(|page| self.page(page))
            .collect::<Result<_>>()?;
        if requested.len() != pages.len() || requested.len() < 2 {
            return Err(Error::invalid(
                op,
                "a spread needs at least two distinct pages",
            ));
        }
        let mut sources: Vec<_> = self
            .project
            .order
            .iter()
            .filter(|id| requested.contains(id))
            .map(|id| self.project.pages[id].clone())
            .collect();
        let index = self
            .project
            .order
            .index_of(&sources[0].id)
            .expect("the page is ordered");
        let name = sources
            .iter()
            .map(|page| page.name.as_str())
            .filter(|name| !name.is_empty())
            .collect::<Vec<_>>()
            .join(" + ");
        let chapter = sources[0].chapter.clone();
        if rtl {
            sources.reverse();
        }
        let width = sources
            .iter()
            .try_fold(0u32, |width, page| width.checked_add(page.size.width))
            .filter(|width| *width <= i32::MAX as u32)
            .ok_or_else(|| Error::invalid(op, "the spread is too wide"))?;
        let height = sources
            .iter()
            .map(|page| page.size.height)
            .max()
            .expect("two pages");
        let mut scan = image::RgbaImage::new(width, height);
        let mut x = 0u32;
        for page in &sources {
            if let Some(background) = page.background.filter(|background| background.visible) {
                let bytes = self.tx.blob(&background.file)?;
                let image = image::load_from_memory(&bytes)
                    .map_err(|error| Error::invalid(op, error.to_string()))?
                    .to_rgba8();
                image::imageops::replace(&mut scan, &image, i64::from(x), 0);
            }
            x += page.size.width;
        }
        let mut encoded = std::io::Cursor::new(Vec::new());
        scan.write_to(&mut encoded, image::ImageFormat::Png)
            .map_err(|error| Error::invalid(op, error.to_string()))?;
        let background = self.tx.put_blob(encoded.get_ref())?;
        self.add_page(op, None, index, None, Some(&background))?;
        let id = self.project.order[index];
        let page = self
            .project
            .pages
            .get_mut(&id)
            .expect("the spread was added");
        page.name = name;
        page.chapter = chapter;
        let mut x = 0i32;
        for source in sources {
            for mut layer in source.layers {
                let copy = LayerId(self.fresh(op)?);
                self.renumber(op, &mut layer, copy, None)?;
                shift(op, &mut layer, x)?;
                let index = self.project.pages[&id].layers.len();
                self.project.insert(id, &[], index, layer);
            }
            x += source.size.width as i32;
        }
        if !keep_originals {
            for source in requested {
                self.remove_page(&source.into())?;
            }
        }
        Ok(())
    }
    pub(super) fn add_page(
        &mut self,
        op: usize,
        id: Option<&Ref>,
        index: usize,
        size: Option<Size>,
        background: Option<&Hash>,
    ) -> Result<()> {
        // A page made from a file takes that file's size.
        let (size, background) = match (size, background) {
            (Some(size), None) => (size, None),
            (None, Some(file)) => (
                self.image_size(op, file)?,
                Some(Background {
                    file: *file,
                    visible: true,
                }),
            ),
            _ => {
                return Err(Error::invalid(
                    op,
                    "a page takes a size or a background, not both",
                ));
            }
        };
        if size.width == 0 || size.height == 0 {
            return Err(Error::invalid(op, "a page needs a nonzero size"));
        }
        if index > self.project.order.len() {
            return Err(Error::invalid(
                op,
                format!("index {index} is past the last page"),
            ));
        }
        let (uuid, label) = self.new_id(op, id)?;
        let id = PageId(uuid);
        self.project.order.insert(index, id);
        self.project.pages.insert(
            id,
            Page {
                id,
                name: String::new(),
                chapter: None,
                size,
                background,
                layers: Vec::new(),
            },
        );
        self.created.push(Created::Page { id, label });
        Ok(())
    }

    pub(super) fn update_page(
        &mut self,
        op: usize,
        page: &Ref,
        patch: &serde_json::Value,
    ) -> Result<()> {
        #[derive(serde::Serialize, serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Metadata {
            name: String,
            chapter: Option<String>,
        }
        if !patch.is_object() {
            return Err(Error::invalid(op, "a page patch must be an object"));
        }
        let id = self.page(page)?;
        let page = &self.project.pages[&id];
        let mut metadata = serde_json::to_value(Metadata {
            name: page.name.clone(),
            chapter: page.chapter.clone(),
        })
        .expect("page metadata serializes");
        json_patch::merge(&mut metadata, patch);
        let metadata: Metadata = serde_json::from_value(metadata)
            .map_err(|error| Error::invalid(op, error.to_string()))?;
        let page = self.project.pages.get_mut(&id).expect("the page was found");
        page.name = metadata.name;
        page.chapter = metadata.chapter;
        Ok(())
    }

    pub(super) fn move_page(&mut self, op: usize, page: &Ref, index: usize) -> Result<()> {
        let id = self.page(page)?;
        if index >= self.project.order.len() {
            return Err(Error::invalid(
                op,
                format!("index {index} is past the last page"),
            ));
        }
        let order = &mut self.project.order;
        let from = order.index_of(&id).expect("every page is in the order");
        order.remove(from);
        order.insert(index, id);
        Ok(())
    }

    /// A new background must be an image of the page's size.
    pub(super) fn set_background(
        &mut self,
        op: usize,
        page: &Ref,
        background: Option<&Background>,
    ) -> Result<()> {
        let id = self.page(page)?;
        if let Some(background) = background {
            let size = self.image_size(op, &background.file)?;
            let page = &self.project.pages[&id];
            if size != page.size {
                return Err(Error::invalid(
                    op,
                    format!(
                        "file {} is {}×{}, but the page is {}×{}",
                        background.file, size.width, size.height, page.size.width, page.size.height
                    ),
                ));
            }
        }
        self.project
            .pages
            .get_mut(&id)
            .expect("the page was just found")
            .background = background.copied();
        Ok(())
    }

    /// Its layers and their pixels go with it.
    pub(super) fn remove_page(&mut self, page: &Ref) -> Result<()> {
        let id = self.page(page)?;
        while !self.project.pages[&id].layers.is_empty() {
            let layer = self.project.take(id, &[0]);
            self.forget(&layer);
        }
        self.project.pages.remove(&id);
        self.project.order.retain(|page| *page != id);
        Ok(())
    }
}

/// Moving a page moves every coordinate owner, including hidden/locked parts;
/// it does not change their independent editing locks.
fn shift(op: usize, layer: &mut Layer, x: i32) -> Result<()> {
    let affine = kurbo::Affine::translate((f64::from(x), 0.0));
    let text = |text: &mut crate::Text| {
        text.transform = (affine * kurbo::Affine::from(text.transform)).into()
    };
    let path = |path: &mut crate::Path| {
        let mut moved = kurbo::BezPath::from(&*path);
        moved.apply_affine(affine);
        *path = crate::Path::try_from(moved.to_svg()).expect("kurbo writes valid SVG");
    };
    let offset = |offset: &mut crate::Offset| -> Result<()> {
        offset.x = offset
            .x
            .checked_add(x)
            .ok_or_else(|| Error::invalid(op, "layer offset is out of range"))?;
        Ok(())
    };
    match &mut layer.content {
        Content::Group(_) => {}
        Content::Image(image) => {
            image.transform = (affine * kurbo::Affine::from(image.transform)).into()
        }
        Content::Raster(raster) => offset(&mut raster.offset)?,
        Content::Mask(mask) => offset(&mut mask.offset)?,
        Content::Shape(shape) => path(&mut shape.path),
        Content::Panel(panel) => path(&mut panel.outline),
        Content::Balloon(balloon) => path(&mut balloon.outline),
        Content::Text(value) => text(value),
        Content::Block(block) => {
            text(&mut block.source);
            if let Some(value) = &mut block.translation {
                text(value);
            }
            offset(&mut block.mask.content.offset)?;
            offset(&mut block.cleanup.content.offset)?;
        }
    }
    for child in &mut layer.children {
        shift(op, child, x)?;
    }
    Ok(())
}
