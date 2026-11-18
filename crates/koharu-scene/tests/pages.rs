//! Page metadata, merging pages into a spread, and copying a layer to
//! another page.

mod common;
use std::io::Cursor;

use common::*;
use koharu_scene::{Content, Error, Hash, PageId, Session};
use serde_json::json;

fn png(width: u32, height: u32, color: [u8; 4]) -> Vec<u8> {
    let mut bytes = Vec::new();
    image::RgbaImage::from_pixel(width, height, image::Rgba(color))
        .write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)
        .expect("a PNG");
    bytes
}

/// Adds a page at the end made from a `width`×`height` scan of `color`.
fn scanned(session: &mut Session, width: u32, height: u32, color: [u8; 4]) -> PageId {
    let bytes = png(width, height, color);
    let file = Hash::of(&bytes);
    let index = session.project().pages().count();
    let change = apply(
        session,
        vec![
            json!({"op": "add_file", "bytes": bytes}),
            json!({"op": "add_page", "index": index, "background": file}),
        ],
    )
    .expect("a page");
    change.pages.added[0]
}

#[test]
fn a_page_patch_sets_only_its_name_and_chapter() {
    let (_dir, mut session, page) = session();
    apply(
        &mut session,
        vec![
            json!({"op": "update_page", "page": page, "patch": {"name": "Cover", "chapter": "1"}}),
        ],
    )
    .expect("metadata");
    apply(
        &mut session,
        vec![json!({"op": "update_page", "page": page, "patch": {"chapter": null}})],
    )
    .expect("no chapter");
    let stored = session.project().page(page).expect("the page").clone();
    assert_eq!((stored.name.as_str(), stored.chapter), ("Cover", None));

    let before = session.project().clone();
    for patch in [json!({"size": {"width": 1, "height": 1}}), json!("Cover")] {
        let error = apply(
            &mut session,
            vec![json!({"op": "update_page", "page": page, "patch": patch})],
        )
        .err()
        .expect("refused");
        assert!(matches!(error, Error::Invalid { .. }), "{patch}: {error}");
    }
    assert_eq!(session.project(), before);
}

#[test]
fn merging_pages_lays_them_out_right_to_left() {
    let (_dir, mut session, first) = session();
    apply(
        &mut session,
        vec![json!({"op": "remove_page", "page": first})],
    )
    .expect("an empty project");
    let right = scanned(&mut session, 4, 4, [255, 0, 0, 255]);
    let left = scanned(&mut session, 6, 3, [0, 0, 255, 255]);
    apply(
        &mut session,
        vec![
            json!({"op": "update_page", "page": right, "patch": {"name": "a", "chapter": "c1"}}),
            json!({"op": "update_page", "page": left, "patch": {"name": "b"}}),
        ],
    )
    .expect("names");
    let paint = add(&mut session, right, None, json!({"kind": "raster"}));
    let shape = add(
        &mut session,
        right,
        None,
        json!({"kind": "shape", "path": "M0 0 L1 0 L1 1 Z"}),
    );
    let before = session.project().clone();

    for pages in [json!([right, right]), json!([right])] {
        let error = apply(
            &mut session,
            vec![json!({"op": "merge_pages", "pages": pages, "rtl": true})],
        )
        .err()
        .expect("refused");
        assert!(matches!(error, Error::Invalid { .. }), "{pages}: {error}");
    }

    apply(
        &mut session,
        vec![json!({"op": "merge_pages", "pages": [left, right], "rtl": true})],
    )
    .expect("a spread");
    let project = session.project();
    let spread = project.pages().next().expect("one page").clone();
    assert_eq!(project.pages().count(), 1);
    assert_eq!(
        (spread.name.as_str(), spread.chapter.as_deref()),
        ("a + b", Some("c1"))
    );
    assert_eq!((spread.size.width, spread.size.height), (10, 4));
    let scan = session
        .file(&spread.background.expect("a background").file)
        .expect("the scan");
    let scan = image::load_from_memory(&scan).expect("a PNG").to_rgba8();
    assert_eq!(
        scan.get_pixel(0, 0).0,
        [0, 0, 255, 255],
        "the second page on the left"
    );
    assert_eq!(scan.get_pixel(0, 3).0, [0, 0, 0, 0], "below a shorter page");
    assert_eq!(
        scan.get_pixel(6, 3).0,
        [255, 0, 0, 255],
        "the first page on the right"
    );

    let [copied_paint, copied_shape] = &spread.layers[..] else {
        panic!("two layers, not {}", spread.layers.len())
    };
    assert!(
        copied_paint.id != paint && copied_shape.id != shape,
        "new ids"
    );
    let Content::Raster(raster) = &copied_paint.content else {
        panic!("not a raster")
    };
    assert_eq!((raster.offset.x, raster.offset.y), (6, 0));
    let Content::Shape(moved) = &copied_shape.content else {
        panic!("not a shape")
    };
    assert_eq!(
        kurbo::BezPath::from(&moved.path).elements()[0],
        kurbo::PathEl::MoveTo((6.0, 0.0).into())
    );

    session.undo().expect("undo").expect("a step");
    assert_eq!(session.project(), before);
}

#[test]
fn a_layer_copies_to_the_top_of_another_page() {
    let (_dir, mut session, page) = session();
    let other = common::page(&mut session);
    let group = add(&mut session, page, None, json!({"kind": "group"}));
    let mask = add(&mut session, page, Some(group), json!({"kind": "mask"}));
    add(&mut session, other, None, json!({"kind": "raster"}));

    apply(
        &mut session,
        vec![json!({"op": "duplicate_layer", "layer": group, "page": other})],
    )
    .expect("a copy");
    let project = session.project();
    let copy = project
        .page(other)
        .expect("the page")
        .layers
        .last()
        .expect("on top");
    assert!(copy.id != group && copy.children[0].id != mask, "new ids");
    assert_eq!(project.page(page).expect("the page").layers.len(), 1);

    let error = apply(
        &mut session,
        vec![json!({"op": "duplicate_layer", "layer": mask, "page": other})],
    )
    .err()
    .expect("a mask never goes at the top level");
    assert!(matches!(error, Error::Invalid { .. }), "{error}");
}
