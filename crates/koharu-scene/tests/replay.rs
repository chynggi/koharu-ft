//! Edits recorded on one project and replayed on another holding the same
//! pages, as a remote run's are on the desktop's.

mod common;

use common::*;
use koharu_scene::{Applied, Error, Page, Rect, Session};
use serde_json::json;

/// Every page, in order.
fn pages(session: &Session) -> Vec<Page> {
    session.project().pages().cloned().collect()
}

#[test]
fn a_replayed_edit_makes_the_same_ids_and_pixels() {
    let (_here, mut here) = {
        let (dir, path) = path();
        (
            dir,
            Session::create(path, koharu_scene::Meta::default()).expect("a project"),
        )
    };
    let (_there, mut there) = {
        let (dir, path) = path();
        (
            dir,
            Session::create(path, koharu_scene::Meta::default()).expect("a project"),
        )
    };
    there.record();
    // Every way an edit makes an id: a page and layers with none given, a
    // label, a block's parts, a duplicate's children, and a merged spread's
    // copies; and pixels written to a part by its label.
    apply(
        &mut there,
        vec![
            json!({"op": "add_page", "id": "$a", "index": 0, "size": {"width": 64, "height": 64}}),
            json!({"op": "add_page", "index": 1, "size": {"width": 32, "height": 64}}),
            json!({"op": "add_layer", "id": "$group", "page": "$a", "index": 0, "layer": {"kind": "group"}}),
            json!({"op": "add_layer", "page": "$a", "parent": "$group", "index": 0, "layer": {"kind": "raster"}}),
            json!({"op": "add_layer", "id": "$b", "page": "$a", "index": 1, "layer": {"kind": "block", "source": {}}}),
            json!({"op": "write_pixels", "layer": "$b.cleanup",
                "rect": {"x": 0, "y": 0, "width": 2, "height": 1}, "pixels": [9, 8, 7, 255, 1, 2, 3, 4]}),
            json!({"op": "duplicate_layer", "layer": "$group"}),
        ],
    )
    .expect("the setup");
    let order: Vec<_> = pages(&there).iter().map(|page| page.id).collect();
    apply(
        &mut there,
        vec![json!({"op": "merge_pages", "pages": order, "rtl": false})],
    )
    .expect("a merge");

    let recorded = there.recorded();
    assert_eq!(recorded.len(), 2);
    assert!(there.recorded().is_empty(), "handed over once");
    for applied in recorded {
        // Over the wire and back, as a remote run sends it.
        let wire = serde_json::to_string(&applied).expect("JSON");
        here.replay(serde_json::from_str(&wire).expect("an edit"))
            .expect("the replay");
    }
    assert_eq!(pages(&here), pages(&there));
    let block = pages(&there)[0]
        .layers
        .iter()
        .find_map(|layer| match &layer.content {
            koharu_scene::Content::Block(block) => Some(block.cleanup.id),
            _ => None,
        });
    let cleanup = block.expect("the spread holds the block's copy");
    let rect = Rect {
        x: 0,
        y: 0,
        width: 2,
        height: 1,
    };
    assert_eq!(
        here.pixels(cleanup, rect).expect("pixels"),
        there.pixels(cleanup, rect).expect("pixels")
    );
}

#[test]
fn a_replay_uses_exactly_the_ids_it_was_given() {
    let (_dir, mut session) = {
        let (dir, path) = path();
        (
            dir,
            Session::create(path, koharu_scene::Meta::default()).expect("a project"),
        )
    };
    session.record();
    apply(
        &mut session,
        vec![json!({"op": "add_page", "index": 0, "size": {"width": 8, "height": 8}})],
    )
    .expect("a page");
    let Applied { edit, ids } = session.recorded().remove(0);
    let replay = |session: &mut Session, ids| {
        session.replay(Applied {
            edit: edit.clone(),
            ids,
        })
    };
    // The page it named is there already.
    assert!(matches!(
        replay(&mut session, ids.clone()),
        Err(Error::Invalid { .. })
    ));
    let fresh = koharu_scene::Uuid::now_v7();
    assert!(matches!(
        replay(&mut session, Vec::new()),
        Err(Error::Invalid { .. })
    ));
    assert!(matches!(
        replay(&mut session, vec![fresh, koharu_scene::Uuid::now_v7()]),
        Err(Error::Invalid { .. })
    ));
    replay(&mut session, vec![fresh]).expect("a replay");
    // At index 0, as the edit says, before the first.
    assert_eq!(
        session.project().pages().next().map(|page| page.id.0),
        Some(fresh)
    );
}

#[test]
fn a_page_snapshot_replays_as_the_same_page() {
    let (_there, mut there) = {
        let (dir, path) = path();
        (
            dir,
            Session::create(path, koharu_scene::Meta::default()).expect("a project"),
        )
    };
    let mut png = Vec::new();
    image::RgbaImage::from_pixel(64, 64, image::Rgba([1, 2, 3, 255]))
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .expect("a PNG");
    let hash = koharu_scene::Hash::of(&png);
    apply(
        &mut there,
        vec![
            json!({"op": "add_file", "bytes": png}),
            json!({"op": "add_page", "id": "$p", "index": 0, "background": hash}),
            json!({"op": "update_page", "page": "$p", "patch": {"name": "cover", "chapter": "1"}}),
            json!({"op": "set_background", "page": "$p", "background": {"file": hash, "visible": false}}),
            json!({"op": "add_layer", "id": "$group", "page": "$p", "index": 0, "layer": {"kind": "group", "name": "art"}}),
            json!({"op": "add_layer", "id": "$raster", "page": "$p", "parent": "$group", "index": 0,
                "layer": {"kind": "raster", "offset": {"x": -3, "y": 70}, "opacity": 0.5}}),
            json!({"op": "write_pixels", "layer": "$raster",
                "rect": {"x": 60, "y": -2, "width": 8, "height": 4}, "pixels": vec![200u8; 8 * 4 * 4]}),
            json!({"op": "add_layer", "id": "$b", "page": "$p", "index": 1,
                "layer": {"kind": "block", "source": {"content": "やあ"}, "translation": {"content": "Hi"}}}),
            json!({"op": "write_pixels", "layer": "$b.mask",
                "rect": {"x": 0, "y": 0, "width": 2, "height": 2}, "pixels": [255, 0, 0, 255]}),
            json!({"op": "update_layer", "layer": "$b", "patch": {"mask": {"locked": true}}}),
            json!({"op": "update_layer", "layer": "$raster", "patch": {"locked": true}}),
            json!({"op": "update_layer", "layer": "$group", "patch": {"locked": true}}),
        ],
    )
    .expect("a page");
    let page = pages(&there)[0].id;
    let snapshot = there.snapshot(page, 0).expect("a snapshot");

    let (_here, mut here) = {
        let (dir, path) = path();
        (
            dir,
            Session::create(path, koharu_scene::Meta::default()).expect("a project"),
        )
    };
    here.replay(snapshot).expect("the replay");
    assert_eq!(pages(&here), pages(&there));
    let page = &pages(&there)[0];
    let raster = page.layers[0].children[0].id;
    let koharu_scene::Content::Block(block) = &page.layers[1].content else {
        panic!("a block");
    };
    for picture in [raster, block.mask.id] {
        assert_eq!(
            here.tiles(picture, None).expect("tiles"),
            there.tiles(picture, None).expect("tiles")
        );
    }
    assert_eq!(
        here.file(&hash).expect("the file"),
        there.file(&hash).expect("the file")
    );
}
