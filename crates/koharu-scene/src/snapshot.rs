//! A page as an edit that makes it again, ids and pixels included, for a
//! project elsewhere to replay, as a remote run's server gets the pages it
//! runs on.

use serde_json::json;

use crate::{
    Applied, Content, Edit, Layer, LayerId, NewLayer, Op, PageId, Rect, Ref, Result, Session, Tile,
};

impl Session {
    /// An edit that adds `page` at `index` as it is here, with its file,
    /// layers, pictures, and locks, under its own ids.
    pub fn snapshot(&self, page: PageId, index: usize) -> Result<Applied> {
        let project = self.project();
        let page = project.page(page)?;
        let mut ops = Vec::new();
        if let Some(background) = &page.background {
            ops.push(Op::AddFile {
                bytes: self.file(&background.file)?,
            });
        }
        ops.push(Op::AddPage {
            id: None,
            index,
            // A page made from its file takes the file's size.
            size: page.background.is_none().then_some(page.size),
            background: page.background.map(|background| background.file),
        });
        let at = Ref::from(page.id);
        if page
            .background
            .is_some_and(|background| !background.visible)
        {
            ops.push(Op::SetBackground {
                page: at.clone(),
                background: page.background,
            });
        }
        if !page.name.is_empty() || page.chapter.is_some() {
            ops.push(Op::UpdatePage {
                page: at.clone(),
                patch: json!({"name": page.name, "chapter": page.chapter}),
            });
        }
        let mut ids = vec![page.id.0];
        let mut locks = Vec::new();
        for (index, layer) in page.layers.iter().enumerate() {
            self.add(&at, None, index, layer, &mut ops, &mut ids, &mut locks)?;
        }
        // Locked last, children before what holds them, since nothing is
        // written inside a lock.
        ops.extend(locks);
        Ok(Applied {
            edit: Edit {
                label: "snapshot".to_owned(),
                ops,
            },
            ids,
        })
    }

    /// Adds `layer` and everything inside it, unlocked, with its pictures,
    /// and leaves its locks in `locks`.
    #[allow(clippy::too_many_arguments)]
    fn add(
        &self,
        page: &Ref,
        parent: Option<LayerId>,
        index: usize,
        layer: &Layer,
        ops: &mut Vec<Op>,
        ids: &mut Vec<crate::Uuid>,
        locks: &mut Vec<Op>,
    ) -> Result<()> {
        let mut content = layer.content.clone();
        let mut part_locks = Vec::new();
        if let Content::Block(block) = &mut content {
            for (part, locked) in [
                ("mask", &mut block.mask.locked),
                ("cleanup", &mut block.cleanup.locked),
            ] {
                if std::mem::take(locked) {
                    part_locks.push(json!({part: {"locked": true}}));
                }
            }
        }
        ops.push(Op::AddLayer {
            id: None,
            page: page.clone(),
            parent: parent.map(Ref::from),
            index,
            layer: Box::new(NewLayer {
                name: layer.name.clone(),
                visible: layer.visible,
                locked: false,
                opacity: layer.opacity,
                blend: Some(layer.blend),
                attributes: layer.attributes.clone(),
                content,
            }),
        });
        ids.push(layer.id.0);
        let pictures = match &layer.content {
            // The order `add_layer` numbers a block's parts in.
            Content::Block(block) => vec![block.mask.id, block.cleanup.id],
            Content::Raster(_) | Content::Mask(_) => vec![layer.id],
            _ => Vec::new(),
        };
        if let Content::Block(block) = &layer.content {
            ids.extend([block.mask.id.0, block.cleanup.id.0]);
        }
        for picture in pictures {
            let tiles = self.tiles(picture, None)?;
            let side = Tile::SIZE;
            for tile in tiles.tiles {
                ops.push(Op::WritePixels {
                    layer: Ref::from(picture),
                    rect: Rect {
                        x: tile.x * side as i32,
                        y: tile.y * side as i32,
                        width: side,
                        height: side,
                    },
                    pixels: tile.pixels,
                });
            }
        }
        for (index, child) in layer.children.iter().enumerate() {
            self.add(page, Some(layer.id), index, child, ops, ids, locks)?;
        }
        let at = Ref::from(layer.id);
        for patch in part_locks {
            locks.push(Op::UpdateLayer {
                layer: at.clone(),
                patch,
            });
        }
        if layer.locked {
            locks.push(Op::UpdateLayer {
                layer: at,
                patch: json!({"locked": true}),
            });
        }
        Ok(())
    }
}
