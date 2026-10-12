use std::collections::{BTreeMap, BTreeSet};

use koharu_scene::EntityId;

use crate::Stage;

#[derive(Clone, Copy, Eq, PartialEq)]
enum WorkState {
    Pending,
    Running,
    Finished,
}

struct StageWork {
    stage: Stage,
    state: WorkState,
}

struct PageWork {
    page: EntityId,
    stages: Vec<StageWork>,
}

impl PageWork {
    fn started(&self) -> bool {
        self.stages
            .iter()
            .any(|work| work.state != WorkState::Pending)
    }

    fn finished(&self) -> bool {
        self.stages
            .iter()
            .all(|work| work.state == WorkState::Finished)
    }

    fn ready(&self, index: usize) -> bool {
        let Some(prerequisite) = prerequisite(self.stages[index].stage) else {
            return true;
        };
        self.stages
            .iter()
            .find(|work| work.stage == prerequisite)
            .is_none_or(|work| work.state == WorkState::Finished)
    }
}

pub(crate) struct Scheduler {
    pages: Vec<PageWork>,
    page_index: BTreeMap<EntityId, usize>,
    page_window: usize,
    active_pages: usize,
    head: usize,
    total: usize,
    translation_batch_size: usize,
}

impl Scheduler {
    pub(crate) fn new(pages: &[EntityId], stages: &[Stage], translation_batch_size: usize) -> Self {
        let pages = pages
            .iter()
            .map(|page| PageWork {
                page: *page,
                stages: stages
                    .iter()
                    .map(|stage| StageWork {
                        stage: *stage,
                        state: WorkState::Pending,
                    })
                    .collect(),
            })
            .collect::<Vec<_>>();
        let total = pages.len().saturating_mul(stages.len());
        Self {
            page_index: pages
                .iter()
                .enumerate()
                .map(|(index, page)| (page.page, index))
                .collect(),
            pages,
            page_window: stages.len().max(1).max(translation_batch_size),
            active_pages: 0,
            head: 0,
            total,
            translation_batch_size: translation_batch_size.max(1),
        }
    }

    pub(crate) fn total(&self) -> usize {
        self.total
    }

    pub(crate) fn start_next(
        &mut self,
        busy_stages: &BTreeSet<Stage>,
    ) -> Option<(Vec<EntityId>, Stage)> {
        for page_index in self.head..self.pages.len() {
            let stage_index =
                self.pages[page_index]
                    .stages
                    .iter()
                    .enumerate()
                    .find_map(|(index, work)| {
                        (work.state == WorkState::Pending
                            && !busy_stages.contains(&work.stage)
                            && self.pages[page_index].ready(index))
                        .then_some(index)
                    });
            let Some(stage_index) = stage_index else {
                continue;
            };
            let stage = self.pages[page_index].stages[stage_index].stage;
            let batch_start =
                page_index / self.translation_batch_size * self.translation_batch_size;
            let batch_end = batch_start
                .saturating_add(self.translation_batch_size)
                .min(self.pages.len());
            if stage == Stage::Translation && page_index != batch_start {
                continue;
            }
            let batch = if stage == Stage::Translation {
                batch_start..batch_end
            } else {
                page_index..page_index + 1
            };
            let new_pages = batch
                .clone()
                .filter(|index| !self.pages[*index].started())
                .count();
            if self.active_pages + new_pages > self.page_window {
                continue;
            }
            if batch.clone().any(|index| {
                let page = &self.pages[index];
                page.stages[stage_index].state != WorkState::Pending || !page.ready(stage_index)
            }) {
                continue;
            }
            let pages = batch
                .map(|index| {
                    let page = &mut self.pages[index];
                    if !page.started() {
                        self.active_pages += 1;
                    }
                    page.stages[stage_index].state = WorkState::Running;
                    page.page
                })
                .collect();
            return Some((pages, stage));
        }
        None
    }

    pub(crate) fn complete_stage(&mut self, page: EntityId, stage: Stage) -> bool {
        let Some(&page_index) = self.page_index.get(&page) else {
            return false;
        };
        let page = &mut self.pages[page_index];
        let was_finished = page.finished();
        if let Some(work) = page.stages.iter_mut().find(|work| work.stage == stage) {
            work.state = WorkState::Finished;
        }
        let page_finished = !was_finished && page.finished();
        if page_finished {
            self.active_pages = self.active_pages.saturating_sub(1);
            while self.head < self.pages.len() && self.pages[self.head].finished() {
                self.head += 1;
            }
        }
        page_finished
    }
}

const fn prerequisite(stage: Stage) -> Option<Stage> {
    match stage {
        Stage::Detection => None,
        Stage::Ocr | Stage::Inpainting => Some(Stage::Detection),
        Stage::Translation => Some(Stage::Ocr),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pages(count: usize) -> Vec<EntityId> {
        (0..count).map(|_| EntityId::new()).collect()
    }

    #[test]
    fn starts_pages_in_order_and_models_independently() {
        let pages = pages(2);
        let mut scheduler = Scheduler::new(&pages, &Stage::ALL, 1);
        let mut busy = BTreeSet::new();

        let first = scheduler.start_next(&busy).unwrap();
        assert_eq!(first, (vec![pages[0]], Stage::Detection));
        busy.insert(Stage::Detection);
        assert!(scheduler.start_next(&busy).is_none());

        busy.clear();
        assert!(!scheduler.complete_stage(pages[0], Stage::Detection));
        let ocr = scheduler.start_next(&busy).unwrap();
        busy.insert(ocr.1);
        let inpainting = scheduler.start_next(&busy).unwrap();
        busy.insert(inpainting.1);
        let next_page = scheduler.start_next(&busy).unwrap();
        busy.insert(next_page.1);
        assert_eq!(ocr, (vec![pages[0]], Stage::Ocr));
        assert_eq!(inpainting, (vec![pages[0]], Stage::Inpainting));
        assert_eq!(next_page, (vec![pages[1]], Stage::Detection));

        assert!(!scheduler.complete_stage(pages[0], Stage::Ocr));
        busy.remove(&Stage::Ocr);
        let translation = scheduler.start_next(&busy).unwrap();
        assert_eq!(translation, (vec![pages[0]], Stage::Translation));
        assert!(busy.contains(&Stage::Detection));
        assert!(busy.contains(&Stage::Inpainting));
    }

    #[test]
    fn sliding_window_backpressures_fast_upstream_models() {
        let pages = pages(4);
        let stages = [Stage::Detection, Stage::Ocr, Stage::Inpainting];
        let mut scheduler = Scheduler::new(&pages, &stages, 1);
        let mut busy = BTreeSet::new();

        assert_eq!(
            scheduler.start_next(&busy),
            Some((vec![pages[0]], Stage::Detection))
        );
        assert!(!scheduler.complete_stage(pages[0], Stage::Detection));
        let ocr = scheduler.start_next(&busy).unwrap();
        busy.insert(ocr.1);
        let inpainting = scheduler.start_next(&busy).unwrap();
        busy.insert(inpainting.1);

        for page in &pages[1..3] {
            assert_eq!(
                scheduler.start_next(&busy),
                Some((vec![*page], Stage::Detection))
            );
            assert!(!scheduler.complete_stage(*page, Stage::Detection));
        }
        assert!(scheduler.start_next(&busy).is_none());

        assert!(!scheduler.complete_stage(pages[0], Stage::Ocr));
        assert!(scheduler.complete_stage(pages[0], Stage::Inpainting));
        busy.clear();
        assert_eq!(
            scheduler.start_next(&busy),
            Some((vec![pages[1]], Stage::Ocr))
        );
    }

    #[test]
    fn batches_translation_only_after_every_page_in_the_group_is_ready() {
        let pages = pages(3);
        let mut scheduler = Scheduler::new(&pages, &[Stage::Ocr, Stage::Translation], 2);
        let busy = BTreeSet::new();

        assert_eq!(
            scheduler.start_next(&busy),
            Some((vec![pages[0]], Stage::Ocr))
        );
        assert!(!scheduler.complete_stage(pages[0], Stage::Ocr));
        assert_eq!(
            scheduler.start_next(&busy),
            Some((vec![pages[1]], Stage::Ocr))
        );
        assert!(!scheduler.complete_stage(pages[1], Stage::Ocr));
        assert_eq!(
            scheduler.start_next(&busy),
            Some((vec![pages[0], pages[1]], Stage::Translation))
        );
        assert!(scheduler.complete_stage(pages[0], Stage::Translation));
        assert!(scheduler.complete_stage(pages[1], Stage::Translation));
        assert_eq!(
            scheduler.start_next(&busy),
            Some((vec![pages[2]], Stage::Ocr))
        );
    }
}
