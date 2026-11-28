//! The projects the start view lists: `.khr` files in one folder, and the
//! `.khrproj` folders Koharu 0.83 and NeKoharu up to 0.85.3-N kept in
//! Documents\Koharu, each of which becomes a `.khr` file here when opened.

use std::{
    collections::HashSet,
    path::{Path, PathBuf},
    time::SystemTime,
};

use anyhow::{Context as _, Result, bail, ensure};
use koharu_renderer::RendererConfig;
use koharu_scene::{EXTENSION, Meta, Session};
use sanitize_filename::OptionsForCheck;

mod khrproj;

const OLD_EXTENSION: &str = "khrproj";

/// The projects folder.
pub struct Library {
    root: PathBuf,
    /// Where the old releases kept their project folders; only read, and a
    /// folder there moves to the trash when its project is deleted.
    old: Option<PathBuf>,
}

impl Library {
    pub fn new(root: PathBuf, old: Option<PathBuf>) -> Result<Self> {
        fs_err::create_dir_all(&root)?;
        Ok(Self { root, old })
    }

    /// The projects' names, the last changed first; two changed at once in
    /// name order, numbers by value. An old folder whose name no file here
    /// has is listed too.
    pub fn list(&self) -> Result<Vec<String>> {
        let mut projects = entries(&self.root, EXTENSION, true)?;
        let names: HashSet<String> = projects.iter().map(|(_, name)| name.clone()).collect();
        if let Some(old) = self.old.as_deref().filter(|old| old.is_dir()) {
            projects.extend(
                entries(old, OLD_EXTENSION, false)?
                    .into_iter()
                    .filter(|(_, name)| !names.contains(name)),
            );
        }
        projects.sort_by(|(a_changed, a), (b_changed, b)| {
            b_changed
                .cmp(a_changed)
                .then_with(|| alphanumeric_sort::compare_str(a, b))
        });
        Ok(projects.into_iter().map(|(_, name)| name).collect())
    }

    pub fn create(&self, name: &str) -> Result<Session> {
        let path = self.path(name)?;
        if path.exists() || self.old_folder(name).is_some() {
            bail!("a project named {name:?} already exists");
        }
        Session::create(&path, preferred()?)
            .with_context(|| format!("failed to create {}", path.display()))
    }

    /// The project `name`. An old folder of that name with no file here
    /// becomes one first; the folder stays as it was.
    pub fn open(&self, name: &str) -> Result<Session> {
        let path = self.path(name)?;
        if !path.exists()
            && let Some(old) = self.old_folder(name)
        {
            khrproj::migrate(&old, &path, preferred()?)
                .with_context(|| format!("failed to upgrade {}", old.display()))?;
        }
        Session::open(&path, preferred()?)
            .with_context(|| format!("failed to open {}", path.display()))
    }

    /// The project `name`, created when there's none.
    pub fn open_or_create(&self, name: &str) -> Result<Session> {
        if self.path(name)?.exists() || self.old_folder(name).is_some() {
            self.open(name)
        } else {
            self.create(name)
        }
    }

    /// Deletes the project `name`. Its old folder, which would list it again,
    /// goes to the trash, since an old release may still open it.
    pub fn delete(&self, name: &str) -> Result<()> {
        let path = self.path(name)?;
        if path.exists() {
            Session::remove(&path)
                .with_context(|| format!("failed to delete {}", path.display()))?;
        }
        if let Some(old) = self.old_folder(name) {
            trash::delete(&old).with_context(|| format!("failed to trash {}", old.display()))?;
        }
        Ok(())
    }

    /// A project's file, for a name that is a file name on every platform.
    fn path(&self, name: &str) -> Result<PathBuf> {
        ensure!(!name.trim().is_empty(), "a project name can't be empty");
        let options = OptionsForCheck {
            windows: true,
            truncate: true,
        };
        ensure!(
            sanitize_filename::is_sanitized_with_options(name, options),
            "{name:?} can't be a file name"
        );
        Ok(self.root.join(format!("{name}.{EXTENSION}")))
    }

    /// The old release's folder of the project `name`, if there is one.
    fn old_folder(&self, name: &str) -> Option<PathBuf> {
        let folder = self.old.as_ref()?.join(format!("{name}.{OLD_EXTENSION}"));
        folder.is_dir().then_some(folder)
    }
}

/// The names in `folder` with `extension`, files or folders, and when each
/// last changed.
fn entries(
    folder: &Path,
    extension: &str,
    files: bool,
) -> Result<Vec<(Option<SystemTime>, String)>> {
    Ok(fs_err::read_dir(folder)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry
                .file_type()
                .is_ok_and(|kind| if files { kind.is_file() } else { kind.is_dir() })
        })
        .filter_map(|entry| {
            let path = entry.path();
            let name = path
                .extension()
                .is_some_and(|each| each.eq_ignore_ascii_case(extension))
                .then(|| Some(path.file_stem()?.to_str()?.to_owned()))
                .flatten()?;
            let changed = entry.metadata().and_then(|metadata| metadata.modified());
            Some((changed.ok(), name))
        })
        .collect())
}

/// What a new project holds, and a file from before projects held their own
/// styles takes: Preferences' styles, as a new InDesign document takes the
/// styles made with none open.
fn preferred() -> Result<Meta> {
    let config: RendererConfig = koharu_config::get()?;
    let mut meta = Meta::default();
    for (kernel, faces) in config.styles {
        meta.styles.insert(kernel, faces.into_iter().collect());
    }
    Ok(meta)
}
