//! The projects the start view lists: `.khr` files in one folder, and the
//! `.khrproj` folders of older releases there, which become `.khr` files
//! when opened.

use std::{path::PathBuf, time::SystemTime};

use anyhow::{Context as _, Result, bail, ensure};
use koharu_renderer::RendererConfig;
use koharu_scene::{EXTENSION, Meta, Session};
use sanitize_filename::OptionsForCheck;

mod khrproj;

/// The project folders of Koharu 0.83 and NeKoharu up to 0.85.3-N.
const OLD_EXTENSION: &str = "khrproj";

/// The projects folder.
pub struct Library {
    root: PathBuf,
}

impl Library {
    pub fn new(root: PathBuf) -> Result<Self> {
        fs_err::create_dir_all(&root)?;
        Ok(Self { root })
    }

    /// The projects' names, the last changed first; two changed at once in
    /// name order, numbers by value. An old folder with no file of its name
    /// is listed too.
    pub fn list(&self) -> Result<Vec<String>> {
        let mut projects: Vec<(Option<SystemTime>, String)> = fs_err::read_dir(&self.root)?
            .filter_map(|entry| entry.ok())
            .filter_map(|entry| {
                let path = entry.path();
                let file = entry.file_type().is_ok_and(|kind| kind.is_file());
                let extension = if file { EXTENSION } else { OLD_EXTENSION };
                let name = path
                    .extension()
                    .is_some_and(|each| each.eq_ignore_ascii_case(extension))
                    .then(|| Some(path.file_stem()?.to_str()?.to_owned()))
                    .flatten()?;
                if !file && path.with_extension(EXTENSION).exists() {
                    return None;
                }
                let changed = entry.metadata().and_then(|metadata| metadata.modified());
                Some((changed.ok(), name))
            })
            .collect();
        projects.sort_by(|(a_changed, a), (b_changed, b)| {
            b_changed
                .cmp(a_changed)
                .then_with(|| alphanumeric_sort::compare_str(a, b))
        });
        Ok(projects.into_iter().map(|(_, name)| name).collect())
    }

    pub fn create(&self, name: &str) -> Result<Session> {
        let path = self.path(name)?;
        if path.exists() || path.with_extension(OLD_EXTENSION).exists() {
            bail!("a project named {name:?} already exists");
        }
        Session::create(&path, preferred()?)
            .with_context(|| format!("failed to create {}", path.display()))
    }

    /// The project `name`. An old folder of that name with no file becomes
    /// one first; the folder stays as it was.
    pub fn open(&self, name: &str) -> Result<Session> {
        let path = self.path(name)?;
        let old = path.with_extension(OLD_EXTENSION);
        if !path.exists() && old.is_dir() {
            khrproj::migrate(&old, &path, preferred()?)
                .with_context(|| format!("failed to upgrade {}", old.display()))?;
        }
        Session::open(&path, preferred()?)
            .with_context(|| format!("failed to open {}", path.display()))
    }

    /// The project `name`, created when there's none.
    pub fn open_or_create(&self, name: &str) -> Result<Session> {
        let path = self.path(name)?;
        if path.exists() || path.with_extension(OLD_EXTENSION).is_dir() {
            self.open(name)
        } else {
            self.create(name)
        }
    }

    /// Deletes the project `name`, its old folder too, which would otherwise
    /// list it again.
    pub fn delete(&self, name: &str) -> Result<()> {
        let path = self.path(name)?;
        let old = path.with_extension(OLD_EXTENSION);
        if path.exists() {
            Session::remove(&path)
                .with_context(|| format!("failed to delete {}", path.display()))?;
        }
        if old.is_dir() {
            fs_err::remove_dir_all(&old)?;
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
