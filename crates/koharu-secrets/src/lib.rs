use secrecy::{SecretBox, zeroize::Zeroize};
use serde::{Deserialize, Serialize, Serializer};

pub use secrecy::{ExposeSecret, SerializableSecret};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(transparent)]
pub struct SecretString(SecretBox<SecretValue>);

impl Default for SecretString {
    fn default() -> Self {
        Self::from(String::new())
    }
}

impl From<String> for SecretString {
    fn from(value: String) -> Self {
        Self(SecretBox::new(Box::new(SecretValue(value))))
    }
}

impl From<&str> for SecretString {
    fn from(value: &str) -> Self {
        Self::from(value.to_owned())
    }
}

impl ExposeSecret<str> for SecretString {
    fn expose_secret(&self) -> &str {
        &self.0.expose_secret().0
    }
}

#[derive(Clone, Deserialize)]
#[serde(transparent)]
struct SecretValue(String);

impl Zeroize for SecretValue {
    fn zeroize(&mut self) {
        self.0.zeroize();
    }
}

impl secrecy::CloneableSecret for SecretValue {}

impl Serialize for SecretValue {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str("[REDACTED]")
    }
}

impl SerializableSecret for SecretValue {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SecretKey<'a> {
    name: &'a str,
    environment_variable: Option<&'a str>,
}

impl<'a> SecretKey<'a> {
    #[must_use]
    pub const fn environment(name: &'a str, environment_variable: &'a str) -> Self {
        Self {
            name,
            environment_variable: Some(environment_variable),
        }
    }

    #[must_use]
    pub const fn stored(name: &'a str) -> Self {
        Self {
            name,
            environment_variable: None,
        }
    }

    #[must_use]
    pub const fn name(self) -> &'a str {
        self.name
    }

    #[must_use]
    pub const fn environment_variable(self) -> Option<&'a str> {
        self.environment_variable
    }
}

#[must_use]
pub const fn is_read_only() -> bool {
    cfg!(target_os = "linux")
}

#[cfg(target_os = "linux")]
pub fn get(key: SecretKey<'_>) -> anyhow::Result<Option<SecretString>> {
    let variable = key.environment_variable().ok_or_else(|| {
        anyhow::anyhow!(
            "secret '{}' is unavailable on Linux; this deployment uses read-only environment credentials",
            key.name()
        )
    })?;
    environment_secret(variable, std::env::var_os(variable))
}

#[cfg(not(target_os = "linux"))]
pub fn get(key: SecretKey<'_>) -> anyhow::Result<Option<SecretString>> {
    match entry(key.name())?.get_password() {
        Ok(value) => Ok(Some(SecretString::from(value))),
        Err(keyring_core::Error::NoEntry) => Ok(None),
        Err(error) => Err(error.into()),
    }
}

#[cfg(target_os = "linux")]
pub fn set(key: SecretKey<'_>, _secret: &SecretString) -> anyhow::Result<()> {
    anyhow::bail!(
        "secret '{}' is managed by {} on Linux and cannot be changed from Koharu",
        key.name(),
        key.environment_variable().unwrap_or("the environment")
    )
}

#[cfg(not(target_os = "linux"))]
pub fn set(key: SecretKey<'_>, secret: &SecretString) -> anyhow::Result<()> {
    entry(key.name())?.set_password(secret.expose_secret())?;
    Ok(())
}

#[cfg(target_os = "linux")]
pub fn delete(key: SecretKey<'_>) -> anyhow::Result<()> {
    anyhow::bail!(
        "secret '{}' is managed by {} on Linux and cannot be cleared from Koharu",
        key.name(),
        key.environment_variable().unwrap_or("the environment")
    )
}

#[cfg(not(target_os = "linux"))]
pub fn delete(key: SecretKey<'_>) -> anyhow::Result<()> {
    match entry(key.name())?.delete_credential() {
        Ok(()) | Err(keyring_core::Error::NoEntry) => Ok(()),
        Err(error) => Err(error.into()),
    }
}

#[cfg(target_os = "linux")]
fn environment_secret(
    variable: &str,
    value: Option<std::ffi::OsString>,
) -> anyhow::Result<Option<SecretString>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let value = value
        .into_string()
        .map_err(|_| anyhow::anyhow!("{variable} contains non-Unicode data"))?;
    Ok((!value.trim().is_empty()).then(|| SecretString::from(value)))
}

#[cfg(not(target_os = "linux"))]
fn entry(key: &str) -> anyhow::Result<keyring_core::Entry> {
    const SERVICE: &str = "koharu";
    Ok(keyring::Entry::new(SERVICE, key)?.inner)
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;

    #[test]
    fn environment_values_are_loaded_without_exposing_them() {
        let secret = environment_secret("TEST_KEY", Some("  secret  ".into()))
            .unwrap()
            .unwrap();
        assert_eq!(secret.expose_secret(), "  secret  ");
        assert_eq!(serde_json::to_string(&secret).unwrap(), "\"[REDACTED]\"");
    }

    #[test]
    fn missing_and_blank_environment_values_are_absent() {
        assert!(environment_secret("TEST_KEY", None).unwrap().is_none());
        assert!(
            environment_secret("TEST_KEY", Some(" \t\n".into()))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn linux_secret_storage_is_read_only() {
        let key = SecretKey::environment("openai", "OPENAI_API_KEY");
        assert!(set(key, &SecretString::from("secret")).is_err());
        assert!(delete(key).is_err());
    }
}
