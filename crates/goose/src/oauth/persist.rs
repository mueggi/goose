use rmcp::transport::auth::{AuthError, CredentialStore, StoredCredentials};
use serde::{Deserialize, Serialize};

use crate::config::Config;

#[derive(Serialize, Deserialize)]
struct PersistedCredentials {
    #[serde(flatten)]
    credentials: StoredCredentials,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    requested_scopes: Option<Vec<String>>,
}

/// Goose-specific credential store that uses the Config system
///
/// This implementation stores OAuth credentials in the goose configuration
/// system, which handles secure storage (e.g., keychain integration).

#[derive(Clone)]
pub struct GooseCredentialStore {
    name: String,
}

impl GooseCredentialStore {
    pub fn new(name: String) -> Self {
        Self { name }
    }

    fn secret_key(&self) -> String {
        format!("oauth_creds_{}", self.name)
    }

    /// Whether credentials are currently stored, without going through the
    /// async `CredentialStore::load`. Intended for synchronous "is this
    /// provider configured" checks (e.g. inventory registrations) that would
    /// otherwise have to duplicate the `oauth_creds_{name}` key format.
    pub fn has_credentials(&self) -> bool {
        matches!(self.load_persisted(), Ok(Some(_)))
    }

    fn load_persisted(&self) -> Result<Option<PersistedCredentials>, AuthError> {
        let config = Config::global();
        let key = self.secret_key();

        match config.get_secret::<PersistedCredentials>(&key) {
            Ok(credentials) => Ok(Some(credentials)),
            Err(_) => Ok(None),
        }
    }

    fn save_persisted(&self, credentials: PersistedCredentials) -> Result<(), AuthError> {
        let config = Config::global();
        let key = self.secret_key();

        config
            .set_secret(&key, &credentials)
            .map_err(|e| AuthError::InternalError(format!("Failed to save credentials: {}", e)))
    }

    pub fn load_requested_scopes(&self) -> Result<Option<Vec<String>>, AuthError> {
        Ok(self
            .load_persisted()?
            .and_then(|credentials| credentials.requested_scopes))
    }

    pub fn save_with_requested_scopes(
        &self,
        credentials: StoredCredentials,
        requested_scopes: Option<Vec<String>>,
    ) -> Result<(), AuthError> {
        self.save_persisted(PersistedCredentials {
            credentials,
            requested_scopes,
        })
    }
}

#[async_trait::async_trait]
impl CredentialStore for GooseCredentialStore {
    async fn load(&self) -> Result<Option<StoredCredentials>, AuthError> {
        Ok(self
            .load_persisted()?
            .map(|credentials| credentials.credentials))
    }

    async fn save(&self, credentials: StoredCredentials) -> Result<(), AuthError> {
        let requested_scopes = self.load_requested_scopes()?;
        self.save_with_requested_scopes(credentials, requested_scopes)
    }

    async fn clear(&self) -> Result<(), AuthError> {
        let config = Config::global();
        let key = self.secret_key();

        config
            .delete_secret(&key)
            .map_err(|e| AuthError::InternalError(format!("Failed to clear credentials: {}", e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Pins the `oauth_creds_{name}` naming convention so callers that need a
    // synchronous "is this provider configured" check (e.g. inventory
    // registrations) can rely on `has_credentials()` instead of
    // reconstructing this key themselves. `Config::global()` is a
    // process-wide singleton this crate's tests otherwise avoid touching
    // (see config::base's tests, which always build a scoped `Config`
    // instead), so this test only pins the key format, not the actual
    // load/save round-trip through Config::global().
    #[test]
    fn secret_key_uses_oauth_creds_prefix() {
        let store = GooseCredentialStore::new("golem".to_string());
        assert_eq!(store.secret_key(), "oauth_creds_golem");
    }

    fn credentials() -> StoredCredentials {
        StoredCredentials::new(
            "client-id".to_string(),
            None,
            vec!["scope.read".to_string()],
            Some(123),
        )
    }

    #[test]
    fn persisted_credentials_read_the_legacy_shape() {
        let legacy = serde_json::to_value(credentials()).unwrap();
        let persisted: PersistedCredentials = serde_json::from_value(legacy).unwrap();

        assert_eq!(persisted.credentials.client_id, "client-id");
        assert_eq!(persisted.credentials.granted_scopes, vec!["scope.read"]);
        assert_eq!(persisted.requested_scopes, None);
    }

    #[test]
    fn persisted_credentials_remain_readable_as_stored_credentials() {
        let persisted = PersistedCredentials {
            credentials: credentials(),
            requested_scopes: Some(vec!["scope.read".to_string(), "scope.write".to_string()]),
        };
        let value = serde_json::to_value(persisted).unwrap();
        let credentials: StoredCredentials = serde_json::from_value(value).unwrap();

        assert_eq!(credentials.client_id, "client-id");
        assert_eq!(credentials.granted_scopes, vec!["scope.read"]);
    }
}
