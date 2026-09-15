use super::api_client::{ApiClient, AuthMethod, AuthProvider};
use super::base::{ConfigKey, MessageStream, Provider, ProviderDef, ProviderMetadata};
use super::openai_compatible::OpenAiCompatibleProvider;
use crate::conversation::message::Message;
use crate::oauth::{oauth_flow, GooseCredentialStore};
use anyhow::Result;
use async_trait::async_trait;
use futures::future::BoxFuture;
use goose_providers::errors::ProviderError;
use goose_providers::model::ModelConfig;
use rmcp::model::Tool;
use rmcp::transport::auth::{AuthError, AuthorizationManager, CredentialStore};
use std::sync::Arc;
use tokio::sync::Mutex as TokioMutex;

/// Self-hosted inference gateway authenticated via discovery-driven
/// OAuth/OIDC (RFC 8414 authorization-server discovery + RFC 7591 dynamic
/// client registration + PKCE) instead of a static API key. The discovery
/// and token machinery is not reimplemented here: it is the same
/// `rmcp::transport::auth::AuthorizationManager` goose already uses for MCP
/// extension auth (see `crate::oauth::oauth_flow`), pointed at an inference
/// base URL instead of an MCP server URL.
pub const GOLEM_PROVIDER_NAME: &str = "golem";
pub const GOLEM_DEFAULT_HOST: &str = "https://agent.asanti.dev/golem/llm/v1";
const GOLEM_DOC_URL: &str = "https://agent.asanti.dev";

/// The `name` passed to `GooseCredentialStore`/`oauth_flow` — deliberately
/// distinct from `GOLEM_PROVIDER_NAME`. Both this provider and any MCP
/// extension the user configures go through the exact same
/// `crate::oauth::oauth_flow`/`GooseCredentialStore` machinery, keyed by a
/// flat string name with no tie to the resource/base_url being
/// authenticated. A user with an MCP extension also named "golem" (e.g.
/// pointing at `https://agent.asanti.dev/golem/mcp`) would silently share —
/// and clobber — this provider's cached token, since both would resolve to
/// the same keyring secret (`oauth_creds_golem`). Whichever flow
/// authenticated most recently would win, handing the other a token scoped
/// to the wrong resource with no error until first use. Using a
/// provider-specific credential name avoids the collision entirely.
pub(crate) const GOLEM_OAUTH_CREDENTIAL_NAME: &str = "golem-llm-provider";

/// The URL used to *discover* OAuth (RFC 8414/9728), as opposed to
/// `base_url`, which is the OpenAI-compatible API root used for chat
/// completions and model listing.
///
/// These can't be the same URL here: an unauthenticated GET on the bare API
/// root (e.g. `https://agent.asanti.dev/golem/llm/v1`) returns a plain `303`
/// redirect to a UI page, not the `401` + `WWW-Authenticate:
/// resource_metadata=...` challenge RFC 9728 discovery needs. An actual
/// protected endpoint under that root — `/models`, which every
/// OpenAI-compatible provider exposes — does return that challenge, and its
/// protected-resource metadata is scoped to the API root regardless of which
/// endpoint under it triggered the challenge. Discovering against the raw
/// `base_url` instead would silently fall back to sending that literal URL
/// as the RFC 8707 `resource` parameter, which the authorization server then
/// rejects with `invalid_target` because it doesn't match the resource it
/// actually registered.
fn oauth_discovery_url(base_url: &str) -> String {
    format!("{}/models", base_url.trim_end_matches('/'))
}

struct GolemAuthProvider {
    base_url: String,
    manager: TokioMutex<AuthorizationManager>,
}

impl GolemAuthProvider {
    async fn new(base_url: String) -> Result<Self> {
        let mut manager = AuthorizationManager::new(oauth_discovery_url(&base_url)).await?;
        manager.set_credential_store(GooseCredentialStore::new(
            GOLEM_OAUTH_CREDENTIAL_NAME.to_string(),
        ));
        Ok(Self {
            base_url,
            manager: TokioMutex::new(manager),
        })
    }

    async fn get_valid_token(&self) -> Result<String, ProviderError> {
        let mut manager = self.manager.lock().await;
        // Best-effort restore from a previous `configure_oauth` login,
        // retried on every call rather than just once at construction time.
        // `initialize_from_store` only re-runs RFC 8414 discovery when the
        // manager doesn't already have server metadata, so once discovery
        // has succeeded once this is just a cheap keyring read; but if
        // discovery failed transiently at startup (e.g. a network blip),
        // this retries it instead of leaving the manager permanently
        // unconfigured. Without this, a later refresh would fail with a
        // confusing "OAuth client not configured" instead of transparently
        // recovering once the credentials and network are actually fine.
        let _ = manager.initialize_from_store().await;
        match manager.get_access_token().await {
            Ok(token) => Ok(token),
            Err(AuthError::AuthorizationRequired) => Err(ProviderError::NotConfigured),
            Err(e) => Err(ProviderError::Authentication(e.to_string())),
        }
    }

    /// Runs the interactive discovery-driven OAuth flow: discovers the
    /// authorization server from `base_url` (RFC 8414), registers a client
    /// dynamically if needed (RFC 7591), opens a browser for PKCE
    /// authorization-code exchange, and persists the resulting credentials
    /// through the same keyring-backed store every provider's API key uses.
    async fn configure(&self) -> Result<(), ProviderError> {
        let name = GOLEM_OAUTH_CREDENTIAL_NAME.to_string();
        let discovery_url = oauth_discovery_url(&self.base_url);
        let fresh = oauth_flow(&discovery_url, &name, None)
            .await
            .map_err(|e| ProviderError::Authentication(format!("golem OAuth flow failed: {e}")))?;
        *self.manager.lock().await = fresh;
        Ok(())
    }
}

#[async_trait]
impl AuthProvider for GolemAuthProvider {
    async fn get_auth_header(&self) -> Result<(String, String)> {
        let token = self.get_valid_token().await?;
        Ok(("Authorization".to_string(), format!("Bearer {token}")))
    }
}

/// Delegating Provider that forwards chat/stream/etc. to an inner
/// `OpenAiCompatibleProvider` pointed at the golem base URL, but overrides
/// `configure_oauth` so the desktop "Sign in" button (and `goose configure`)
/// drive the discovery-driven OAuth flow.
#[derive(serde::Serialize)]
pub struct GolemOAuthProvider {
    #[serde(skip)]
    inner: OpenAiCompatibleProvider,
    #[serde(skip)]
    auth: Arc<GolemAuthProvider>,
}

impl GolemOAuthProvider {
    pub async fn cleanup() -> Result<()> {
        let name = GOLEM_OAUTH_CREDENTIAL_NAME.to_string();
        GooseCredentialStore::new(name)
            .clear()
            .await
            .map_err(|e| anyhow::anyhow!(e))
    }
}

#[async_trait]
impl Provider for GolemOAuthProvider {
    fn get_name(&self) -> &str {
        self.inner.get_name()
    }

    async fn stream(
        &self,
        model_config: &ModelConfig,
        system: &str,
        messages: &[Message],
        tools: &[Tool],
    ) -> Result<MessageStream, ProviderError> {
        self.auth.get_valid_token().await?;
        self.inner
            .stream(model_config, system, messages, tools)
            .await
    }

    async fn fetch_supported_models(&self) -> Result<Vec<String>, ProviderError> {
        self.auth.get_valid_token().await?;
        self.inner.fetch_supported_models().await
    }

    async fn configure_oauth(&self) -> Result<(), ProviderError> {
        self.auth.configure().await
    }
}

impl goose_providers::base::ProviderDescriptor for GolemOAuthProvider {
    fn metadata() -> ProviderMetadata {
        ProviderMetadata::new(
            GOLEM_PROVIDER_NAME,
            "Golem (agent.asanti.dev)",
            "Self-hosted inference gateway authenticated via discovery-driven OAuth/OIDC (RFC 8414 discovery + dynamic client registration) instead of an API key. Models are discovered live after signing in.",
            "",
            vec![],
            GOLEM_DOC_URL,
            vec![
                ConfigKey::new_oauth("GOLEM_OAUTH_TOKEN", true, true, None, false),
                ConfigKey::new("GOLEM_HOST", false, false, Some(GOLEM_DEFAULT_HOST), false),
            ],
        )
        .with_setup(
            crate::providers::catalog::ProviderSetupMetadata::new(
                crate::providers::catalog::ProviderSetupCategory::Model,
                crate::providers::catalog::ProviderSetupMethod::OauthBrowser,
                crate::providers::catalog::ProviderSetupGroup::Default,
            )
            .with_docs_url(GOLEM_DOC_URL)
            .with_capabilities(false, true, false),
        )
    }
}

impl ProviderDef for GolemOAuthProvider {
    type Provider = Self;

    fn from_env(
        _extensions: Vec<crate::config::ExtensionConfig>,
        tls_config: Option<crate::providers::api_client::TlsConfig>,
    ) -> BoxFuture<'static, Result<Self::Provider>> {
        Box::pin(async move {
            let config = crate::config::Config::global();
            let host: String = config
                .get_param("GOLEM_HOST")
                .unwrap_or_else(|_| GOLEM_DEFAULT_HOST.to_string());

            let auth = Arc::new(GolemAuthProvider::new(host.clone()).await?);
            let auth_for_client = Arc::clone(&auth);
            let api_client = ApiClient::new_with_tls(
                host,
                AuthMethod::Custom(Box::new(SharedAuthProvider(auth_for_client))),
                tls_config,
            )?
            .with_request_builder(crate::session_context::session_id_request_builder());

            let inner = OpenAiCompatibleProvider::new(
                GOLEM_PROVIDER_NAME.to_string(),
                api_client,
                String::new(),
            );

            Ok(Self { inner, auth })
        })
    }
}

/// Adapter so the same `GolemAuthProvider` can be both owned by the wrapper
/// (for `configure_oauth`) and embedded as an `AuthMethod::Custom` boxed
/// `AuthProvider` in the inner `ApiClient`.
struct SharedAuthProvider(Arc<GolemAuthProvider>);

#[async_trait]
impl AuthProvider for SharedAuthProvider {
    async fn get_auth_header(&self) -> Result<(String, String)> {
        self.0.get_auth_header().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Regression test: `crate::oauth::oauth_flow`/`GooseCredentialStore` key
    // stored credentials by a flat string name shared with MCP extension
    // auth. A user with an MCP extension also named "golem" got this
    // provider's cached token silently overwritten with one scoped to the
    // extension's resource instead (observed as an `aud` claim of
    // `.../golem/mcp` instead of `.../golem/llm`, with no error until the
    // token was actually used). Pinning that these differ prevents that
    // collision from coming back.
    #[test]
    fn oauth_credential_name_differs_from_the_provider_name() {
        assert_ne!(GOLEM_OAUTH_CREDENTIAL_NAME, GOLEM_PROVIDER_NAME);
    }

    // Regression test: agent.asanti.dev's protected-resource metadata is
    // scoped to `.../golem/llm` (no `/v1`), and only an actual protected
    // endpoint under the API root triggers the 401 challenge that reveals
    // it — the bare API root returns a 303 UI redirect instead. Discovering
    // against the raw base_url silently sent that literal URL as the RFC
    // 8707 `resource` parameter, which the server rejected with
    // `invalid_target`.
    #[test]
    fn oauth_discovery_url_appends_models_to_the_api_root() {
        assert_eq!(
            oauth_discovery_url(GOLEM_DEFAULT_HOST),
            "https://agent.asanti.dev/golem/llm/v1/models"
        );
    }

    #[test]
    fn oauth_discovery_url_tolerates_a_trailing_slash() {
        assert_eq!(
            oauth_discovery_url("https://agent.asanti.dev/golem/llm/v1/"),
            "https://agent.asanti.dev/golem/llm/v1/models"
        );
    }

    /// Builds a `GolemAuthProvider` around a fresh, unconfigured
    /// `AuthorizationManager` (default in-memory credential store, nothing
    /// restored) so tests never touch the real keyring-backed
    /// `GooseCredentialStore`.
    async fn unconfigured_auth_provider() -> GolemAuthProvider {
        let manager = AuthorizationManager::new("http://127.0.0.1:1".to_string())
            .await
            .unwrap();
        GolemAuthProvider {
            base_url: "http://127.0.0.1:1".to_string(),
            manager: TokioMutex::new(manager),
        }
    }

    #[tokio::test]
    async fn missing_token_does_not_start_oauth() {
        let auth = unconfigured_auth_provider().await;

        let error = auth.get_valid_token().await.unwrap_err();

        assert_eq!(error, ProviderError::NotConfigured);
    }

    #[tokio::test]
    async fn stream_preserves_not_configured_error() {
        let auth = Arc::new(unconfigured_auth_provider().await);
        let api_client =
            ApiClient::new_with_tls("http://127.0.0.1:1".to_string(), AuthMethod::NoAuth, None)
                .unwrap();
        let provider = GolemOAuthProvider {
            inner: OpenAiCompatibleProvider::new(
                GOLEM_PROVIDER_NAME.to_string(),
                api_client,
                String::new(),
            ),
            auth,
        };

        let error = provider
            .stream(&ModelConfig::new("test-model"), "", &[], &[])
            .await
            .err()
            .unwrap();

        assert_eq!(error, ProviderError::NotConfigured);
    }
}
