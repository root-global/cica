//! Linear agent channel.
//!
//! Every other channel is outbound-only: Telegram long-polls, Slack uses Socket
//! Mode, Signal talks to a local daemon. This one is the first that *accepts* a
//! connection — Linear POSTs an `AgentSessionEvent` webhook when the app is
//! @mentioned on an issue, and the reply goes back out as an "agent activity".
//!
//! Two deadlines shape the design:
//!
//! * Linear wants HTTP 200 within **5 seconds**, and turns run 40–200s. So the
//!   handler verifies the signature, acknowledges, and spawns the turn.
//! * An agent must emit a `thought` activity within **10 seconds** of a session
//!   opening or Linear marks it failed. `start_typing()` already runs before
//!   every turn, so the typing indicator *is* the acknowledgement.
//!
//! cica never terminates TLS. The listener binds plain HTTP behind a terminator
//! (an ALB, or a reverse proxy) — but it verifies the HMAC itself regardless,
//! because the signature is the only thing that proves Linear sent the request.

use anyhow::{Context, Result};
use async_trait::async_trait;
use axum::Router;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::routing::{get, post};
use hmac::{Hmac, Mac};
use serde::Deserialize;
use serde_json::json;
use sha2::Sha256;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use subtle::ConstantTimeEq;
use tokio::sync::oneshot;
use tracing::{debug, info, warn};

use super::{
    Channel, Identity, TypingGuard, determine_action, execute_action, execute_claude_query,
};
use crate::config::LinearConfig;
use crate::runtime::Runtime;
use crate::sandbox::Affinity;

const LINEAR_API: &str = "https://api.linear.app/graphql";

/// How far a `webhookTimestamp` may be from our own clock. Linear's own guidance
/// is one minute; being stricter would make us fragile to ordinary clock skew.
const TIMESTAMP_TOLERANCE: Duration = Duration::from_secs(60);

/// Header carrying the hex-encoded HMAC-SHA256 of the raw request body.
const SIGNATURE_HEADER: &str = "linear-signature";

// ---------------------------------------------------------------------------
// Webhook payloads
// ---------------------------------------------------------------------------

/// The subset of `AgentSessionEvent` we act on. Linear sends a good deal more;
/// unknown fields are ignored so a payload addition upstream cannot break us.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AgentSessionEvent {
    /// `created` when the app is first mentioned, `prompted` for every later
    /// comment in the same session.
    action: String,
    agent_session: AgentSession,
    /// Linear pre-formats the issue, its comments and any workspace guidance
    /// into one string. Cheaper and more faithful than re-fetching the ticket.
    #[serde(default)]
    prompt_context: Option<String>,
    /// Present on `prompted`: the comment that triggered this turn.
    #[serde(default)]
    agent_activity: Option<AgentActivity>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AgentSession {
    id: String,
    #[serde(default)]
    issue: Option<Issue>,
    #[serde(default)]
    creator: Option<User>,
    #[serde(default)]
    comment: Option<Comment>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Issue {
    id: String,
    #[serde(default)]
    identifier: Option<String>,
    #[serde(default)]
    title: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct User {
    id: String,
    #[serde(default)]
    email: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    display_name: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Comment {
    #[serde(default)]
    body: Option<String>,
    #[serde(default)]
    user: Option<User>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AgentActivity {
    #[serde(default)]
    content: Option<AgentActivityContent>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AgentActivityContent {
    #[serde(default)]
    body: Option<String>,
}

/// What the webhook asks us to do, once parsed and stripped of Linear's shape.
#[derive(Debug, PartialEq, Eq)]
struct Invocation {
    session_id: String,
    issue_id: String,
    /// `DAT-633` — for logs and the session label, not for addressing.
    issue_ref: String,
    user_id: String,
    user_email: Option<String>,
    user_name: Option<String>,
    title: Option<String>,
    prompt: String,
    context: Option<String>,
}

impl AgentSessionEvent {
    /// Reduce the payload to an invocation, or `None` when there is nothing to
    /// do (an action we don't handle, or a session with no issue behind it).
    fn to_invocation(&self) -> Option<Invocation> {
        if self.action != "created" && self.action != "prompted" {
            debug!(action = %self.action, "ignoring agent session action");
            return None;
        }

        let issue = self.agent_session.issue.as_ref()?;

        // On `created` the triggering text is the comment that mentioned us; on
        // `prompted` it arrives as the activity body.
        let prompt = self
            .agent_activity
            .as_ref()
            .and_then(|a| a.content.as_ref())
            .and_then(|c| c.body.clone())
            .or_else(|| {
                self.agent_session
                    .comment
                    .as_ref()
                    .and_then(|c| c.body.clone())
            })
            .unwrap_or_default();

        let commenter = self
            .agent_session
            .comment
            .as_ref()
            .and_then(|c| c.user.as_ref())
            .or(self.agent_session.creator.as_ref());

        Some(Invocation {
            session_id: self.agent_session.id.clone(),
            issue_id: issue.id.clone(),
            issue_ref: issue.identifier.clone().unwrap_or_else(|| issue.id.clone()),
            user_id: commenter.map(|u| u.id.clone()).unwrap_or_default(),
            user_email: commenter.and_then(|u| u.email.clone()),
            user_name: commenter.and_then(|u| u.display_name.clone().or(u.name.clone())),
            title: issue.title.clone(),
            prompt: strip_leading_mentions(&prompt),
            context: self.prompt_context.clone(),
        })
    }
}

/// Drop the `@Sprout` that addressed us, and nothing else.
///
/// Linear delivers the comment as written, so "@Sprout what does this decide?"
/// arrives with the mention attached — addressing, not instruction. Only
/// *leading* mentions go: "ask @dave whether it shipped" is context the turn
/// needs, and stripping every `@word` would throw it away.
fn strip_leading_mentions(text: &str) -> String {
    let mut rest = text.trim_start();
    while let Some(after) = rest.strip_prefix('@') {
        let end = after.find(char::is_whitespace).unwrap_or(after.len());
        rest = after[end..].trim_start();
    }
    rest.trim().to_string()
}

// ---------------------------------------------------------------------------
// Signature verification
// ---------------------------------------------------------------------------

#[derive(Debug, PartialEq, Eq)]
pub enum VerifyError {
    MissingSignature,
    BadEncoding,
    Mismatch,
    StaleTimestamp,
}

/// Verify a webhook against the signing secret.
///
/// The HMAC is computed over the *raw* bytes. Deserializing first and
/// re-serializing would produce a different byte string and never match, which
/// is why the handler takes `Bytes` rather than `Json<T>`.
pub fn verify_signature(
    secret: &str,
    body: &[u8],
    signature: Option<&str>,
    timestamp_ms: Option<i64>,
    now: SystemTime,
) -> std::result::Result<(), VerifyError> {
    let signature = signature.ok_or(VerifyError::MissingSignature)?;
    let provided = hex::decode(signature.trim()).map_err(|_| VerifyError::BadEncoding)?;

    let mut mac =
        Hmac::<Sha256>::new_from_slice(secret.as_bytes()).expect("HMAC accepts keys of any length");
    mac.update(body);
    let expected = mac.finalize().into_bytes();

    if provided.len() != expected.len() || provided.ct_eq(&expected).unwrap_u8() != 1 {
        return Err(VerifyError::Mismatch);
    }

    // Replay guard. A payload without a timestamp still verifies — the field is
    // documented but optional, and rejecting on its absence would drop valid
    // deliveries.
    if let Some(timestamp_ms) = timestamp_ms {
        let now_ms = now
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);
        if (now_ms - timestamp_ms).unsigned_abs() > TIMESTAMP_TOLERANCE.as_millis() as u64 {
            return Err(VerifyError::StaleTimestamp);
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Agent activities (the outbound half)
// ---------------------------------------------------------------------------

/// The activity types Linear accepts. `Thought` and `Action` may be ephemeral —
/// shown briefly and replaced by whatever the agent emits next.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivityKind {
    Thought,
    Response,
    Error,
}

impl ActivityKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Thought => "thought",
            Self::Response => "response",
            Self::Error => "error",
        }
    }

    /// Only `thought` and `action` may be marked ephemeral; sending
    /// `ephemeral: true` with a `response` is rejected.
    fn may_be_ephemeral(self) -> bool {
        matches!(self, Self::Thought)
    }
}

/// Build the `agentActivityCreate` variables for one activity.
pub fn activity_variables(
    session_id: &str,
    kind: ActivityKind,
    body: &str,
    ephemeral: bool,
) -> serde_json::Value {
    let mut input = json!({
        "agentSessionId": session_id,
        "content": { "type": kind.as_str(), "body": body },
    });
    if ephemeral && kind.may_be_ephemeral() {
        input["ephemeral"] = json!(true);
    }
    json!({ "input": input })
}

const ACTIVITY_MUTATION: &str = r#"
mutation AgentActivityCreate($input: AgentActivityCreateInput!) {
  agentActivityCreate(input: $input) {
    success
  }
}
"#;

/// Minimal GraphQL client. cica already depends on `reqwest`; a generated
/// client for one mutation would be more machinery than the job needs.
#[derive(Clone)]
struct LinearApi {
    http: reqwest::Client,
    access_token: String,
}

impl LinearApi {
    fn new(access_token: String) -> Self {
        Self {
            http: reqwest::Client::new(),
            access_token,
        }
    }

    async fn create_activity(
        &self,
        session_id: &str,
        kind: ActivityKind,
        body: &str,
        ephemeral: bool,
    ) -> Result<()> {
        let response = self
            .http
            .post(LINEAR_API)
            .header("Authorization", format!("Bearer {}", self.access_token))
            .json(&json!({
                "query": ACTIVITY_MUTATION,
                "variables": activity_variables(session_id, kind, body, ephemeral),
            }))
            .send()
            .await
            .context("posting agent activity")?;

        let status = response.status();
        let payload: serde_json::Value = response.json().await.unwrap_or_else(|_| json!({}));

        // GraphQL reports failures in the body with a 200, so the status alone
        // is not enough.
        if !status.is_success() || payload.get("errors").is_some() {
            anyhow::bail!(
                "agentActivityCreate failed ({}): {}",
                status,
                payload.get("errors").unwrap_or(&payload)
            );
        }

        debug!(session_id, kind = kind.as_str(), "posted agent activity");
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Channel impl
// ---------------------------------------------------------------------------

pub struct LinearChannel {
    api: LinearApi,
    session_id: String,
}

impl LinearChannel {
    fn new(api: LinearApi, session_id: String) -> Self {
        Self { api, session_id }
    }
}

#[async_trait]
impl Channel for LinearChannel {
    fn name(&self) -> &'static str {
        "linear"
    }

    fn display_name(&self) -> &'static str {
        "Linear"
    }

    async fn send_message(&self, message: &str) -> Result<()> {
        self.api
            .create_activity(&self.session_id, ActivityKind::Response, message, false)
            .await
    }

    /// A failed turn in a ticket thread is permanent and public, so it goes out
    /// as an `error` activity — Linear renders it as a failure and moves the
    /// session to `error` rather than leaving it looking answered.
    async fn send_error(&self, message: &str) -> Result<()> {
        self.api
            .create_activity(&self.session_id, ActivityKind::Error, message, false)
            .await
    }

    /// The 10-second acknowledgement. An ephemeral `thought` goes out
    /// immediately and is replaced by the `response` when the turn lands, so a
    /// long turn shows as thinking rather than as silence.
    fn start_typing(&self) -> TypingGuard {
        let (cancel_tx, cancel_rx) = oneshot::channel();
        let api = self.api.clone();
        let session_id = self.session_id.clone();

        tokio::spawn(async move {
            if let Err(e) = api
                .create_activity(
                    &session_id,
                    ActivityKind::Thought,
                    "Looking into this…",
                    true,
                )
                .await
            {
                // Losing the acknowledgement costs us the session, so this is a
                // warning rather than a debug line.
                warn!("Failed to post the Linear thought acknowledgement: {}", e);
            }

            // Linear replaces an ephemeral thought with the next activity, so
            // there is nothing to refresh and nothing to tear down — just hold
            // the guard's channel so dropping it stays meaningful.
            let _ = cancel_rx.await;
        });

        TypingGuard::new(cancel_tx)
    }
}

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct AppState {
    config: Arc<LinearConfig>,
    api: LinearApi,
    rt: Arc<Runtime>,
}

pub async fn run(config: LinearConfig, rt: Arc<Runtime>) -> Result<()> {
    if config.access_token.is_empty() {
        anyhow::bail!("[channels.linear] access_token is empty");
    }
    if config.webhook_secret.is_empty() {
        // Refusing to start is deliberate: an unverified inbound endpoint on the
        // router is worse than no Linear channel at all.
        anyhow::bail!("[channels.linear] webhook_secret is empty; refusing to listen unverified");
    }

    let listen_addr = config.listen_addr.clone();
    let state = AppState {
        api: LinearApi::new(config.access_token.clone()),
        config: Arc::new(config),
        rt,
    };

    let app = Router::new()
        .route("/webhooks/linear", post(webhook))
        .route("/health", get(health))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&listen_addr)
        .await
        .with_context(|| format!("binding the Linear webhook listener on {listen_addr}"))?;

    info!("Linear webhook listener on {}", listen_addr);
    axum::serve(listener, app)
        .await
        .context("Linear webhook listener stopped")?;

    Ok(())
}

/// Load-balancer health check. Deliberately says nothing about the workspace,
/// the configuration, or whether a turn is in flight.
async fn health() -> &'static str {
    "ok"
}

async fn webhook(State(state): State<AppState>, headers: HeaderMap, body: Bytes) -> StatusCode {
    let signature = headers
        .get(SIGNATURE_HEADER)
        .and_then(|value| value.to_str().ok());

    // Parse only far enough to read the timestamp; nothing is trusted until the
    // signature checks out.
    let timestamp_ms = serde_json::from_slice::<serde_json::Value>(&body)
        .ok()
        .and_then(|v| v.get("webhookTimestamp").and_then(|t| t.as_i64()));

    if let Err(e) = verify_signature(
        &state.config.webhook_secret,
        &body,
        signature,
        timestamp_ms,
        SystemTime::now(),
    ) {
        warn!("Rejected a Linear webhook: {:?}", e);
        return StatusCode::UNAUTHORIZED;
    }

    let event: AgentSessionEvent = match serde_json::from_slice(&body) {
        Ok(event) => event,
        Err(e) => {
            // A shape we don't understand is not a delivery failure; retrying it
            // three times would not help.
            debug!("Ignoring an unparseable Linear webhook: {}", e);
            return StatusCode::OK;
        }
    };

    let Some(invocation) = event.to_invocation() else {
        return StatusCode::OK;
    };

    // Acknowledge now and work afterwards: Linear's budget is 5s, a turn is
    // 40-200s.
    tokio::spawn(async move {
        if let Err(e) = handle_invocation(state, invocation).await {
            warn!("Linear turn failed: {}", e);
        }
    });

    StatusCode::OK
}

async fn handle_invocation(state: AppState, invocation: Invocation) -> Result<()> {
    // Memories and USER.md are keyed <channel>_<user_id>, so without a mapping
    // the same human is a stranger on Linear. Resolve to their identity on
    // whichever channel they normally use.
    let (identity_channel, identity_user) = state
        .config
        .resolve_identity(invocation.user_email.as_deref(), &invocation.user_id);

    info!(
        issue = %invocation.issue_ref,
        title = invocation.title.as_deref().unwrap_or_default(),
        identity = %format!("{identity_channel}:{identity_user}"),
        "Linear mention: {}",
        invocation.prompt
    );

    let channel: Arc<dyn Channel> = Arc::new(LinearChannel::new(
        state.api.clone(),
        invocation.session_id.clone(),
    ));

    // Keyed to the *issue*, not the agent session: a mention next week resumes
    // the same conversation instead of starting cold.
    let session_key = format!("linear:{}", invocation.issue_id);

    let action = determine_action(
        &state.rt,
        &identity_channel,
        &identity_user,
        &invocation.prompt,
        &[],
        None,
        invocation.user_name.clone(),
        Some(&session_key),
    )?;

    let Some(query_text) =
        execute_action(&state.rt, channel.as_ref(), &identity_user, action).await?
    else {
        return Ok(());
    };

    // Linear's own promptContext carries the issue, its comments and any
    // workspace guidance, already formatted.
    let prompt = match invocation.context.as_deref() {
        Some(context) if !context.is_empty() => {
            format!("{context}\n\n---\n\n{query_text}")
        }
        _ => query_text,
    };

    execute_claude_query(
        state.rt.clone(),
        channel,
        &Identity::mapped(identity_channel, identity_user),
        Affinity::LinearIssue {
            issue_id: invocation.issue_id.clone(),
        },
        vec![prompt],
        Some(session_key),
        Vec::new(),
    )
    .await;

    Ok(())
}

/// Validate an access token; returns the app user's display name on success.
/// `viewer` on an `actor=app` token resolves to the app user, which is exactly
/// the identity whose name will appear on every activity.
pub async fn validate_token(access_token: &str) -> Result<String> {
    let response = reqwest::Client::new()
        .post(LINEAR_API)
        .header("Authorization", format!("Bearer {access_token}"))
        .json(&json!({ "query": "query Me { viewer { id name } }" }))
        .send()
        .await
        .context("querying the Linear viewer")?;

    let payload: serde_json::Value = response.json().await.context("reading the viewer reply")?;
    if let Some(errors) = payload.get("errors") {
        anyhow::bail!("{errors}");
    }

    payload
        .pointer("/data/viewer/name")
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .ok_or_else(|| anyhow::anyhow!("no viewer in the reply: {payload}"))
}

/// Post an activity outside a turn — used by the cron result sender.
pub async fn send_activity(access_token: &str, session_id: &str, message: &str) -> Result<()> {
    LinearApi::new(access_token.to_string())
        .create_activity(session_id, ActivityKind::Response, message, false)
        .await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sign(secret: &str, body: &[u8]) -> String {
        let mut mac = Hmac::<Sha256>::new_from_slice(secret.as_bytes()).unwrap();
        mac.update(body);
        hex::encode(mac.finalize().into_bytes())
    }

    fn now_ms() -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64
    }

    #[test]
    fn a_correctly_signed_body_verifies() {
        let body = br#"{"action":"created"}"#;
        let sig = sign("shh", body);
        assert_eq!(
            verify_signature("shh", body, Some(&sig), Some(now_ms()), SystemTime::now()),
            Ok(())
        );
    }

    #[test]
    fn a_body_signed_with_another_secret_is_rejected() {
        let body = br#"{"action":"created"}"#;
        let sig = sign("not-our-secret", body);
        assert_eq!(
            verify_signature("shh", body, Some(&sig), None, SystemTime::now()),
            Err(VerifyError::Mismatch)
        );
    }

    #[test]
    fn a_tampered_body_is_rejected() {
        let sig = sign("shh", br#"{"action":"created"}"#);
        assert_eq!(
            verify_signature(
                "shh",
                br#"{"action":"prompted"}"#,
                Some(&sig),
                None,
                SystemTime::now()
            ),
            Err(VerifyError::Mismatch)
        );
    }

    #[test]
    fn a_missing_signature_is_rejected() {
        assert_eq!(
            verify_signature("shh", b"{}", None, None, SystemTime::now()),
            Err(VerifyError::MissingSignature)
        );
    }

    #[test]
    fn a_non_hex_signature_is_rejected_before_any_comparison() {
        assert_eq!(
            verify_signature("shh", b"{}", Some("zzzz"), None, SystemTime::now()),
            Err(VerifyError::BadEncoding)
        );
    }

    #[test]
    fn a_replayed_body_is_rejected_on_its_timestamp() {
        let body = br#"{"action":"created"}"#;
        let sig = sign("shh", body);
        let two_minutes_ago = now_ms() - 120_000;
        assert_eq!(
            verify_signature(
                "shh",
                body,
                Some(&sig),
                Some(two_minutes_ago),
                SystemTime::now()
            ),
            Err(VerifyError::StaleTimestamp)
        );
    }

    #[test]
    fn a_signature_of_the_wrong_length_is_rejected() {
        // Truncated to 16 bytes: a prefix of a valid signature must not pass.
        let body = br#"{"action":"created"}"#;
        let sig = sign("shh", body);
        assert_eq!(
            verify_signature("shh", body, Some(&sig[..32]), None, SystemTime::now()),
            Err(VerifyError::Mismatch)
        );
    }

    const CREATED: &str = r#"{
      "action": "created",
      "webhookTimestamp": 1757000000000,
      "agentSession": {
        "id": "sess_1",
        "issue": { "id": "iss_abc", "identifier": "DAT-633", "title": "Trigger Sprout" },
        "creator": { "id": "usr_1", "email": "Rodrigo@RootGlobal.io", "name": "Rodrigo Neves" },
        "comment": {
          "body": "@Sprout what does this ticket decide?",
          "user": { "id": "usr_1", "email": "Rodrigo@RootGlobal.io", "displayName": "rodrigo" }
        }
      },
      "promptContext": "<issue>DAT-633</issue>"
    }"#;

    const PROMPTED: &str = r#"{
      "action": "prompted",
      "agentSession": {
        "id": "sess_1",
        "issue": { "id": "iss_abc", "identifier": "DAT-633" }
      },
      "agentActivity": { "content": { "type": "prompt", "body": "and what about the estimate?" } }
    }"#;

    #[test]
    fn a_created_event_becomes_an_invocation() {
        let event: AgentSessionEvent = serde_json::from_str(CREATED).unwrap();
        let inv = event.to_invocation().unwrap();

        assert_eq!(inv.session_id, "sess_1");
        assert_eq!(inv.issue_id, "iss_abc");
        assert_eq!(inv.issue_ref, "DAT-633");
        assert_eq!(inv.user_id, "usr_1");
        assert_eq!(inv.user_email.as_deref(), Some("Rodrigo@RootGlobal.io"));
        assert_eq!(inv.user_name.as_deref(), Some("rodrigo"));
        assert_eq!(inv.title.as_deref(), Some("Trigger Sprout"));
        // The mention is addressing, not instruction.
        assert_eq!(inv.prompt, "what does this ticket decide?");
        assert_eq!(inv.context.as_deref(), Some("<issue>DAT-633</issue>"));
    }

    #[test]
    fn a_prompted_event_takes_its_text_from_the_activity() {
        let event: AgentSessionEvent = serde_json::from_str(PROMPTED).unwrap();
        let inv = event.to_invocation().unwrap();

        assert_eq!(inv.prompt, "and what about the estimate?");
        // Same issue, so the same session key — this is what makes a follow-up
        // resume rather than start cold.
        assert_eq!(inv.issue_id, "iss_abc");
        assert_eq!(inv.context, None);
    }

    #[test]
    fn an_unhandled_action_yields_nothing() {
        let payload = CREATED.replace(r#""action": "created""#, r#""action": "deleted""#);
        let event: AgentSessionEvent = serde_json::from_str(&payload).unwrap();
        assert!(event.to_invocation().is_none());
    }

    #[test]
    fn a_session_without_an_issue_yields_nothing() {
        // Agent sessions can hang off other surfaces; we only answer on issues.
        let event: AgentSessionEvent =
            serde_json::from_str(r#"{"action":"created","agentSession":{"id":"sess_1"}}"#).unwrap();
        assert!(event.to_invocation().is_none());
    }

    #[test]
    fn unknown_payload_fields_do_not_break_parsing() {
        let payload = CREATED.replace(
            r#""action": "created""#,
            r#""action": "created", "somethingLinearAddedLater": {"a": 1}"#,
        );
        assert!(serde_json::from_str::<AgentSessionEvent>(&payload).is_ok());
    }

    #[test]
    fn a_thought_may_be_ephemeral_but_a_response_may_not() {
        let thought = activity_variables("s1", ActivityKind::Thought, "hm", true);
        assert_eq!(thought["input"]["ephemeral"], json!(true));
        assert_eq!(thought["input"]["content"]["type"], "thought");

        let response = activity_variables("s1", ActivityKind::Response, "done", true);
        assert!(response["input"].get("ephemeral").is_none());
        assert_eq!(response["input"]["content"]["type"], "response");

        let error = activity_variables("s1", ActivityKind::Error, "boom", false);
        assert_eq!(error["input"]["content"]["type"], "error");
        assert_eq!(error["input"]["agentSessionId"], "s1");
    }

    #[test]
    fn the_mention_that_addressed_us_is_dropped() {
        assert_eq!(
            strip_leading_mentions("@Sprout what does this ticket decide?"),
            "what does this ticket decide?"
        );
        // Linear can put the agent behind a team mention too.
        assert_eq!(
            strip_leading_mentions("@Sprout @Data please look"),
            "please look"
        );
        assert_eq!(strip_leading_mentions("  @Sprout   hello  "), "hello");
        assert_eq!(strip_leading_mentions("@Sprout"), "");
    }

    #[test]
    fn a_mention_inside_the_question_is_context_and_survives() {
        // Stripping every @word would lose who the turn is being asked about.
        assert_eq!(
            strip_leading_mentions("@Sprout did @dave say it shipped?"),
            "did @dave say it shipped?"
        );
        assert_eq!(
            strip_leading_mentions("no mention here at all"),
            "no mention here at all"
        );
    }
}
