//! DeepSeek Harness (`dsh`) over ACP. Reuses the Cursor ACP protocol; dsh keeps its sessions as JSONL
//! files under `$DSH_HOME/sessions`. Fork addition (Sellwy/cloudroom).
//!
//! Provider and model come from the agent account's own dsh profile (`~/.dsh/profiles/acp/cordis.patch.yml`
//! and `~/.dsh/settings.yaml`), so Cloud can run dsh through OpenRouter or DeepSeek's API alike.
use super::{Handle, Resume, command as child_command, cursor};
use crate::config::{Config, HarnessConfig};
use serde_json::json;
use std::io;
use tokio::process::Command;

pub(super) const DEEPSEEK: cursor::Flavor = cursor::Flavor {
    harness: "deepseek",
    name: "DeepSeek Harness",
    native_path: |home, _| home.join(SESSIONS),
    valid_id,
    capture: false,
    notices: &[],
    // dsh's `session/resume` does not replay history; Cloudroom already has it.
    replay: false,
    explain_stops: true,
    empty_turn: None,
};
const SESSIONS: &str = "sessions";
/// The efforts dsh's pi-ai routes can offer; each model accepts its own subset.
pub const REASONING_LEVELS: &[&str] = &["off", "minimal", "low", "medium", "high", "xhigh", "max"];

fn invalid(message: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

/// dsh session IDs are opaque; accept the safe file-name alphabet only, since they name JSONL files.
fn valid_id(id: &str) -> bool {
    (8..=128).contains(&id.len())
        && !id.starts_with(['-', '.'])
        && id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
        && !id.contains("..")
}

pub(super) fn command(config: &Config, profile: &HarnessConfig) -> io::Result<Command> {
    if profile.home.canonicalize()? != config.account_home.join(".dsh").canonicalize()? {
        return Err(invalid(
            "DeepSeek Harness home must belong to the configured agent account".into(),
        ));
    }
    let mut command = child_command(&profile.binary, config);
    command
        .env("DSH_HOME", &profile.home)
        // No usage logs to DeepSeek's telemetry endpoint from Cloud agents.
        .env("DSH_TELEMETRY_DISABLED", "1")
        .args(["--profile", "acp"]);
    Ok(command)
}

pub(super) async fn start(handle: &Handle) -> io::Result<String> {
    cursor::initialize(handle, "DeepSeek Harness").await?;
    let mut params = json!({"cwd":handle.repository,"mcpServers":[]});
    let method = if let Some(saved) = &handle.resume {
        params["sessionId"] = json!(saved.id);
        "session/resume"
    } else {
        "session/new"
    };
    let (id, session) = cursor::attach(handle, "DeepSeek Harness", method, params).await?;
    let requested = handle.profile.model.as_str();
    if requested != "default" && session["models"]["currentModelId"] != requested {
        handle
            .call(
                "session/set_model",
                json!({"sessionId":id,"modelId":requested}),
            )
            .await
            .map_err(|error| {
                invalid(format!(
                    "DeepSeek Harness model {requested} is unavailable in Cloud: {error}"
                ))
            })?;
    }
    if let Some(reasoning) = handle.reasoning.as_deref() {
        // Only send efforts dsh advertises for this session; others keep the model's default.
        let offered = session["configOptions"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|option| option["id"] == "reasoning_effort");
        if offered {
            handle
                .call(
                    "session/set_config_option",
                    json!({"sessionId":id,"configId":"reasoning_effort","value":reasoning}),
                )
                .await?;
        }
    }
    Ok(id)
}

pub(super) fn validate(
    profile: &HarnessConfig,
    saved: &Resume,
    _identity: super::files::Identity,
) -> io::Result<()> {
    if !valid_id(&saved.id) || saved.path != profile.home.join(SESSIONS) {
        return Err(io::Error::other(
            "DeepSeek Harness native path does not match its session",
        ));
    }
    if !profile.home.join(SESSIONS).is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "DeepSeek Harness has no saved sessions",
        ));
    }
    Ok(())
}
