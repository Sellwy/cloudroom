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
    cursor::initialize_resumable(handle, "DeepSeek Harness").await?;
    let mut params = json!({"cwd":handle.repository,"mcpServers":[]});
    let method = if let Some(saved) = &handle.resume {
        params["sessionId"] = json!(saved.id);
        "session/resume"
    } else {
        "session/new"
    };
    let (id, session) = cursor::attach(handle, "DeepSeek Harness", method, params).await?;
    // dsh has no `session/set_model`; it switches models through its `model` config option, whose values
    // are JSON pairs like `["openrouter","z-ai/glm-5.3-flash"]`. `acp-default` is the app's "agent default".
    let requested = handle.profile.model.as_str();
    if !matches!(requested, "default" | "acp-default") {
        let model = model_option(&session);
        if model["currentValue"] != requested {
            if !option_values(&model).any(|value| value == requested) {
                return Err(invalid(format!(
                    "DeepSeek Harness model {requested} is unavailable in Cloud"
                )));
            }
            handle
                .call(
                    "session/set_config_option",
                    json!({"sessionId":id,"configId":model["id"],"value":requested}),
                )
                .await
                .map_err(|error| {
                    invalid(format!(
                        "DeepSeek Harness model {requested} is unavailable in Cloud: {error}"
                    ))
                })?;
        }
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

/// The session's model selector (`category: "model"`, else `id: "model"`), or null.
fn model_option(session: &serde_json::Value) -> serde_json::Value {
    let options = session["configOptions"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    options
        .iter()
        .find(|option| option["category"] == "model")
        .or_else(|| options.iter().find(|option| option["id"] == "model"))
        .cloned()
        .unwrap_or_default()
}

/// Every selectable value of a select option, including ones nested in ACP option groups.
fn option_values(option: &serde_json::Value) -> impl Iterator<Item = &str> {
    option["options"]
        .as_array()
        .into_iter()
        .flatten()
        .flat_map(|entry| match entry["options"].as_array() {
            Some(group) => group.iter().collect::<Vec<_>>(),
            None => vec![entry],
        })
        .filter_map(|entry| entry["value"].as_str())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_grouped_model_values() {
        let session = json!({"configOptions":[{"id":"model","category":"model","type":"select",
            "currentValue":"[\"openrouter\",\"z-ai/glm-5.3-flash\"]",
            "options":[{"group":"openrouter","name":"openrouter","options":[
                {"value":"[\"openrouter\",\"z-ai/glm-5.3-flash\"]","name":"GLM 5.3 Flash"},
                {"value":"[\"openrouter\",\"z-ai/glm-5.3\"]","name":"GLM 5.3"}]}]}]});
        let model = model_option(&session);
        assert_eq!(model["id"], "model");
        assert_eq!(
            option_values(&model).collect::<Vec<_>>(),
            [
                "[\"openrouter\",\"z-ai/glm-5.3-flash\"]",
                "[\"openrouter\",\"z-ai/glm-5.3\"]"
            ]
        );
    }
}
