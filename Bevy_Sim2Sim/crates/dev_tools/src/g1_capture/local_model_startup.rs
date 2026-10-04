//! Finite, explicitly configured preparation before the first task image.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::PathBuf,
    time::{Duration, Instant},
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Configuration {
    pub ready_path: PathBuf,
    pub ready_sha256: String,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Ready {
    schema: String,
    episode_id: u64,
}

pub(super) struct Gate {
    configuration: Configuration,
    started: Instant,
    pub renderer_announced: bool,
}

impl Gate {
    pub fn expired(&self) -> bool {
        self.started.elapsed() > Duration::from_secs(360)
    }
    pub fn new(configuration: Configuration) -> Result<Self, String> {
        if !configuration.ready_path.is_absolute()
            || configuration.ready_path.exists()
            || configuration.ready_path.is_symlink()
            || configuration
                .ready_path
                .parent()
                .is_none_or(|p| !p.is_dir() || p.is_symlink())
            || configuration.ready_sha256.len() != 64
            || !configuration
                .ready_sha256
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        {
            return Err(
                "local model preparation requires a new owned absolute ready file and bound hash"
                    .into(),
            );
        }
        Ok(Self {
            configuration,
            started: Instant::now(),
            renderer_announced: false,
        })
    }

    pub fn poll(
        &self,
        episode_id: u64,
        native_tick: u64,
    ) -> Result<Option<serde_json::Value>, String> {
        if native_tick != 0 || episode_id == 0 || self.expired() {
            return Err(
                "local model preparation changed the zero-Tick boundary or exceeded360s".into(),
            );
        }
        match fs::symlink_metadata(&self.configuration.ready_path) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e.to_string()),
            Ok(meta) if !meta.is_file() || meta.file_type().is_symlink() || meta.len() > 1024 => {
                return Err(
                    "local model preparation ready file exceeds its type/byte budget".into(),
                );
            }
            Ok(_) => {}
        }
        let bytes = fs::read(&self.configuration.ready_path).map_err(|e| e.to_string())?;
        if format!("{:x}", Sha256::digest(&bytes)) != self.configuration.ready_sha256 {
            return Err("local model preparation ready-file identity mismatch".into());
        }
        let ready: Ready = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        if ready.schema != "g1_local_model_services_ready_v1" || ready.episode_id != episode_id {
            return Err("local model preparation has a foreign schema/episode".into());
        }
        Ok(Some(
            serde_json::json!({"schema":"g1_renderer_before_local_models_preparation_v1",
            "episode_id":episode_id,"native_tick":native_tick,"preparation_wall_ms":self.started.elapsed().as_secs_f64()*1000.,
            "ready_sha256":self.configuration.ready_sha256,"renderer_ready_before_models":self.renderer_announced,
            "maximum_preparation_wall_seconds":360,"first_task_image_not_yet_requested":true,
            "preparation_is_not_physical_or_model_inference":true,"task_qualified":false}),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> (PathBuf, Gate, Vec<u8>) {
        let root = std::env::temp_dir().join(format!(
            "g1_local_model_gate_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&root).unwrap();
        let bytes = serde_json::to_vec(&Ready {
            schema: "g1_local_model_services_ready_v1".into(),
            episode_id: 7,
        })
        .unwrap();
        let gate = Gate::new(Configuration {
            ready_path: root.join("ready.json"),
            ready_sha256: format!("{:x}", Sha256::digest(&bytes)),
        })
        .unwrap();
        (root, gate, bytes)
    }
    #[test]
    fn waits_without_starting_then_admits_only_bound_zero_tick_episode() {
        let (root, gate, bytes) = fixture();
        assert!(gate.poll(7, 0).unwrap().is_none());
        fs::write(&gate.configuration.ready_path, &bytes).unwrap();
        assert!(gate.poll(7, 0).unwrap().is_some());
        assert!(gate.poll(8, 0).is_err());
        assert!(gate.poll(7, 1).is_err());
        fs::write(&gate.configuration.ready_path, b"{}").unwrap();
        assert!(gate.poll(7, 0).is_err());
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn expired_preparation_and_old_existing_file_cannot_admit() {
        let (root, mut gate, bytes) = fixture();
        gate.started = Instant::now() - Duration::from_secs(361);
        assert!(gate.poll(7, 0).is_err());
        fs::write(&gate.configuration.ready_path, &bytes).unwrap();
        assert!(
            Gate::new(Configuration {
                ready_path: gate.configuration.ready_path.clone(),
                ready_sha256: gate.configuration.ready_sha256.clone()
            })
            .is_err()
        );
        fs::remove_dir_all(root).unwrap();
    }
}
