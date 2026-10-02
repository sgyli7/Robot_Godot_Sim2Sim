//! Required station assets and runtime GPU failure reporting.

use bevy::{
    asset::LoadState,
    pbr::Material,
    prelude::*,
    render::{
        Render, RenderApp, RenderSystems,
        render_resource::{CachedPipelineState, PipelineCache, PollType},
        renderer::RenderDevice,
    },
    shader::{Shader, ShaderCacheError, ShaderRef},
};
use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::Duration,
};

/// Shared runtime health survives the application runner taking the worlds.
#[derive(Resource, Clone, Default)]
pub struct StationRenderHealth(Arc<Mutex<HealthState>>);

#[derive(Default)]
struct HealthState {
    snapshot: StationRenderSnapshot,
    error_drained: bool,
}

/// Runtime readiness and failure; these values do not qualify visual correctness.
#[derive(Clone, Default)]
pub struct StationRenderSnapshot {
    pub ready: bool,
    pub error: Option<String>,
    pub compiled: usize,
    pub pending: usize,
}

impl StationRenderHealth {
    /// Read current health without changing loaded scene configuration.
    pub fn snapshot(&self) -> StationRenderSnapshot {
        self.0.lock().unwrap().snapshot.clone()
    }
}

#[derive(Resource, Clone)]
struct RequiredRenderAssets {
    fragments: Vec<Handle<Shader>>,
    font: Handle<Font>,
}

/// Canonicalize the asset root and reject missing required station shaders/font.
pub fn validate_render_asset_root(asset_root: &Path) -> Result<PathBuf, String> {
    let root = asset_root
        .canonicalize()
        .map_err(|e| format!("Asset root {}: {e}", asset_root.display()))?;
    for shader in ["station_enamel.wgsl", "station_ink.wgsl"] {
        let path = root.join("game/shaders").join(shader);
        std::fs::read_to_string(&path)
            .map_err(|e| format!("Required shader {}: {e}", path.display()))?;
    }
    let font = root.join("third_party/fonts/noto_sans_cjk_regular.otf");
    std::fs::read(&font).map_err(|e| format!("Required station font {}: {e}", font.display()))?;
    Ok(root)
}

/// Install shared asset/pipeline failure handling after the render plugins.
pub fn install_station_render_health(app: &mut App) -> Result<StationRenderHealth, String> {
    let server = app.world().resource::<AssetServer>();
    let handles = RequiredRenderAssets {
        fragments: vec![
            server.load("game/shaders/station_enamel.wgsl"),
            server.load("game/shaders/station_ink.wgsl"),
        ],
        font: server.load("third_party/fonts/noto_sans_cjk_regular.otf"),
    };
    install_render_health(app, handles)
}

/// Source G1 PBR-only diagnostics must await the material actually drawn;
/// unused station enamel/ink pipelines are never queued by that scene.
pub fn install_pbr_render_health(app: &mut App) -> Result<StationRenderHealth, String> {
    let server = app.world().resource::<AssetServer>();
    let fragment = match <StandardMaterial as Material>::fragment_shader() {
        ShaderRef::Handle(handle) => handle,
        ShaderRef::Path(path) => server.load(path),
        ShaderRef::Default => return Err("standard PBR fragment shader unavailable".into()),
    };
    let handles = RequiredRenderAssets {
        fragments: vec![fragment],
        font: server.load("third_party/fonts/noto_sans_cjk_regular.otf"),
    };
    install_render_health(app, handles)
}

fn install_render_health(
    app: &mut App,
    handles: RequiredRenderAssets,
) -> Result<StationRenderHealth, String> {
    let status = StationRenderHealth::default();
    let render = app
        .get_sub_app_mut(RenderApp)
        .ok_or("GPU render application unavailable")?;
    render
        .insert_resource(status.clone())
        .insert_resource(handles.clone())
        .add_systems(Render, observe_pipelines.in_set(RenderSystems::Cleanup));
    app.insert_resource(status.clone())
        .insert_resource(handles)
        .add_systems(Update, observe_asset_failures);
    Ok(status)
}

fn observe_asset_failures(
    server: Res<AssetServer>,
    handles: Res<RequiredRenderAssets>,
    status: Res<StationRenderHealth>,
) {
    for handle in handles
        .fragments
        .iter()
        .map(|handle| handle.id().untyped())
        .chain([handles.font.id().untyped()])
    {
        if let Some(LoadState::Failed(error)) = server.get_load_state(handle) {
            let mut state = status.0.lock().unwrap();
            state
                .snapshot
                .error
                .get_or_insert_with(|| format!("Required render asset failed: {error}"));
        }
    }
}

fn observe_pipelines(
    cache: Res<PipelineCache>,
    handles: Res<RequiredRenderAssets>,
    status: Res<StationRenderHealth>,
    device: Res<RenderDevice>,
) {
    let mut state = status.0.lock().unwrap();
    state.snapshot.compiled = 0;
    state.snapshot.pending = 0;
    let mut fragments_ready = vec![false; handles.fragments.len()];
    for pipeline in cache.pipelines() {
        match &pipeline.state {
            CachedPipelineState::Ok(_) => {
                state.snapshot.compiled += 1;
                if let bevy::material::descriptor::PipelineDescriptor::RenderPipelineDescriptor(d) =
                    &pipeline.descriptor
                    && let Some(fragment) = &d.fragment
                {
                    for (ready, required) in fragments_ready.iter_mut().zip(&handles.fragments) {
                        *ready |= &fragment.shader == required;
                    }
                }
            }
            CachedPipelineState::Err(
                ShaderCacheError::ShaderNotLoaded(_)
                | ShaderCacheError::ShaderImportNotYetAvailable,
            )
            | CachedPipelineState::Queued
            | CachedPipelineState::Creating(_) => state.snapshot.pending += 1,
            CachedPipelineState::Err(error) => {
                state
                    .snapshot
                    .error
                    .get_or_insert_with(|| format!("Station shader/pipeline failed: {error:?}"));
            }
        }
    }
    state.snapshot.ready = fragments_ready.iter().all(|ready| *ready)
        && state.snapshot.pending == 0
        && state.snapshot.error.is_none();
    if state.snapshot.error.is_some() && state.snapshot.pending == 0 && !state.error_drained {
        if let Err(error) = device.poll(PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(2)),
        }) {
            state.snapshot.error = Some(format!("Failed to drain renderer after error: {error}"));
        }
        state.error_drained = true;
    }
}
