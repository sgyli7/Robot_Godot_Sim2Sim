//! Development-only thread CPU clocks around Bevy's original extract callbacks.
//!
//! Wall time includes waiting and descheduling. Thread CPU time includes render
//! preparation executed on the main thread while it awaits the render world.

use bevy::{
    prelude::*,
    render::{
        Render, RenderApp, RenderSystems,
        diagnostic::{
            DiagnosticsRecorder, RecordDiagnostics, begin_diagnostics_frame, resolve_encoder,
        },
        pipelined_rendering::RenderExtractApp,
        renderer::{PendingCommandBuffers, RenderDevice, RenderGraph, RenderGraphSystems},
    },
};
use serde::Serialize;
use serde_json::{Value, json};
use std::{
    sync::{Arc, Mutex},
    time::Instant,
};

/// Shared samples of unchanged Bevy callbacks and live main-frame intervals.
#[derive(Clone, Resource)]
pub struct RenderCpuProfile(Arc<Mutex<ProfileState>>);

struct ProfileState {
    epoch: Instant,
    active_frame: Option<u64>,
    last_frame: Option<(Instant, Option<u64>)>,
    samples: Vec<PhaseSample>,
    main_budget_ns: u64,
}

#[derive(Resource, Default)]
struct RenderPhaseClock(Option<(u64, Instant, Option<u64>)>);

#[derive(Serialize)]
struct PhaseSample {
    phase: &'static str,
    thread_id: String,
    display_frame: u64,
    start_since_run_ns: u64,
    wall_ns: u64,
    thread_cpu_ns: Option<u64>,
}

impl RenderCpuProfile {
    /// Start collecting at the same active-frame boundary as the live receipt.
    pub fn mark_frame(&self, display_frame: u64) {
        let now = Instant::now();
        let cpu = thread_cpu_ns();
        let mut state = self.0.lock().unwrap();
        if let Some((previous, previous_cpu)) = state.last_frame {
            let sample = PhaseSample {
                phase: "main_frame_interval",
                thread_id: format!("{:?}", std::thread::current().id()),
                display_frame,
                start_since_run_ns: nanos(previous.duration_since(state.epoch)),
                wall_ns: nanos(now.duration_since(previous)),
                thread_cpu_ns: previous_cpu.zip(cpu).and_then(|(a, b)| b.checked_sub(a)),
            };
            state.samples.push(sample);
        }
        state.active_frame = Some(display_frame);
        state.last_frame = Some((now, cpu));
    }

    /// Return measured CPU time separately from enclosing wall-clock spans.
    pub fn report(&self) -> Value {
        let state = self.0.lock().unwrap();
        json!({
            "schema": "live_render_thread_cpu_v1",
            "cpu_clock_available": thread_cpu_ns().is_some(),
            "semantics": "single_thread_cpu_per_sample; main_frame_interval_extract_handoff_on_main_thread; render_schedule_on_render_calling_thread_excludes_parallel_workers; wall_minus_cpu_includes_waiting_and_descheduling; handoff_includes_extract_and_cooperative_render_tasks",
            "main_budget_probe_ns": state.main_budget_ns,
            "samples": state.samples,
        })
    }

    /// Exclude final screenshot and shutdown work after the active measurement.
    pub fn stop(&self) {
        self.0.lock().unwrap().active_frame = None;
    }

    fn measure(&self, phase: &'static str, callback: impl FnOnce()) {
        let (frame, epoch) = {
            let state = self.0.lock().unwrap();
            (state.active_frame, state.epoch)
        };
        let Some(display_frame) = frame else {
            callback();
            return;
        };
        let started = Instant::now();
        let cpu_start = thread_cpu_ns();
        callback();
        let cpu_end = thread_cpu_ns();
        let sample = PhaseSample {
            phase,
            thread_id: format!("{:?}", std::thread::current().id()),
            display_frame,
            start_since_run_ns: nanos(started.duration_since(epoch)),
            wall_ns: nanos(started.elapsed()),
            thread_cpu_ns: cpu_start.zip(cpu_end).and_then(|(a, b)| b.checked_sub(a)),
        };
        self.0.lock().unwrap().samples.push(sample);
    }
}

/// Wrap the original Bevy callbacks without changing schedules or channel order.
pub fn install(app: &mut App, epoch: Instant) -> Result<RenderCpuProfile, String> {
    let main_budget_ns = parse_main_budget(
        std::env::var("SAI_LAB_LIVE_PROFILE_MAIN_BUDGET_MS")
            .ok()
            .as_deref(),
    )?;
    if main_budget_ns > 0 && thread_cpu_ns().is_none() {
        return Err("main budget probe requires a thread CPU clock".into());
    }
    let profile = RenderCpuProfile(Arc::new(Mutex::new(ProfileState {
        epoch,
        active_frame: None,
        last_frame: None,
        samples: Vec::new(),
        main_budget_ns,
    })));
    let render_app = app
        .get_sub_app_mut(RenderApp)
        .ok_or("render app unavailable")?;
    let mut extract = render_app
        .take_extract()
        .ok_or("render extract unavailable")?;
    let extract_profile = profile.clone();
    render_app.set_extract(move |main, render| {
        extract_profile.measure("extract", || extract(main, render));
    });
    render_app.insert_resource(profile.clone());
    render_app.init_resource::<RenderPhaseClock>();
    render_app.add_systems(
        Render,
        (
            begin_render_cpu.before(RenderSystems::ExtractCommands),
            end_render_cpu.after(RenderSystems::PostCleanup),
        ),
    );
    render_app.add_systems(
        RenderGraph,
        (
            begin_gpu_frame
                .after(begin_diagnostics_frame)
                .before(RenderGraphSystems::Render),
            end_gpu_frame
                .after(RenderGraphSystems::Render)
                .before(resolve_encoder),
        ),
    );
    let extract_app = app
        .get_sub_app_mut(RenderExtractApp)
        .ok_or("pipelined extract app unavailable")?;
    let mut handoff = extract_app
        .take_extract()
        .ok_or("render handoff unavailable")?;
    let handoff_profile = profile.clone();
    extract_app.set_extract(move |main, render| {
        handoff_profile.measure("handoff", || handoff(main, render));
    });
    if main_budget_ns > 0 {
        app.insert_resource(profile.clone())
            .add_systems(Last, main_budget_probe);
    }
    Ok(profile)
}

fn parse_main_budget(value: Option<&str>) -> Result<u64, String> {
    let Some(value) = value else {
        return Ok(0);
    };
    let millis: f64 = value
        .parse()
        .map_err(|_| "main budget must be milliseconds".to_string())?;
    if !millis.is_finite() || !(0.0..=16.0).contains(&millis) {
        return Err("main budget must be between 0 and 16 milliseconds".into());
    }
    Ok((millis * 1_000_000.0).round() as u64)
}

// Exclusive systems run on the render schedule's calling thread at both ends.
fn begin_render_cpu(world: &mut World) {
    let frame = world
        .resource::<RenderCpuProfile>()
        .0
        .lock()
        .unwrap()
        .active_frame;
    world.resource_mut::<RenderPhaseClock>().0 =
        frame.map(|frame| (frame, Instant::now(), thread_cpu_ns()));
}

fn end_render_cpu(world: &mut World) {
    let cpu_end = thread_cpu_ns();
    let Some((display_frame, started, cpu_start)) =
        world.resource_mut::<RenderPhaseClock>().0.take()
    else {
        return;
    };
    let profile = world.resource::<RenderCpuProfile>();
    let mut state = profile.0.lock().unwrap();
    let sample = PhaseSample {
        phase: "render_schedule",
        thread_id: format!("{:?}", std::thread::current().id()),
        display_frame,
        start_since_run_ns: nanos(started.duration_since(state.epoch)),
        wall_ns: nanos(started.elapsed()),
        thread_cpu_ns: cpu_start.zip(cpu_end).and_then(|(a, b)| b.checked_sub(a)),
    };
    state.samples.push(sample);
}

// Push finished command buffers directly: push_encoder would append them after
// the render systems' already finished buffers and invalidate the start marker.
// Exclusive boundaries also preserve the recorder's thread-local span stack.
fn begin_gpu_frame(world: &mut World) {
    let mut encoder = world
        .resource::<RenderDevice>()
        .create_command_encoder(&default());
    world
        .resource::<DiagnosticsRecorder>()
        .begin_time_span(&mut encoder, "live_frame_gpu".into());
    world
        .resource_mut::<PendingCommandBuffers>()
        .push([encoder.finish()]);
}

fn end_gpu_frame(world: &mut World) {
    let mut encoder = world
        .resource::<RenderDevice>()
        .create_command_encoder(&default());
    world
        .resource::<DiagnosticsRecorder>()
        .end_time_span(&mut encoder);
    world
        .resource_mut::<PendingCommandBuffers>()
        .push([encoder.finish()]);
}

fn main_budget_probe(world: &mut World) {
    let profile = world.resource::<RenderCpuProfile>().clone();
    run_main_budget(&profile);
}

fn run_main_budget(profile: &RenderCpuProfile) {
    let budget = {
        let state = profile.0.lock().unwrap();
        if state.active_frame.is_none() {
            return;
        }
        state.main_budget_ns
    };
    profile.measure("main_budget_probe", || {
        let start = thread_cpu_ns().expect("budget probe CPU clock became unavailable");
        while thread_cpu_ns()
            .expect("budget probe CPU clock became unavailable")
            .saturating_sub(start)
            < budget
        {
            let mut value = std::hint::black_box(1_u64);
            for _ in 0..256 {
                value = value.wrapping_mul(1664525).wrapping_add(1013904223);
            }
            std::hint::black_box(value);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn main_budget_is_opt_in_and_bounded() {
        assert_eq!(parse_main_budget(None).unwrap(), 0);
        assert_eq!(parse_main_budget(Some("0")).unwrap(), 0);
        assert_eq!(parse_main_budget(Some("10.5")).unwrap(), 10_500_000);
        for value in ["", "-1", "17", "NaN", "inf", " 10", "ten"] {
            assert!(parse_main_budget(Some(value)).is_err(), "{value}");
        }
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn stopped_profile_does_not_record_shutdown_or_run_budget() {
        let profile = RenderCpuProfile(Arc::new(Mutex::new(ProfileState {
            epoch: Instant::now(),
            active_frame: None,
            last_frame: None,
            samples: Vec::new(),
            main_budget_ns: 1_000_000,
        })));
        run_main_budget(&profile);
        assert!(profile.0.lock().unwrap().samples.is_empty());
        profile.mark_frame(0);
        run_main_budget(&profile);
        assert_eq!(profile.0.lock().unwrap().samples.len(), 1);
        profile.stop();
        run_main_budget(&profile);
        assert_eq!(profile.0.lock().unwrap().samples.len(), 1);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn main_budget_system_runs_on_schedule_calling_thread() {
        let profile = RenderCpuProfile(Arc::new(Mutex::new(ProfileState {
            epoch: Instant::now(),
            active_frame: Some(0),
            last_frame: None,
            samples: Vec::new(),
            main_budget_ns: 1_000_000,
        })));
        let mut app = App::new();
        app.insert_resource(profile.clone())
            .add_systems(Last, main_budget_probe);
        app.world_mut().run_schedule(Last);
        let state = profile.0.lock().unwrap();
        assert_eq!(
            state.samples[0].thread_id,
            format!("{:?}", std::thread::current().id())
        );
        assert!(state.samples[0].thread_cpu_ns.unwrap() >= 1_000_000);
    }
}

fn nanos(duration: std::time::Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

#[cfg(target_os = "linux")]
fn thread_cpu_ns() -> Option<u64> {
    let mut stamp = std::mem::MaybeUninit::<libc::timespec>::uninit();
    // SAFETY: clock_gettime writes one timespec to the valid output pointer.
    if unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, stamp.as_mut_ptr()) } != 0 {
        return None;
    }
    // SAFETY: a successful clock_gettime initialized both timespec fields.
    let stamp = unsafe { stamp.assume_init() };
    u64::try_from(stamp.tv_sec)
        .ok()?
        .checked_mul(1_000_000_000)?
        .checked_add(u64::try_from(stamp.tv_nsec).ok()?)
}

#[cfg(not(target_os = "linux"))]
fn thread_cpu_ns() -> Option<u64> {
    None
}
