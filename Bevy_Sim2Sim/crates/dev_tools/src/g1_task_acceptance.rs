//! Judge a caller-supplied physics trace against the frozen placement window.
//!
//! This module is a development tool. It does not load a robot, a policy, a scene, or a hidden
//! goal pose, and it does not emit a control command. A passing [`EvaluationResult`] means only
//! that the supplied samples met the checks below. It does not mean the G1 pick-and-place
//! behavior is solved or qualified.
//!
//! Coordinates are meters. Linear velocity is meters per second. Angular velocity is radians per
//! second. The target contains the object when every object bound lies inside the target bound,
//! and equal limits count as inside. A speed passes only when its Euclidean norm is strictly
//! below the frozen limit. The stable window is the current uninterrupted suffix of placement-ready
//! samples whose simulation timestamps cover two seconds. Sample count is not a substitute for
//! that span. Inside one episode, every tick advances by exactly one and every time step matches
//! `1/sample_hz` within one microsecond. A missing or reordered sample is an error. A fall is
//! remembered for the episode until [`PlacementEvaluator::reset`].

use serde::Deserialize;
use serde::Deserializer;
use serde::Serialize;
use std::fmt;
use thiserror::Error;

/// Official physics sample rate, in hertz.
pub const OFFICIAL_SAMPLE_HZ: f64 = 50.0;

/// Continuous stable suffix required for acceptance, in seconds.
pub const REQUIRED_STABLE_SECONDS: f64 = 2.0;

/// Linear speed must stay strictly below this limit, in meters per second.
pub const LINEAR_SPEED_LIMIT_M_S: f64 = 0.02;

/// Angular speed must stay strictly below this limit, in radians per second.
pub const ANGULAR_SPEED_LIMIT_RAD_S: f64 = 0.1;

/// Absolute tolerance for one configured time step, in seconds.
pub const TIME_STEP_TOLERANCE_SECONDS: f64 = 1e-6;

/// Rounding budget for a two-second span. It is far smaller than one time step.
const DURATION_ROUNDING_EPSILON: f64 = 1e-12;

/// Extra units in the last place allowed when a step lands on the tolerance boundary.
const STEP_ERROR_ULPS: f64 = 8.0;

/// Sample rate used to interpret a caller-supplied trace.
///
/// The official rate is 50 Hz. Another rate is legal only when its nominal step is longer than
/// [`TIME_STEP_TOLERANCE_SECONDS`] and shorter than [`REQUIRED_STABLE_SECONDS`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct AcceptanceConfig {
    /// Physics sample rate, in hertz.
    pub sample_hz: f64,
}

impl AcceptanceConfig {
    /// Builds a config after checking the frozen limits and the sample-rate region.
    ///
    /// # Errors
    ///
    /// Returns [`AcceptanceError::InvalidSampleRate`] when the rate is outside the legal region,
    /// and [`AcceptanceError::InvalidAcceptanceRegion`] when a frozen limit is unusable.
    pub fn new(sample_hz: f64) -> Result<Self, AcceptanceError> {
        let config = Self { sample_hz };
        config.validate()?;
        Ok(config)
    }

    /// Returns the official 50 Hz configuration.
    pub fn official() -> Self {
        match Self::new(OFFICIAL_SAMPLE_HZ) {
            Ok(config) => config,
            Err(error) => {
                unreachable!("official sample rate is inside the legal region: {error}")
            }
        }
    }

    /// Checks the frozen limits and the configured sample rate.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::new`].
    pub fn validate(&self) -> Result<(), AcceptanceError> {
        validate_frozen_region()?;
        validate_sample_hz(self.sample_hz)
    }

    /// Nominal simulation step `1/sample_hz`, in seconds.
    ///
    /// # Errors
    ///
    /// Returns an error when the configuration is outside the legal region.
    pub fn nominal_step_seconds(&self) -> Result<f64, AcceptanceError> {
        self.validate()?;
        Ok(1.0 / self.sample_hz)
    }
}

impl Default for AcceptanceConfig {
    /// Returns the official 50 Hz configuration.
    fn default() -> Self {
        Self::official()
    }
}

impl<'de> Deserialize<'de> for AcceptanceConfig {
    /// Deserializes a sample rate and rejects values outside the legal region.
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        /// Plain rate field before validation.
        #[derive(Deserialize)]
        struct RawConfig {
            /// Unchecked sample rate, in hertz.
            sample_hz: f64,
        }

        let raw = RawConfig::deserialize(deserializer)?;
        Self::new(raw.sample_hz).map_err(serde::de::Error::custom)
    }
}

/// One physics sample collected by the caller.
///
/// The evaluator does not read these values from a simulator. Invalid numbers are rejected when
/// the sample is recorded.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct TruthSample {
    /// Episode that produced this sample.
    pub episode_id: u64,
    /// Monotonic physics tick inside the episode.
    pub tick: u64,
    /// Simulation time of this sample, in seconds.
    pub sim_time_seconds: f64,
    /// Minimum corner of the object axis-aligned box, in meters.
    pub object_bounds_min: [f64; 3],
    /// Maximum corner of the object axis-aligned box, in meters.
    pub object_bounds_max: [f64; 3],
    /// Minimum corner of the target axis-aligned box, in meters.
    pub target_bounds_min: [f64; 3],
    /// Maximum corner of the target axis-aligned box, in meters.
    pub target_bounds_max: [f64; 3],
    /// Object linear velocity, in meters per second.
    pub linear_velocity: [f64; 3],
    /// Object angular velocity, in radians per second.
    pub angular_velocity: [f64; 3],
    /// `true` when either hand is touching the object.
    pub hand_contact: bool,
    /// `true` when the target is supporting the object.
    pub target_support_contact: bool,
    /// `true` when the robot pose in this sample is standing.
    pub robot_standing: bool,
    /// `true` when the robot has fallen in this sample.
    pub robot_fallen: bool,
}

/// Why the frozen placement conditions are unmet.
///
/// Reasons are listed in this declaration order. A fall stays in the list for the rest of the
/// episode. The other placement reasons describe only the current stable suffix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FailureReason {
    /// The robot fell in at least one recorded sample of this episode.
    RobotFallen,
    /// The latest sample does not show a standing robot.
    RobotNotStanding,
    /// The object box extends outside the target box.
    ObjectOutsideTarget,
    /// A hand is touching the object.
    HandContact,
    /// The target is not providing support contact.
    MissingTargetSupport,
    /// The linear-speed norm has reached 0.02 m/s.
    LinearSpeedNotBelowLimit,
    /// The angular-speed norm has reached 0.1 rad/s.
    AngularSpeedNotBelowLimit,
    /// The current stable suffix is shorter than two seconds.
    InsufficientStableDuration,
    /// `finish` was called before any sample was recorded.
    NoSamples,
}

impl fmt::Display for FailureReason {
    /// Formats the unmet condition in English.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RobotFallen => formatter.write_str("the robot fell during this episode"),
            Self::RobotNotStanding => {
                formatter.write_str("the latest sample does not show the robot standing")
            }
            Self::ObjectOutsideTarget => {
                formatter.write_str("the object bounds extend outside the target bounds")
            }
            Self::HandContact => formatter.write_str("a hand is in contact with the object"),
            Self::MissingTargetSupport => {
                formatter.write_str("the object has no target support contact")
            }
            Self::LinearSpeedNotBelowLimit => write!(
                formatter,
                "linear speed has reached the {LINEAR_SPEED_LIMIT_M_S} m/s limit"
            ),
            Self::AngularSpeedNotBelowLimit => write!(
                formatter,
                "angular speed has reached the {ANGULAR_SPEED_LIMIT_RAD_S} rad/s limit"
            ),
            Self::InsufficientStableDuration => write!(
                formatter,
                "the continuous stable suffix is shorter than {REQUIRED_STABLE_SECONDS} seconds"
            ),
            Self::NoSamples => formatter.write_str("no physics sample was recorded"),
        }
    }
}

/// Judgment of the samples recorded so far.
///
/// `accepted` is true only when `failure_reasons` is empty. An empty episode, a short stable
/// suffix, and every physical miss stay unaccepted.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluationResult {
    /// `true` when every frozen condition currently holds.
    pub accepted: bool,
    /// Duration of the current uninterrupted stable suffix, in seconds.
    pub stable_duration_seconds: f64,
    /// Unmet conditions, in [`FailureReason`] declaration order.
    pub failure_reasons: Vec<FailureReason>,
}

/// Which sample box failed the region check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundsName {
    /// Object minimum and maximum corners.
    Object,
    /// Target minimum and maximum corners.
    Target,
}

impl fmt::Display for BoundsName {
    /// Formats the box name used in error text.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Object => "object bounds",
            Self::Target => "target bounds",
        })
    }
}

/// A trace or configuration that cannot be judged.
///
/// These errors are corrupt evidence. They are distinct from [`FailureReason`], which reports a
/// readable trace that missed the placement conditions.
#[derive(Debug, Clone, PartialEq, Error, Serialize, Deserialize)]
pub enum AcceptanceError {
    /// The sample rate is outside the finite positive step region.
    #[error(
        "sample rate must be finite and positive, with a nominal step longer than {tolerance_seconds} seconds and shorter than {stable_window_seconds} seconds"
    )]
    InvalidSampleRate {
        /// Fixed step tolerance, in seconds.
        tolerance_seconds: f64,
        /// Frozen stable window, in seconds.
        stable_window_seconds: f64,
    },
    /// A frozen duration or speed limit is no longer a finite positive value.
    #[error("frozen acceptance limits must be finite and positive")]
    InvalidAcceptanceRegion,
    /// A sample component is NaN or infinite.
    #[error("{field} is not finite")]
    NonFiniteField {
        /// Dotted sample field, with an axis index when the value is a vector.
        field: String,
    },
    /// A box has a minimum corner beyond its maximum corner.
    #[error("{bounds} axis {axis} has min greater than max")]
    IllegalBounds {
        /// Object or target box.
        bounds: BoundsName,
        /// Axis index: 0, 1, or 2.
        axis: u8,
    },
    /// A sample arrived with a different episode id before reset.
    #[error("episode changed from {previous} to {observed} without reset")]
    EpisodeChanged {
        /// Episode already being recorded.
        previous: u64,
        /// Episode id on the rejected sample.
        observed: u64,
    },
    /// The tick skipped, repeated, moved backward, or overflowed.
    #[error("tick {observed} is not the immediate successor of tick {previous}")]
    TickDisorder {
        /// Tick of the previous recorded sample.
        previous: u64,
        /// Tick of the rejected sample.
        observed: u64,
    },
    /// Simulation time did not advance by one nominal step.
    #[error(
        "sim time {observed_seconds} is not {nominal_step_seconds} seconds after {previous_seconds} within {tolerance_seconds} seconds"
    )]
    TimeDisorder {
        /// Simulation time of the previous recorded sample, in seconds.
        previous_seconds: f64,
        /// Simulation time of the rejected sample, in seconds.
        observed_seconds: f64,
        /// Expected step, in seconds.
        nominal_step_seconds: f64,
        /// Allowed absolute step error, in seconds.
        tolerance_seconds: f64,
    },
    /// The sample says the robot is standing and fallen together.
    #[error("robot_standing and robot_fallen cannot both be true")]
    ContradictoryRobotState,
    /// More samples were recorded than `u64` can count.
    #[error("sample count does not fit in u64")]
    SampleSequenceOverflow,
}

/// Latest recorded sample used to check tick and time order.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
struct SequenceCursor {
    /// Episode that owns the recorded suffix.
    episode_id: u64,
    /// Tick of the latest recorded sample.
    tick: u64,
    /// Simulation time of the latest recorded sample, in seconds.
    sim_time_seconds: f64,
}

/// Streaming judge for one placement episode at a time.
///
/// Feed samples with [`Self::record_sample`]. Call [`Self::finish`] at the end of the trace.
/// Both return the same physical judgment. A protocol error stays latched until [`Self::reset`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlacementEvaluator {
    /// Rate and frozen-limit gate for this judge.
    config: AcceptanceConfig,
    /// Previous sample, absent until the first sample is recorded.
    cursor: Option<SequenceCursor>,
    /// Simulation time where the current stable suffix began, in seconds.
    streak_start_time_seconds: Option<f64>,
    /// A fall was present in any recorded sample of this episode.
    ever_fallen: bool,
    /// Standing flag from the latest recorded sample.
    robot_standing: bool,
    /// Placement misses on the latest recorded sample.
    placement_failures: Vec<FailureReason>,
    /// Number of samples committed in this episode.
    samples_seen: u64,
    /// First protocol error, retained so later samples cannot pass.
    protocol_error: Option<AcceptanceError>,
}

impl PlacementEvaluator {
    /// Builds an empty judge after validating `config`.
    ///
    /// # Errors
    ///
    /// Returns a configuration error when the rate or a frozen limit is illegal.
    pub fn new(config: AcceptanceConfig) -> Result<Self, AcceptanceError> {
        config.validate()?;
        Ok(Self::from_config(config))
    }

    /// Builds an empty judge at the official 50 Hz rate.
    pub fn official() -> Self {
        match Self::new(AcceptanceConfig::official()) {
            Ok(evaluator) => evaluator,
            Err(error) => unreachable!("official acceptance configuration is valid: {error}"),
        }
    }

    /// Returns the configuration held by this judge.
    pub fn config(&self) -> AcceptanceConfig {
        self.config
    }

    /// Clears episode memory, including a fall and a latched protocol error.
    ///
    /// The sample rate stays in place. The next sample may use any episode id and tick.
    pub fn reset(&mut self) {
        let config = self.config;
        *self = Self::from_config(config);
    }

    /// Records one caller-supplied sample and returns the judgment of the trace so far.
    ///
    /// # Errors
    ///
    /// Returns a latched or newly detected [`AcceptanceError`] for an illegal rate, a non-finite
    /// value, an inverted box, a contradictory robot pose, an episode change, or tick/time
    /// disorder. The rejected sample is not committed.
    pub fn record_sample(
        &mut self,
        sample: &TruthSample,
    ) -> Result<EvaluationResult, AcceptanceError> {
        if let Some(error) = &self.protocol_error {
            return Err(error.clone());
        }
        if let Err(error) = self.config.validate() {
            return self.latch(error);
        }
        if let Err(error) = validate_sample(sample) {
            return self.latch(error);
        }
        let nominal_step = 1.0 / self.config.sample_hz;
        if let Some(cursor) = self.cursor
            && let Some(error) = sequence_error(&cursor, sample, nominal_step)
        {
            return self.latch(error);
        }
        self.commit_sample(sample)
    }

    /// Reports the current judgment, including a short stable suffix.
    ///
    /// # Errors
    ///
    /// Returns the latched protocol error, or a configuration error if the rate was changed to
    /// an illegal value after construction.
    pub fn finish(&self) -> Result<EvaluationResult, AcceptanceError> {
        if let Some(error) = &self.protocol_error {
            return Err(error.clone());
        }
        self.config.validate()?;
        Ok(self.evaluation())
    }

    /// Stores a protocol error and returns it.
    fn latch(&mut self, error: AcceptanceError) -> Result<EvaluationResult, AcceptanceError> {
        self.protocol_error = Some(error.clone());
        Err(error)
    }

    /// Commits a sample that has already passed validation and sequence checks.
    fn commit_sample(&mut self, sample: &TruthSample) -> Result<EvaluationResult, AcceptanceError> {
        let Some(next_count) = self.samples_seen.checked_add(1) else {
            return self.latch(AcceptanceError::SampleSequenceOverflow);
        };
        self.cursor = Some(SequenceCursor {
            episode_id: sample.episode_id,
            tick: sample.tick,
            sim_time_seconds: sample.sim_time_seconds,
        });
        self.samples_seen = next_count;
        if sample.robot_fallen {
            self.ever_fallen = true;
        }
        self.robot_standing = sample.robot_standing;
        let placement_failures = placement_failures(sample);
        if placement_failures.is_empty() && sample.robot_standing {
            if self.streak_start_time_seconds.is_none() {
                self.streak_start_time_seconds = Some(sample.sim_time_seconds);
            }
        } else {
            self.streak_start_time_seconds = None;
        }
        self.placement_failures = placement_failures;
        Ok(self.evaluation())
    }

    /// Builds an empty judge without checking the configuration again.
    fn from_config(config: AcceptanceConfig) -> Self {
        Self {
            config,
            cursor: None,
            streak_start_time_seconds: None,
            ever_fallen: false,
            robot_standing: false,
            placement_failures: Vec::new(),
            samples_seen: 0,
            protocol_error: None,
        }
    }

    /// Duration of the current stable suffix, in seconds.
    fn stable_duration_seconds(&self) -> f64 {
        match (self.streak_start_time_seconds, self.cursor) {
            (Some(start), Some(cursor)) => cursor.sim_time_seconds - start,
            _ => 0.0,
        }
    }

    /// Whether the current suffix spans the frozen two-second window.
    fn duration_satisfied(&self) -> bool {
        let span = self.stable_duration_seconds();
        span.is_finite() && span + DURATION_ROUNDING_EPSILON >= REQUIRED_STABLE_SECONDS
    }

    /// Collects every unmet condition for the samples committed so far.
    fn evaluation(&self) -> EvaluationResult {
        let stable_duration_seconds = self.stable_duration_seconds();
        let mut failure_reasons = Vec::new();
        if self.samples_seen == 0 {
            failure_reasons.push(FailureReason::NoSamples);
        } else {
            if self.ever_fallen {
                failure_reasons.push(FailureReason::RobotFallen);
            }
            if !self.robot_standing {
                failure_reasons.push(FailureReason::RobotNotStanding);
            }
            failure_reasons.extend(self.placement_failures.iter().copied());
            if !self.duration_satisfied() {
                failure_reasons.push(FailureReason::InsufficientStableDuration);
            }
        }
        EvaluationResult {
            accepted: failure_reasons.is_empty(),
            stable_duration_seconds,
            failure_reasons,
        }
    }
}

impl Default for PlacementEvaluator {
    /// Returns an empty official-rate judge.
    fn default() -> Self {
        Self::official()
    }
}

/// Rejects a frozen limit that cannot define the acceptance window.
fn validate_frozen_region() -> Result<(), AcceptanceError> {
    let limits = [
        REQUIRED_STABLE_SECONDS,
        LINEAR_SPEED_LIMIT_M_S,
        ANGULAR_SPEED_LIMIT_RAD_S,
        TIME_STEP_TOLERANCE_SECONDS,
    ];
    let squares = [
        LINEAR_SPEED_LIMIT_M_S * LINEAR_SPEED_LIMIT_M_S,
        ANGULAR_SPEED_LIMIT_RAD_S * ANGULAR_SPEED_LIMIT_RAD_S,
    ];
    let limits_ok = limits.iter().all(|limit| limit.is_finite() && *limit > 0.0);
    let squares_ok = squares
        .iter()
        .all(|square| square.is_finite() && *square > 0.0);
    let epsilon_ok = DURATION_ROUNDING_EPSILON.is_finite()
        && DURATION_ROUNDING_EPSILON > 0.0
        && DURATION_ROUNDING_EPSILON * 1_000.0 < TIME_STEP_TOLERANCE_SECONDS;
    let ulps_ok = STEP_ERROR_ULPS.is_finite() && STEP_ERROR_ULPS >= 1.0;
    if limits_ok && squares_ok && epsilon_ok && ulps_ok {
        Ok(())
    } else {
        Err(AcceptanceError::InvalidAcceptanceRegion)
    }
}

/// Rejects a sample rate whose step is outside the legal timing region.
fn validate_sample_hz(sample_hz: f64) -> Result<(), AcceptanceError> {
    let illegal = !sample_hz.is_finite() || sample_hz <= 0.0 || {
        let step = 1.0 / sample_hz;
        !step.is_finite() || step <= TIME_STEP_TOLERANCE_SECONDS || step >= REQUIRED_STABLE_SECONDS
    };
    if illegal {
        Err(AcceptanceError::InvalidSampleRate {
            tolerance_seconds: TIME_STEP_TOLERANCE_SECONDS,
            stable_window_seconds: REQUIRED_STABLE_SECONDS,
        })
    } else {
        Ok(())
    }
}

/// Rejects non-finite components, inverted boxes, and a contradictory robot pose.
fn validate_sample(sample: &TruthSample) -> Result<(), AcceptanceError> {
    require_finite(sample.sim_time_seconds, "sim_time_seconds")?;
    require_vector(sample.object_bounds_min, "object_bounds_min")?;
    require_vector(sample.object_bounds_max, "object_bounds_max")?;
    require_vector(sample.target_bounds_min, "target_bounds_min")?;
    require_vector(sample.target_bounds_max, "target_bounds_max")?;
    require_vector(sample.linear_velocity, "linear_velocity")?;
    require_vector(sample.angular_velocity, "angular_velocity")?;
    require_ordered_bounds(
        sample.object_bounds_min,
        sample.object_bounds_max,
        BoundsName::Object,
    )?;
    require_ordered_bounds(
        sample.target_bounds_min,
        sample.target_bounds_max,
        BoundsName::Target,
    )?;
    if sample.robot_standing && sample.robot_fallen {
        return Err(AcceptanceError::ContradictoryRobotState);
    }
    Ok(())
}

/// Rejects a non-finite scalar.
fn require_finite(value: f64, field: &str) -> Result<(), AcceptanceError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(AcceptanceError::NonFiniteField {
            field: field.to_string(),
        })
    }
}

/// Rejects a non-finite vector component.
fn require_vector(values: [f64; 3], label: &str) -> Result<(), AcceptanceError> {
    for (axis, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(AcceptanceError::NonFiniteField {
                field: format!("{label}[{axis}]"),
            });
        }
    }
    Ok(())
}

/// Rejects a box whose minimum exceeds its maximum on any axis.
fn require_ordered_bounds(
    min_corner: [f64; 3],
    max_corner: [f64; 3],
    bounds: BoundsName,
) -> Result<(), AcceptanceError> {
    for axis in 0..3 {
        if min_corner[axis] > max_corner[axis] {
            return Err(AcceptanceError::IllegalBounds {
                bounds,
                axis: axis as u8,
            });
        }
    }
    Ok(())
}

/// Returns the sequence error for a sample that follows `cursor`.
fn sequence_error(
    cursor: &SequenceCursor,
    sample: &TruthSample,
    nominal_step: f64,
) -> Option<AcceptanceError> {
    if sample.episode_id != cursor.episode_id {
        return Some(AcceptanceError::EpisodeChanged {
            previous: cursor.episode_id,
            observed: sample.episode_id,
        });
    }
    let Some(expected_tick) = cursor.tick.checked_add(1) else {
        return Some(AcceptanceError::TickDisorder {
            previous: cursor.tick,
            observed: sample.tick,
        });
    };
    if sample.tick != expected_tick {
        return Some(AcceptanceError::TickDisorder {
            previous: cursor.tick,
            observed: sample.tick,
        });
    }
    let actual_step = sample.sim_time_seconds - cursor.sim_time_seconds;
    if step_within_tolerance(actual_step, nominal_step) {
        None
    } else {
        Some(AcceptanceError::TimeDisorder {
            previous_seconds: cursor.sim_time_seconds,
            observed_seconds: sample.sim_time_seconds,
            nominal_step_seconds: nominal_step,
            tolerance_seconds: TIME_STEP_TOLERANCE_SECONDS,
        })
    }
}

/// Whether `actual_step` matches `nominal_step` within the fixed one-microsecond tolerance.
fn step_within_tolerance(actual_step: f64, nominal_step: f64) -> bool {
    let error = (actual_step - nominal_step).abs();
    if !error.is_finite() {
        return false;
    }
    let scale = actual_step.abs().max(nominal_step.abs());
    let rounding_slack = scale * f64::EPSILON * STEP_ERROR_ULPS;
    error <= TIME_STEP_TOLERANCE_SECONDS + rounding_slack
}

/// Placement misses on one sample, excluding the episode-level robot checks.
fn placement_failures(sample: &TruthSample) -> Vec<FailureReason> {
    let mut reasons = Vec::new();
    if !target_contains_object(sample) {
        reasons.push(FailureReason::ObjectOutsideTarget);
    }
    if sample.hand_contact {
        reasons.push(FailureReason::HandContact);
    }
    if !sample.target_support_contact {
        reasons.push(FailureReason::MissingTargetSupport);
    }
    if !speed_strictly_below(sample.linear_velocity, LINEAR_SPEED_LIMIT_M_S) {
        reasons.push(FailureReason::LinearSpeedNotBelowLimit);
    }
    if !speed_strictly_below(sample.angular_velocity, ANGULAR_SPEED_LIMIT_RAD_S) {
        reasons.push(FailureReason::AngularSpeedNotBelowLimit);
    }
    reasons
}

/// Whether the target box contains the object box, including shared faces.
fn target_contains_object(sample: &TruthSample) -> bool {
    (0..3).all(|axis| {
        sample.object_bounds_min[axis] >= sample.target_bounds_min[axis]
            && sample.object_bounds_max[axis] <= sample.target_bounds_max[axis]
    })
}

/// Whether the Euclidean speed is strictly below `limit`.
///
/// The squares are compared so a value sitting on the limit cannot pass through `sqrt` rounding.
fn speed_strictly_below(velocity: [f64; 3], limit: f64) -> bool {
    let mut square = 0.0;
    for component in velocity {
        square += component * component;
    }
    let limit_square = limit * limit;
    square.is_finite() && limit_square.is_finite() && square < limit_square
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Official two-second span at 50 Hz, measured in steps of 0.02 seconds.
    const OFFICIAL_SUCCESS_INTERVALS: u64 = 100;

    /// One frame short of the frozen window at 50 Hz: 99/50 = 1.98 seconds.
    const OFFICIAL_SHORT_INTERVALS: u64 = 99;

    fn official() -> PlacementEvaluator {
        PlacementEvaluator::official()
    }

    fn stable_sample(episode_id: u64, tick: u64, sim_time_seconds: f64) -> TruthSample {
        TruthSample {
            episode_id,
            tick,
            sim_time_seconds,
            object_bounds_min: [-0.05, -0.04, 0.80],
            object_bounds_max: [0.05, 0.04, 0.95],
            target_bounds_min: [-0.05, -0.04, 0.80],
            target_bounds_max: [0.05, 0.04, 0.95],
            linear_velocity: [0.019, 0.0, 0.0],
            angular_velocity: [0.0, 0.09, 0.0],
            hand_contact: false,
            target_support_contact: true,
            robot_standing: true,
            robot_fallen: false,
        }
    }

    fn push_window<F>(
        evaluator: &mut PlacementEvaluator,
        episode_id: u64,
        first_tick: u64,
        intervals: u64,
        start_seconds: f64,
        sample_hz: f64,
        mut mutator: F,
    ) -> Result<EvaluationResult, AcceptanceError>
    where
        F: FnMut(u64, &mut TruthSample),
    {
        let step = 1.0 / sample_hz;
        let mut last = None;
        for offset in 0..=intervals {
            let mut sample = stable_sample(
                episode_id,
                first_tick + offset,
                start_seconds + offset as f64 * step,
            );
            mutator(offset, &mut sample);
            last = Some(evaluator.record_sample(&sample)?);
        }
        Ok(last.expect("window contains a sample"))
    }

    #[track_caller]
    fn assert_accepted(result: &EvaluationResult) {
        assert!(result.accepted);
        assert!(result.failure_reasons.is_empty());
        assert!(result.stable_duration_seconds >= REQUIRED_STABLE_SECONDS);
    }

    #[track_caller]
    fn assert_reasons(result: &EvaluationResult, reasons: &[FailureReason]) {
        assert!(!result.accepted);
        assert_eq!(result.failure_reasons, reasons);
    }

    #[test]
    fn two_second_equal_bounds_window_is_accepted_and_later_contact_revokes_it() {
        let linear_square = 0.019_f64 * 0.019_f64;
        let angular_square = 0.09_f64 * 0.09_f64;
        assert!(linear_square < LINEAR_SPEED_LIMIT_M_S * LINEAR_SPEED_LIMIT_M_S);
        assert!(angular_square < ANGULAR_SPEED_LIMIT_RAD_S * ANGULAR_SPEED_LIMIT_RAD_S);

        let mut evaluator = official();
        assert_eq!(evaluator.config().sample_hz, OFFICIAL_SAMPLE_HZ);
        let short = push_window(
            &mut evaluator,
            4,
            1_000,
            OFFICIAL_SHORT_INTERVALS,
            25.0,
            OFFICIAL_SAMPLE_HZ,
            |_, _| {},
        )
        .unwrap();
        assert_reasons(&short, &[FailureReason::InsufficientStableDuration]);
        assert!((short.stable_duration_seconds - 1.98).abs() < 1e-12);

        let ready = evaluator
            .record_sample(&stable_sample(4, 1_100, 27.0))
            .unwrap();
        assert_accepted(&ready);
        assert_eq!(ready.stable_duration_seconds, REQUIRED_STABLE_SECONDS);
        assert_accepted(&evaluator.finish().unwrap());

        let mut contacted = stable_sample(4, 1_101, 27.0 + 1.0 / OFFICIAL_SAMPLE_HZ);
        contacted.hand_contact = true;
        let revoked = evaluator.record_sample(&contacted).unwrap();
        assert_reasons(
            &revoked,
            &[
                FailureReason::HandContact,
                FailureReason::InsufficientStableDuration,
            ],
        );
        assert_reasons(&evaluator.finish().unwrap(), &revoked.failure_reasons);
    }

    #[test]
    fn interrupted_standing_restarts_the_stable_window() {
        let mut evaluator = official();
        let result = push_window(
            &mut evaluator,
            1,
            0,
            150,
            0.0,
            OFFICIAL_SAMPLE_HZ,
            |offset, sample| sample.robot_standing = offset != 50,
        )
        .unwrap();
        assert_reasons(&result, &[FailureReason::InsufficientStableDuration]);
        assert!((result.stable_duration_seconds - 1.98).abs() < 1e-12);
        assert_accepted(
            &evaluator
                .record_sample(&stable_sample(1, 151, 3.02))
                .unwrap(),
        );
    }

    #[test]
    fn stable_span_of_1_98_seconds_is_not_acceptance() {
        let mut evaluator = official();
        assert_reasons(&evaluator.finish().unwrap(), &[FailureReason::NoSamples]);
        let result = push_window(
            &mut evaluator,
            1,
            0,
            OFFICIAL_SHORT_INTERVALS,
            0.0,
            OFFICIAL_SAMPLE_HZ,
            |_, _| {},
        )
        .unwrap();
        assert_eq!(result.stable_duration_seconds, 1.98);
        assert_reasons(&result, &[FailureReason::InsufficientStableDuration]);
        assert_reasons(&evaluator.finish().unwrap(), &result.failure_reasons);
    }

    #[test]
    fn object_outside_the_target_aabb_is_rejected() {
        let mut evaluator = official();
        let result = push_window(
            &mut evaluator,
            1,
            0,
            OFFICIAL_SUCCESS_INTERVALS,
            0.0,
            OFFICIAL_SAMPLE_HZ,
            |_, sample| {
                sample.object_bounds_max[0] = sample.target_bounds_max[0] + 1e-4;
            },
        )
        .unwrap();
        assert_reasons(
            &result,
            &[
                FailureReason::ObjectOutsideTarget,
                FailureReason::InsufficientStableDuration,
            ],
        );
        assert_eq!(result.stable_duration_seconds, 0.0);
        assert_reasons(&evaluator.finish().unwrap(), &result.failure_reasons);
    }

    #[test]
    fn hand_contact_throughout_the_window_is_rejected() {
        let mut evaluator = official();
        let result = push_window(
            &mut evaluator,
            1,
            0,
            OFFICIAL_SUCCESS_INTERVALS,
            0.0,
            OFFICIAL_SAMPLE_HZ,
            |_, sample| sample.hand_contact = true,
        )
        .unwrap();
        assert_reasons(
            &result,
            &[
                FailureReason::HandContact,
                FailureReason::InsufficientStableDuration,
            ],
        );
    }

    #[test]
    fn missing_target_support_is_rejected() {
        let mut evaluator = official();
        let result = push_window(
            &mut evaluator,
            1,
            0,
            OFFICIAL_SUCCESS_INTERVALS,
            0.0,
            OFFICIAL_SAMPLE_HZ,
            |_, sample| sample.target_support_contact = false,
        )
        .unwrap();
        assert_reasons(
            &result,
            &[
                FailureReason::MissingTargetSupport,
                FailureReason::InsufficientStableDuration,
            ],
        );
        assert_reasons(&evaluator.finish().unwrap(), &result.failure_reasons);
    }

    #[test]
    fn speed_at_or_above_the_limit_and_a_drop_are_rejected() {
        let mut at_linear_limit = official();
        let linear = push_window(
            &mut at_linear_limit,
            1,
            0,
            OFFICIAL_SUCCESS_INTERVALS,
            0.0,
            OFFICIAL_SAMPLE_HZ,
            |_, sample| sample.linear_velocity = [LINEAR_SPEED_LIMIT_M_S, 0.0, 0.0],
        )
        .unwrap();
        assert_reasons(
            &linear,
            &[
                FailureReason::LinearSpeedNotBelowLimit,
                FailureReason::InsufficientStableDuration,
            ],
        );

        let mut diagonal_linear = official();
        let diagonal = push_window(
            &mut diagonal_linear,
            1,
            0,
            OFFICIAL_SUCCESS_INTERVALS,
            0.0,
            OFFICIAL_SAMPLE_HZ,
            |_, sample| sample.linear_velocity = [0.015, 0.015, 0.0],
        )
        .unwrap();
        assert!(
            diagonal
                .failure_reasons
                .contains(&FailureReason::LinearSpeedNotBelowLimit)
        );

        let mut at_angular_limit = official();
        let angular = push_window(
            &mut at_angular_limit,
            1,
            0,
            OFFICIAL_SUCCESS_INTERVALS,
            0.0,
            OFFICIAL_SAMPLE_HZ,
            |_, sample| sample.angular_velocity = [0.0, ANGULAR_SPEED_LIMIT_RAD_S, 0.0],
        )
        .unwrap();
        assert_reasons(
            &angular,
            &[
                FailureReason::AngularSpeedNotBelowLimit,
                FailureReason::InsufficientStableDuration,
            ],
        );

        let mut dropped = official();
        let falling = push_window(
            &mut dropped,
            1,
            0,
            OFFICIAL_SUCCESS_INTERVALS,
            0.0,
            OFFICIAL_SAMPLE_HZ,
            |_, sample| {
                sample.object_bounds_min = [-0.05, -0.04, -1.00];
                sample.object_bounds_max = [0.05, 0.04, -0.80];
                sample.linear_velocity = [0.0, 0.0, -1.0];
                sample.target_support_contact = false;
            },
        )
        .unwrap();
        assert_reasons(
            &falling,
            &[
                FailureReason::ObjectOutsideTarget,
                FailureReason::MissingTargetSupport,
                FailureReason::LinearSpeedNotBelowLimit,
                FailureReason::InsufficientStableDuration,
            ],
        );
    }

    #[test]
    fn standing_after_a_fall_remains_a_failed_episode() {
        let mut recovered_pose = official();
        let recovered = push_window(
            &mut recovered_pose,
            7,
            0,
            OFFICIAL_SUCCESS_INTERVALS + 1,
            0.0,
            OFFICIAL_SAMPLE_HZ,
            |offset, sample| {
                if offset == 0 {
                    sample.robot_fallen = true;
                    sample.robot_standing = false;
                }
            },
        )
        .unwrap();
        assert_reasons(&recovered, &[FailureReason::RobotFallen]);
        assert!(recovered.stable_duration_seconds >= REQUIRED_STABLE_SECONDS);
        assert_reasons(
            &recovered_pose.finish().unwrap(),
            &[FailureReason::RobotFallen],
        );

        let mut fell_at_end = official();
        let ready = push_window(
            &mut fell_at_end,
            8,
            0,
            OFFICIAL_SUCCESS_INTERVALS,
            0.0,
            OFFICIAL_SAMPLE_HZ,
            |_, _| {},
        )
        .unwrap();
        assert_accepted(&ready);
        assert_accepted(&fell_at_end.finish().unwrap());
        let mut fallen = stable_sample(
            8,
            OFFICIAL_SUCCESS_INTERVALS + 1,
            REQUIRED_STABLE_SECONDS + 1.0 / OFFICIAL_SAMPLE_HZ,
        );
        fallen.robot_fallen = true;
        fallen.robot_standing = false;
        let after_fall = fell_at_end.record_sample(&fallen).unwrap();
        assert_reasons(
            &after_fall,
            &[
                FailureReason::RobotFallen,
                FailureReason::RobotNotStanding,
                FailureReason::InsufficientStableDuration,
            ],
        );
        assert_reasons(&fell_at_end.finish().unwrap(), &after_fall.failure_reasons);
    }

    #[test]
    fn reset_clears_a_fall_and_starts_a_new_episode() {
        let mut evaluator = official();
        let mut fallen = stable_sample(4, 20, 3.0);
        fallen.robot_fallen = true;
        fallen.robot_standing = false;
        let failed = evaluator.record_sample(&fallen).unwrap();
        assert!(failed.failure_reasons.contains(&FailureReason::RobotFallen));

        let mut other_episode = stable_sample(5, 0, 0.0);
        other_episode.robot_standing = true;
        let episode_error = evaluator.record_sample(&other_episode).unwrap_err();
        assert_eq!(
            episode_error,
            AcceptanceError::EpisodeChanged {
                previous: 4,
                observed: 5,
            }
        );
        assert!(evaluator.finish().is_err());

        evaluator.reset();
        let accepted = push_window(
            &mut evaluator,
            5,
            0,
            OFFICIAL_SUCCESS_INTERVALS,
            0.0,
            OFFICIAL_SAMPLE_HZ,
            |_, _| {},
        )
        .unwrap();
        assert_accepted(&accepted);
        assert_accepted(&evaluator.finish().unwrap());

        evaluator.reset();
        assert!(evaluator.record_sample(&stable_sample(4, 0, 0.0)).is_ok());
    }

    #[test]
    fn skipped_ticks_bad_timestamps_and_non_finite_values_are_errors() {
        let mut skipped = official();
        skipped.record_sample(&stable_sample(1, 0, 0.0)).unwrap();
        let skip_error = skipped
            .record_sample(&stable_sample(1, 2, 0.02))
            .unwrap_err();
        assert_eq!(
            skip_error,
            AcceptanceError::TickDisorder {
                previous: 0,
                observed: 2,
            }
        );
        assert_eq!(skipped.finish().unwrap_err(), skip_error);
        assert_eq!(
            skipped
                .record_sample(&stable_sample(1, 1, 0.02))
                .unwrap_err(),
            skip_error
        );

        let mut backward_tick = official();
        backward_tick
            .record_sample(&stable_sample(1, 5, 1.0))
            .unwrap();
        assert_eq!(
            backward_tick
                .record_sample(&stable_sample(1, 4, 1.02))
                .unwrap_err(),
            AcceptanceError::TickDisorder {
                previous: 5,
                observed: 4,
            }
        );

        let mut time_gap = official();
        time_gap.record_sample(&stable_sample(1, 0, 0.0)).unwrap();
        assert!(matches!(
            time_gap
                .record_sample(&stable_sample(1, 1, 0.04))
                .unwrap_err(),
            AcceptanceError::TimeDisorder { .. }
        ));

        let mut backward_time = official();
        backward_time
            .record_sample(&stable_sample(1, 0, 1.0))
            .unwrap();
        assert!(matches!(
            backward_time
                .record_sample(&stable_sample(1, 1, 0.5))
                .unwrap_err(),
            AcceptanceError::TimeDisorder { .. }
        ));

        let nominal = 1.0 / OFFICIAL_SAMPLE_HZ;
        let mut on_tolerance = official();
        on_tolerance
            .record_sample(&stable_sample(1, 0, 0.0))
            .unwrap();
        assert!(
            on_tolerance
                .record_sample(&stable_sample(1, 1, nominal + TIME_STEP_TOLERANCE_SECONDS))
                .is_ok()
        );
        assert!(
            on_tolerance
                .record_sample(&stable_sample(
                    1,
                    2,
                    nominal + TIME_STEP_TOLERANCE_SECONDS + nominal - TIME_STEP_TOLERANCE_SECONDS,
                ))
                .is_ok()
        );

        let mut over_tolerance = official();
        over_tolerance
            .record_sample(&stable_sample(1, 0, 0.0))
            .unwrap();
        assert!(matches!(
            over_tolerance
                .record_sample(&stable_sample(
                    1,
                    1,
                    nominal + TIME_STEP_TOLERANCE_SECONDS + 1e-9,
                ))
                .unwrap_err(),
            AcceptanceError::TimeDisorder { .. }
        ));

        let mut non_finite = official();
        let mut nan_velocity = stable_sample(1, 0, 0.0);
        nan_velocity.linear_velocity[0] = f64::NAN;
        let nan_error = non_finite.record_sample(&nan_velocity).unwrap_err();
        assert_eq!(
            nan_error,
            AcceptanceError::NonFiniteField {
                field: "linear_velocity[0]".to_string(),
            }
        );
        let trailing = push_window(
            &mut non_finite,
            1,
            0,
            OFFICIAL_SUCCESS_INTERVALS,
            0.0,
            OFFICIAL_SAMPLE_HZ,
            |_, _| {},
        );
        assert_eq!(trailing.unwrap_err(), nan_error);
        assert_eq!(non_finite.finish().unwrap_err(), nan_error);

        let mut infinite = official();
        let mut infinite_spin = stable_sample(1, 0, 0.0);
        infinite_spin.angular_velocity[2] = f64::INFINITY;
        assert_eq!(
            infinite.record_sample(&infinite_spin).unwrap_err(),
            AcceptanceError::NonFiniteField {
                field: "angular_velocity[2]".to_string(),
            }
        );

        let mut inverted = official();
        let mut inverted_object = stable_sample(1, 0, 0.0);
        inverted_object.object_bounds_min[0] = 0.2;
        inverted_object.object_bounds_max[0] = 0.1;
        assert_eq!(
            inverted.record_sample(&inverted_object).unwrap_err(),
            AcceptanceError::IllegalBounds {
                bounds: BoundsName::Object,
                axis: 0,
            }
        );

        let mut point = official();
        let mut degenerate = stable_sample(1, 0, 0.0);
        degenerate.object_bounds_min = [0.0, 0.0, 0.0];
        degenerate.object_bounds_max = [0.0, 0.0, 0.0];
        degenerate.target_bounds_min = [0.0, 0.0, 0.0];
        degenerate.target_bounds_max = [0.0, 0.0, 0.0];
        assert!(point.record_sample(&degenerate).is_ok());

        let mut contradictory = official();
        let mut both = stable_sample(1, 0, 0.0);
        both.robot_standing = true;
        both.robot_fallen = true;
        assert_eq!(
            contradictory.record_sample(&both).unwrap_err(),
            AcceptanceError::ContradictoryRobotState
        );

        non_finite.reset();
        assert!(non_finite.record_sample(&stable_sample(1, 0, 0.0)).is_ok());
    }

    #[test]
    fn sample_rate_is_configurable_and_rejects_values_outside_the_legal_region() {
        let official_config = AcceptanceConfig::official();
        assert_eq!(official_config.sample_hz, 50.0);
        assert_eq!(AcceptanceConfig::default().sample_hz, 50.0);
        assert_eq!(official_config.nominal_step_seconds().unwrap(), 1.0 / 50.0);
        assert!(AcceptanceConfig::new(100.0).is_ok());
        assert!(AcceptanceConfig::new(1.0).is_ok());
        for illegal in [0.0, -5.0, 0.5, 1_000_000.0, f64::NAN, f64::INFINITY] {
            assert!(
                matches!(
                    AcceptanceConfig::new(illegal),
                    Err(AcceptanceError::InvalidSampleRate { .. })
                ),
                "rate {illegal} was accepted"
            );
        }
        assert!(matches!(
            PlacementEvaluator::new(AcceptanceConfig {
                sample_hz: f64::NAN,
            }),
            Err(AcceptanceError::InvalidSampleRate { .. })
        ));

        let mut slow = PlacementEvaluator::new(AcceptanceConfig::new(1.0).unwrap()).unwrap();
        let one_second = push_window(&mut slow, 1, 0, 1, 0.0, 1.0, |_, _| {}).unwrap();
        assert_reasons(&one_second, &[FailureReason::InsufficientStableDuration]);
        assert_eq!(one_second.stable_duration_seconds, 1.0);
        slow.reset();
        let two_seconds = push_window(&mut slow, 1, 0, 2, 0.0, 1.0, |_, _| {}).unwrap();
        assert_accepted(&two_seconds);

        let parsed: AcceptanceConfig = serde_json::from_str(r#"{"sample_hz":50}"#).unwrap();
        assert_eq!(parsed, AcceptanceConfig::official());
        assert!(serde_json::from_str::<AcceptanceConfig>(r#"{"sample_hz":0}"#).is_err());
        assert!(serde_json::from_str::<AcceptanceConfig>(r#"{"sample_hz":1e20}"#).is_err());
    }

    #[test]
    fn one_hundred_hertz_still_requires_a_two_second_span() {
        let config = AcceptanceConfig::new(100.0).unwrap();
        let mut short = PlacementEvaluator::new(config).unwrap();
        let one_second = push_window(&mut short, 1, 0, 100, 0.0, 100.0, |_, _| {}).unwrap();
        assert_eq!(one_second.stable_duration_seconds, 1.0);
        assert_reasons(&one_second, &[FailureReason::InsufficientStableDuration]);

        let mut full = PlacementEvaluator::new(config).unwrap();
        let accepted = push_window(&mut full, 1, 0, 200, 0.0, 100.0, |_, _| {}).unwrap();
        assert_accepted(&accepted);
        assert_eq!(accepted.stable_duration_seconds, REQUIRED_STABLE_SECONDS);
    }

    #[test]
    fn separated_stable_segments_do_not_add_to_two_seconds() {
        let mut evaluator = official();
        let first =
            push_window(&mut evaluator, 1, 0, 60, 0.0, OFFICIAL_SAMPLE_HZ, |_, _| {}).unwrap();
        assert_eq!(first.stable_duration_seconds, 1.2);
        assert_reasons(&first, &[FailureReason::InsufficientStableDuration]);
        let mut contact = stable_sample(1, 61, 61.0 / OFFICIAL_SAMPLE_HZ);
        contact.hand_contact = true;
        let interrupted = evaluator.record_sample(&contact).unwrap();
        assert!(
            interrupted
                .failure_reasons
                .contains(&FailureReason::HandContact)
        );
        let suffix = push_window(
            &mut evaluator,
            1,
            62,
            60,
            62.0 / OFFICIAL_SAMPLE_HZ,
            OFFICIAL_SAMPLE_HZ,
            |_, _| {},
        )
        .unwrap();
        assert_eq!(suffix.stable_duration_seconds, 1.2);
        assert_reasons(&suffix, &[FailureReason::InsufficientStableDuration]);
        assert_reasons(&evaluator.finish().unwrap(), &suffix.failure_reasons);
    }

    #[test]
    fn serde_roundtrip_preserves_config_results_and_checkpoints() {
        let mut evaluator = official();
        let recorded = push_window(
            &mut evaluator,
            3,
            10,
            OFFICIAL_SUCCESS_INTERVALS,
            4.0,
            OFFICIAL_SAMPLE_HZ,
            |_, _| {},
        )
        .unwrap();
        assert_accepted(&recorded);
        let before = evaluator.finish().unwrap();
        assert_accepted(&before);

        let json = serde_json::to_string(&evaluator).unwrap();
        let restored: PlacementEvaluator = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.finish().unwrap(), before);

        let encoded = serde_json::to_string(&before).unwrap();
        let decoded: EvaluationResult = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, before);
        assert_eq!(
            serde_json::from_str::<FailureReason>(r#""RobotFallen""#).unwrap(),
            FailureReason::RobotFallen
        );
    }
}
