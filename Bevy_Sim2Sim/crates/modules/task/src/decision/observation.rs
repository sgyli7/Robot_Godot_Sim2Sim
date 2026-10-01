//! Validated RGB camera input and public user goals.

use std::io::Cursor;

use image::{DynamicImage, ImageFormat, ImageReader, RgbImage};
use serde::Serialize;

use crate::types::{ObservationStamp, RobotSelfState, TaskProfile};

use super::DecisionError;

const MAX_IMAGE_EDGE: u32 = 2048;
const MAX_PNG_BYTES: usize = 16 * 1024 * 1024;

/// A real camera RGB frame encoded as PNG. Construction validates dimensions and
/// decodes input PNGs, preventing arbitrary URLs or files entering model input.
#[derive(Debug, Clone)]
pub struct CameraRgb {
    name: String,
    width: u32,
    height: u32,
    png: Vec<u8>,
}

impl CameraRgb {
    pub fn from_rgb(
        name: impl Into<String>,
        width: u32,
        height: u32,
        pixels: Vec<u8>,
    ) -> Result<Self, DecisionError> {
        validate_dimensions(width, height)?;
        let image = RgbImage::from_raw(width, height, pixels)
            .ok_or_else(|| DecisionError::Observation("RGB byte count mismatch".into()))?;
        let mut png = Cursor::new(Vec::new());
        DynamicImage::ImageRgb8(image)
            .write_to(&mut png, ImageFormat::Png)
            .map_err(|error| DecisionError::Observation(error.to_string()))?;
        Self::validated(name.into(), width, height, png.into_inner())
    }

    pub fn from_png(name: impl Into<String>, png: Vec<u8>) -> Result<Self, DecisionError> {
        if png.len() > MAX_PNG_BYTES {
            return Err(DecisionError::Observation(
                "PNG is larger than 16 MiB".into(),
            ));
        }
        let (width, height) = ImageReader::with_format(Cursor::new(&png), ImageFormat::Png)
            .into_dimensions()
            .map_err(|error| DecisionError::Observation(error.to_string()))?;
        validate_dimensions(width, height)?;
        // Re-encode the decoded RGB pixels: strip metadata and any alpha channel.
        let rgb = image::load_from_memory_with_format(&png, ImageFormat::Png)
            .map_err(|error| DecisionError::Observation(error.to_string()))?
            .to_rgb8();
        Self::from_rgb(name, width, height, rgb.into_raw())
    }

    fn validated(
        name: String,
        width: u32,
        height: u32,
        png: Vec<u8>,
    ) -> Result<Self, DecisionError> {
        if name.is_empty() || name.len() > 64 || name.chars().any(char::is_control) {
            return Err(DecisionError::Observation("invalid camera name".into()));
        }
        if png.len() > MAX_PNG_BYTES {
            return Err(DecisionError::Observation(
                "PNG is larger than 16 MiB".into(),
            ));
        }
        Ok(Self {
            name,
            width,
            height,
            png,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }
    pub fn png(&self) -> &[u8] {
        &self.png
    }
}

fn validate_dimensions(width: u32, height: u32) -> Result<(), DecisionError> {
    if width == 0 || height == 0 || width > MAX_IMAGE_EDGE || height > MAX_IMAGE_EDGE {
        return Err(DecisionError::Observation(
            "camera dimensions must be 1..=2048".into(),
        ));
    }
    Ok(())
}

/// All perception input is explicitly supplied; this module cannot query ECS
/// bodies, physics truth, scene entity identifiers, or object transforms.
#[derive(Debug, Clone)]
pub struct ObservationSnapshot {
    pub stamp: ObservationStamp,
    pub camera: CameraRgb,
    pub robot: RobotSelfState,
}

impl ObservationSnapshot {
    pub fn validate(&self) -> Result<(), DecisionError> {
        let state = &self.robot;
        if state.joint_positions.is_empty()
            || state.joint_positions.len() > 64
            || state.joint_positions.len() != state.joint_velocities.len()
            || !state
                .joint_positions
                .iter()
                .chain(&state.joint_velocities)
                .chain(&state.base_velocity_mps)
                .chain(&state.projected_gravity)
                .all(|value| value.is_finite())
        {
            return Err(DecisionError::Observation(
                "invalid finite proprioception dimensions".into(),
            ));
        }
        Ok(())
    }
}

/// A user-visible goal and optional published marker/map description. This text
/// must not be filled from a hidden task-object state or an oracle evaluator.
#[derive(Debug, Clone, Serialize)]
pub struct TaskGoal {
    pub instruction: String,
    pub profile: TaskProfile,
    pub public_scene_description: String,
}

impl TaskGoal {
    pub fn validate(&self) -> Result<(), DecisionError> {
        if self.instruction.trim().is_empty()
            || self.instruction.len() > 4096
            || self.public_scene_description.len() > 4096
        {
            return Err(DecisionError::Configuration(
                "empty or oversized public task goal".into(),
            ));
        }
        Ok(())
    }
}
