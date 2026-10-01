//! Engine-independent MicroDuck policy and BAM actuator contracts.

pub mod actuator;
pub mod basis;
pub mod body_pose;
pub mod collision_hull;
pub mod collision_profile;
pub mod contract;
pub mod definition;
pub mod delay;
pub mod goose;
pub mod joint_feedback;
pub mod kinematics;
pub mod policy;

pub use contract::{ACTION_DIMENSION, OBSERVATION_DIMENSION, RobotError};
