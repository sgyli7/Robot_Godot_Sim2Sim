//! MultibodyJoints using the reduced-coordinates formalism or using constraints.
// Local modification: expose default-disabled read-only observation diagnostics.

#[cfg(all(feature = "alloc", feature = "sim2sim-observation"))]
mod sim2sim_observation;
#[cfg(all(feature = "alloc", feature = "sim2sim-observation"))]
pub use sim2sim_observation::MultibodyObservation;
#[cfg(all(feature = "alloc", feature = "sim2sim-observation"))]
pub(crate) use sim2sim_observation::{ContactObservationManifest, ContactConstraintIdentity, ContactSideOwner};

#[cfg(feature = "alloc")]
pub use self::multibody::{Multibody, MultibodyDofCoupling};
#[cfg(feature = "alloc")]
pub use self::multibody_ik::InverseKinematicsOption;
#[cfg(feature = "alloc")]
pub use self::multibody_joint::MultibodyJoint;
pub use self::multibody_joint_handle::{MultibodyIndex, MultibodyJointHandle};
#[cfg(feature = "alloc")]
pub use self::multibody_joint_set::{MultibodyJointSet, MultibodyLinkId};
#[cfg(feature = "alloc")]
pub use self::multibody_link::MultibodyLink;
#[cfg(feature = "alloc")]
pub use self::unit_multibody_joint::{
    unit_joint_friction_constraint, unit_joint_limit_constraint, unit_joint_motor_constraint,
};

#[cfg(feature = "alloc")]
mod multibody;
mod multibody_joint_handle;
#[cfg(feature = "alloc")]
mod multibody_joint_set;
#[cfg(feature = "alloc")]
mod multibody_link;
#[cfg(feature = "alloc")]
mod multibody_workspace;

#[cfg(feature = "alloc")]
mod multibody_ik;
#[cfg(feature = "alloc")]
mod multibody_joint;
#[cfg(feature = "alloc")]
mod unit_multibody_joint;

#[cfg(all(test, feature = "alloc"))]
mod multibody_regression_tests;
