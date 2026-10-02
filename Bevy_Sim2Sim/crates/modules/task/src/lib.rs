//! Local visual task decisions and explicitly bounded robot skill contracts.

pub mod decision;
#[cfg(feature = "ui")]
pub mod interactive;
pub mod policy;
pub mod types;
#[cfg(feature = "ui")]
pub mod ui;
