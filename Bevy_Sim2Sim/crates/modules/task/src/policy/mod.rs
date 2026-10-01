//! Matched Arena task actions and a simulation-time action buffer.
//!
//! These decoded actions target the Arena decoupled controller. They are not
//! SONIC tokens. Empty or expired output requires the executor's validated stop
//! behavior; this module never manufactures a successful task result.

mod action;
mod observation;
mod queue;

pub use action::*;
pub use observation::*;
pub use queue::*;
