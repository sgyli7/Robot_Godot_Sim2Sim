//! Hash-bound source collision metadata and engine-independent eligibility only.
//!
//! This admitted profile has no runtime registry, backend hooks or force API.

use std::{collections::BTreeMap, fs, path::Path};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{RobotError, definition::RobotDefinition};

/// Exact native binaries whose collision metadata was independently audited.
pub const LEG_NATIVE_MJB_SHA256: &str =
    "832e1f08a1e328d8498be565874691348b89bc13041ff0d9eebf3f06c6bdc47c";
pub const ROLLER_NATIVE_MJB_SHA256: &str =
    "ea54c7c2b0aa8fbc3e9a0e3a431214650cbd4bdbd934d6707bb0aa52ac475191";
pub const NATIVE_MUJOCO_RUNTIME_SHA256: &str =
    "f968dba54d54b8cee281023655a1702061221fce67348bf9ddec2f3ea3f31686";

/// Independently read from the frozen native MJBs, never from a definition.
/// Canonical UTF-8 JSON sorts every object key and has no whitespace/newline;
/// only definition_file_sha256 is excluded to permit genuine format conversion.
pub const LEG_NATIVE_COLLISION_SEMANTICS_SHA256: &str =
    "723e7b77b7b1c623d08b6954e2a18859c8d8a2c9606701c34fa71105dbfd0373";
pub const ROLLER_NATIVE_COLLISION_SEMANTICS_SHA256: &str =
    "602133a5ac5939862304e6ae3c46a2bb6fbbca49f4bb22b1102b821ed60d93d6";

/// Explicit masks are values, with no implicit environment mask.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeomMasks {
    pub contype: u32,
    pub conaffinity: u32,
}

/// MuJoCo's two directed bit intersections use OR, preserving all 32 bits.
pub const fn masks_allow(first: GeomMasks, second: GeomMasks) -> bool {
    first.contype & second.conaffinity != 0 || second.contype & first.conaffinity != 0
}

/// All source counts are required even when a supported feature has zero rows.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceCollisionCounts {
    pub nq: usize,
    pub nv: usize,
    pub nu: usize,
    pub nbody: usize,
    pub njnt: usize,
    pub ngeom: usize,
    pub nmesh: usize,
    pub nsensor: usize,
    pub nkey: usize,
    pub npair: usize,
    pub nexclude: usize,
    pub neq: usize,
    pub nflex: usize,
}

/// Interchange data is untrusted until loaded into the private validated wrapper.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceCollisionData {
    pub schema: String,
    pub family: String,
    pub definition_file_sha256: String,
    pub native_mjb_sha256: String,
    pub mujoco_version: String,
    pub native_runtime_sha256: String,
    pub counts: SourceCollisionCounts,
    pub body_parentid: Vec<usize>,
    pub body_weldid: Vec<usize>,
    pub jnt_type: Vec<u32>,
    pub jnt_bodyid: Vec<usize>,
    pub jnt_qposadr: Vec<usize>,
    pub jnt_dofadr: Vec<usize>,
    pub geom_bodyid: Vec<usize>,
    pub geom_type: Vec<u32>,
    pub geom_contype: Vec<u32>,
    pub geom_conaffinity: Vec<u32>,
    pub exclude_signature: Vec<u32>,
    pub pair_geom1: Vec<usize>,
    pub pair_geom2: Vec<usize>,
    pub eq_type: Vec<u32>,
    pub disableflags: u32,
    pub enableflags: u32,
    pub callback_absent: bool,
}

/// A source pair classification does not claim that its geometries overlap.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum SourcePairEligibility {
    Eligible,
    SameBody,
    SameWeld,
    NonWorldParent,
    MasksRejected,
}

impl SourcePairEligibility {
    pub const fn is_eligible(self) -> bool {
        matches!(self, Self::Eligible)
    }
}

/// Validated metadata is immutable and bound to one exact definition file.
pub struct SourceCollisionProfile {
    data: SourceCollisionData,
    file_sha256: String,
    native_semantic_sha256: String,
}

impl SourceCollisionProfile {
    pub fn load_json(
        file: &Path,
        expected_file_sha256: &str,
        definition: &RobotDefinition,
    ) -> Result<Self, RobotError> {
        let (bytes, file_sha256) = read_bound_bytes(file, expected_file_sha256)?;
        let data = serde_json::from_slice(&bytes).map_err(|error| invalid(error.to_string()))?;
        Self::from_data(data, file_sha256, definition)
    }

    pub fn load_ron(
        file: &Path,
        expected_file_sha256: &str,
        definition: &RobotDefinition,
    ) -> Result<Self, RobotError> {
        let (bytes, file_sha256) = read_bound_bytes(file, expected_file_sha256)?;
        let data = ron::de::from_bytes(&bytes).map_err(|error| invalid(error.to_string()))?;
        Self::from_data(data, file_sha256, definition)
    }

    fn from_data(
        data: SourceCollisionData,
        file_sha256: String,
        definition: &RobotDefinition,
    ) -> Result<Self, RobotError> {
        let native_semantic_sha256 = validate(&data, definition)?;
        Ok(Self {
            data,
            file_sha256,
            native_semantic_sha256,
        })
    }

    pub fn data(&self) -> &SourceCollisionData {
        &self.data
    }

    pub fn file_sha256(&self) -> &str {
        &self.file_sha256
    }

    pub fn native_semantic_sha256(&self) -> &str {
        &self.native_semantic_sha256
    }

    /// Development conversion serializes the already admitted immutable data.
    pub fn to_ron(&self) -> Result<String, RobotError> {
        ron::ser::to_string_pretty(&self.data, ron::ser::PrettyConfig::default())
            .map_err(|error| invalid(error.to_string()))
    }

    pub fn geom_masks(&self, source_geom: usize) -> Result<GeomMasks, RobotError> {
        if source_geom >= self.data.counts.ngeom {
            return Err(invalid("source collision geom index is out of range"));
        }
        Ok(GeomMasks {
            contype: self.data.geom_contype[source_geom],
            conaffinity: self.data.geom_conaffinity[source_geom],
        })
    }

    /// Classify two source geoms without any backend handles or cached contacts.
    pub fn source_geom_pair_eligibility(
        &self,
        first: usize,
        second: usize,
    ) -> Result<SourcePairEligibility, RobotError> {
        let first_masks = self.geom_masks(first)?;
        let second_masks = self.geom_masks(second)?;
        let body1 = self.data.geom_bodyid[first];
        let body2 = self.data.geom_bodyid[second];
        if body1 == body2 {
            return Ok(SourcePairEligibility::SameBody);
        }
        let weld1 = self.data.body_weldid[body1];
        let weld2 = self.data.body_weldid[body2];
        if weld1 == weld2 {
            return Ok(SourcePairEligibility::SameWeld);
        }
        let parent_weld1 = self.data.body_weldid[self.data.body_parentid[weld1]];
        let parent_weld2 = self.data.body_weldid[self.data.body_parentid[weld2]];
        if weld1 != 0 && weld2 != 0 && (weld1 == parent_weld2 || weld2 == parent_weld1) {
            return Ok(SourcePairEligibility::NonWorldParent);
        }
        Ok(if masks_allow(first_masks, second_masks) {
            SourcePairEligibility::Eligible
        } else {
            SourcePairEligibility::MasksRejected
        })
    }

    /// Only masks of an independent environment geom are evaluated here.
    /// Callers must provide masks explicitly; no ground/prop identity is inferred.
    pub fn environment_masks_allow(
        &self,
        source_geom: usize,
        environment: GeomMasks,
    ) -> Result<bool, RobotError> {
        Ok(masks_allow(self.geom_masks(source_geom)?, environment))
    }
}

fn validate(
    data: &SourceCollisionData,
    definition: &RobotDefinition,
) -> Result<String, RobotError> {
    let model = definition.model();
    let expected_mjb = match data.family.as_str() {
        "leg_allcollisions" => LEG_NATIVE_MJB_SHA256,
        "roller_allcollisions" => ROLLER_NATIVE_MJB_SHA256,
        _ => return Err(invalid("unsupported source collision family")),
    };
    if data.schema != "microduck_source_collision_profile_v1"
        || data.family != model.family
        || data.definition_file_sha256 != definition.file_sha256()
        || data.native_mjb_sha256 != expected_mjb
        || data.mujoco_version != "3.10.0"
        || data.native_runtime_sha256 != NATIVE_MUJOCO_RUNTIME_SHA256
    {
        return Err(invalid(
            "source collision identity does not match audited native definition",
        ));
    }
    if data.disableflags != 0 || data.enableflags != 0 || !data.callback_absent {
        return Err(invalid("unsupported source collision options or callback"));
    }
    let counts = &data.counts;
    if counts.npair != 0
        || counts.nexclude != 0
        || counts.neq != 0
        || counts.nflex != 0
        || !data.exclude_signature.is_empty()
        || !data.pair_geom1.is_empty()
        || !data.pair_geom2.is_empty()
        || !data.eq_type.is_empty()
    {
        return Err(invalid(
            "unsupported explicit collision pair, exclusion, equality or flex",
        ));
    }
    let expected = &model.counts;
    if counts.nq != expected.nq
        || counts.nv != expected.nv
        || counts.nu != expected.nu
        || counts.nbody != expected.nbody
        || counts.njnt != expected.njnt
        || counts.ngeom != expected.ngeom
        || counts.nmesh != expected.nmesh
        || counts.nsensor != expected.nsensor
        || counts.nkey != expected.nkey
        || counts.neq != expected.neq
    {
        return Err(invalid("source collision counts do not match definition"));
    }
    if data.body_weldid != (0..counts.nbody).collect::<Vec<_>>() {
        return Err(invalid("unsupported source collision weld topology"));
    }
    let fields = &model.fields;
    if data.body_parentid != fields.body_parentid
        || data.jnt_type != fields.jnt_type
        || data.jnt_bodyid != fields.jnt_bodyid
        || data.jnt_qposadr != fields.jnt_qposadr
        || data.jnt_dofadr != fields.jnt_dofadr
        || data.geom_bodyid != fields.geom_bodyid
        || data.geom_type != fields.geom_type
        || data.geom_contype != fields.geom_contype
        || data.geom_conaffinity != fields.geom_conaffinity
    {
        return Err(invalid("source collision arrays do not match definition"));
    }
    let semantic_sha256 = native_semantic_sha256(data)?;
    let expected_semantics = match data.family.as_str() {
        "leg_allcollisions" => LEG_NATIVE_COLLISION_SEMANTICS_SHA256,
        "roller_allcollisions" => ROLLER_NATIVE_COLLISION_SEMANTICS_SHA256,
        _ => return Err(invalid("unsupported source collision family")),
    };
    if semantic_sha256 != expected_semantics {
        return Err(invalid(
            "source collision semantic digest does not match audited native MJB",
        ));
    }
    Ok(semantic_sha256)
}

fn native_semantic_sha256(data: &SourceCollisionData) -> Result<String, RobotError> {
    let mut value = serde_json::to_value(data).map_err(|error| invalid(error.to_string()))?;
    value
        .as_object_mut()
        .ok_or_else(|| invalid("source collision semantic data must be an object"))?
        .remove("definition_file_sha256");
    let mut bytes = Vec::new();
    write_canonical_json(&value, &mut bytes)?;
    Ok(format!("{:x}", Sha256::digest(&bytes)))
}

fn write_canonical_json(value: &serde_json::Value, bytes: &mut Vec<u8>) -> Result<(), RobotError> {
    match value {
        serde_json::Value::Object(fields) => {
            bytes.push(b'{');
            // Explicit ordering remains stable if serde_json preserve_order is
            // enabled elsewhere in the dependency graph.
            let ordered: BTreeMap<_, _> = fields.iter().collect();
            for (index, (key, entry)) in ordered.into_iter().enumerate() {
                if index != 0 {
                    bytes.push(b',');
                }
                serde_json::to_writer(&mut *bytes, key)
                    .map_err(|error| invalid(error.to_string()))?;
                bytes.push(b':');
                write_canonical_json(entry, bytes)?;
            }
            bytes.push(b'}');
        }
        serde_json::Value::Array(entries) => {
            bytes.push(b'[');
            for (index, entry) in entries.iter().enumerate() {
                if index != 0 {
                    bytes.push(b',');
                }
                write_canonical_json(entry, bytes)?;
            }
            bytes.push(b']');
        }
        _ => serde_json::to_writer(bytes, value).map_err(|error| invalid(error.to_string()))?,
    }
    Ok(())
}

fn read_bound_bytes(file: &Path, expected_sha256: &str) -> Result<(Vec<u8>, String), RobotError> {
    if expected_sha256.len() != 64
        || !expected_sha256
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(invalid(
            "source collision expected file SHA256 is malformed",
        ));
    }
    let bytes = fs::read(file).map_err(|error| invalid(error.to_string()))?;
    let file_sha256 = format!("{:x}", Sha256::digest(&bytes));
    if file_sha256 != expected_sha256 {
        return Err(invalid("source collision file SHA256 mismatch"));
    }
    Ok((bytes, file_sha256))
}

fn invalid(message: impl Into<String>) -> RobotError {
    RobotError::Contract(message.into())
}

#[cfg(test)]
mod tests {
    use super::{GeomMasks, masks_allow};

    #[test]
    fn asymmetric_direction_and_high_bit_are_preserved() {
        let send = GeomMasks {
            contype: 1,
            conaffinity: 0,
        };
        let receive = GeomMasks {
            contype: 0,
            conaffinity: 1,
        };
        assert!(masks_allow(send, receive));
        assert!(masks_allow(receive, send));
        assert!(masks_allow(
            GeomMasks {
                contype: 1 << 31,
                conaffinity: 0
            },
            GeomMasks {
                contype: 0,
                conaffinity: 1 << 31
            }
        ));
        assert!(!masks_allow(send, send));
        assert!(!masks_allow(
            GeomMasks {
                contype: 0,
                conaffinity: 0
            },
            receive
        ));
    }
}
