//! Explicit background coverage for source T2 and station transfer diagnostics.
//! This changes scene coverage only; robot, policy and action contracts remain
//! separate. No arbitrary path filter or alternate support geometry is exposed.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum T2BackgroundSelection {
    #[default]
    OriginalScene,
    StationTaskFixtures,
}

pub const T2_STATIC_GROUP: &str = "/World/envs/env_0/galileo_locomanip/native_static_group";
pub const T2_TABLE_OWNER: &str =
    "/World/envs/env_0/galileo_locomanip/TaskAssets/table/Geometry/sm_tabletop_a01_01";
pub const STATION_PHYSICAL_FIXTURES: [&str; 6] = [
    "/World/envs/env_0/galileo_locomanip/TaskAssets/shelf/SM_OpenIndustrialSteelShelving_A03_01/SM_OpenIndustrialSteelShelving_A01_Frame",
    "/World/envs/env_0/galileo_locomanip/TaskAssets/shelf/SM_OpenIndustrialSteelShelving_A03_01/SM_OpenIndustrialSteelShelving_A01_Joints",
    "/World/envs/env_0/galileo_locomanip/TaskAssets/shelf/SM_OpenIndustrialSteelShelving_A03_01/SM_OpenIndustrialSteelShelving_A01_Shelf",
    "/World/envs/env_0/galileo_locomanip/TaskAssets/table/Geometry/sm_tabletop_a01_01/sm_tabletop_a01_gaskets_01",
    "/World/envs/env_0/galileo_locomanip/TaskAssets/table/Geometry/sm_tabletop_a01_01/sm_tabletop_a01_legs_01",
    "/World/envs/env_0/galileo_locomanip/TaskAssets/table/Geometry/sm_tabletop_a01_01/sm_tabletop_a01_top_01",
];
pub const STATION_VISUAL_FIXTURES: [&str; 6] = [
    "/Lab/TaskAssets/shelf/SM_OpenIndustrialSteelShelving_A03_01/SM_OpenIndustrialSteelShelving_A01_Frame",
    "/Lab/TaskAssets/shelf/SM_OpenIndustrialSteelShelving_A03_01/SM_OpenIndustrialSteelShelving_A01_Joints",
    "/Lab/TaskAssets/shelf/SM_OpenIndustrialSteelShelving_A03_01/SM_OpenIndustrialSteelShelving_A01_Shelf",
    "/Lab/TaskAssets/table/Geometry/sm_tabletop_a01_01/sm_tabletop_a01_gaskets_01",
    "/Lab/TaskAssets/table/Geometry/sm_tabletop_a01_01/sm_tabletop_a01_legs_01",
    "/Lab/TaskAssets/table/Geometry/sm_tabletop_a01_01/sm_tabletop_a01_top_01",
];

impl T2BackgroundSelection {
    pub fn physical_fixture(self, body: &str, collider: &str) -> bool {
        match self {
            Self::OriginalScene => true,
            Self::StationTaskFixtures => STATION_PHYSICAL_FIXTURES
                .iter()
                .position(|path| *path == collider)
                .is_some_and(|i| {
                    body == if i < 3 {
                        T2_STATIC_GROUP
                    } else {
                        T2_TABLE_OWNER
                    }
                }),
        }
    }

    pub fn visual_fixture(self, mesh: &str, owner: Option<&str>) -> bool {
        match self {
            Self::OriginalScene => true,
            Self::StationTaskFixtures => STATION_VISUAL_FIXTURES
                .iter()
                .position(|path| *path == mesh)
                .is_some_and(|i| owner == if i < 3 { None } else { Some(T2_TABLE_OWNER) }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn station_support_coverage_rejects_foreign_geometry_and_owners() {
        let selection = T2BackgroundSelection::StationTaskFixtures;
        for (i, path) in STATION_PHYSICAL_FIXTURES.iter().enumerate() {
            let owner = if i < 3 {
                T2_STATIC_GROUP
            } else {
                T2_TABLE_OWNER
            };
            assert!(selection.physical_fixture(owner, path));
            assert!(!selection.physical_fixture("foreign owner", path));
            assert!(!selection.physical_fixture(owner, &format!("{path}/extra")));
        }
        for (i, path) in STATION_VISUAL_FIXTURES.iter().enumerate() {
            let owner = if i < 3 { None } else { Some(T2_TABLE_OWNER) };
            assert!(selection.visual_fixture(path, owner));
            assert!(!selection.visual_fixture(path, Some("foreign owner")));
        }
        assert!(!selection.physical_fixture(T2_STATIC_GROUP, "/Lab/BackgroundAssets/warehouse"));
        assert!(!selection.visual_fixture("/Lab/TaskAssets/power_drill", None));
    }
}
