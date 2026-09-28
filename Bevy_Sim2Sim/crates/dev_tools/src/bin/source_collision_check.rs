//! Compare pure typed eligibility against frozen actual native MuJoCo contacts.

use std::{collections::HashMap, error::Error, fs, io::Write, path::Path};

use robot_minigame::{
    collision_profile::{GeomMasks, SourceCollisionProfile, masks_allow},
    definition::RobotDefinition,
};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

type CheckResult<T> = Result<T, Box<dyn Error>>;

#[derive(Deserialize)]
struct Artifact {
    path: String,
    sha256: String,
    bytes: usize,
}

#[derive(Deserialize)]
struct OracleEntry {
    name: String,
    result: Artifact,
}

#[derive(Deserialize)]
struct OracleManifest {
    schema: String,
    inputs: Artifact,
    mask_cases: usize,
    family_cases: usize,
    results: Vec<OracleEntry>,
}

#[derive(Deserialize)]
struct OracleAttributes {
    geom_contype: Vec<u32>,
    geom_conaffinity: Vec<u32>,
}

#[derive(Deserialize)]
struct NativePair {
    pair: [i64; 2],
    native_contact_generated: bool,
}

#[derive(Deserialize)]
struct NativeFixture {
    schema: String,
    name: String,
    native_attributes: OracleAttributes,
    world_fixture_masks: Option<[u32; 2]>,
    all_pairs: Vec<NativePair>,
    actual_ncon: usize,
    physics_integrations: u64,
    policy_inferences: u64,
}

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn bound_bytes(path: &Path, expected: &str) -> CheckResult<Vec<u8>> {
    let bytes = fs::read(path)?;
    if hash(&bytes) != expected {
        return Err(format!("input SHA256 mismatch: {}", path.display()).into());
    }
    Ok(bytes)
}

fn artifact_bytes(root: &Path, artifact: &Artifact) -> CheckResult<Vec<u8>> {
    let path = Path::new(&artifact.path);
    if path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, std::path::Component::Normal(_)))
    {
        return Err("oracle artifact path must remain inside its directory".into());
    }
    let bytes = bound_bytes(&root.join(path), &artifact.sha256)?;
    if bytes.len() != artifact.bytes {
        return Err("oracle artifact byte length mismatch".into());
    }
    Ok(bytes)
}

fn write_new(path: &Path, bytes: &[u8]) -> CheckResult<()> {
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?
        .write_all(bytes)?;
    Ok(())
}

fn load_profile(
    path: &Path,
    expected: &str,
    definition: &RobotDefinition,
) -> CheckResult<SourceCollisionProfile> {
    Ok(match path.extension().and_then(|value| value.to_str()) {
        Some("json") => SourceCollisionProfile::load_json(path, expected, definition)?,
        Some("ron") => SourceCollisionProfile::load_ron(path, expected, definition)?,
        _ => return Err("profile must be actual JSON or RON".into()),
    })
}

fn compare_oracle(
    profile: &SourceCollisionProfile,
    path: &Path,
    expected: &str,
) -> CheckResult<Value> {
    let root = path
        .parent()
        .ok_or("oracle manifest lacks parent directory")?;
    let manifest: OracleManifest = serde_json::from_slice(&bound_bytes(path, expected)?)?;
    if manifest.schema != "native_collision_eligibility_oracle_manifest_v1"
        || manifest.mask_cases != 256
        || manifest.family_cases != 8
    {
        return Err("unexpected independently frozen native oracle identity".into());
    }
    let provenance: Value = serde_json::from_slice(&artifact_bytes(root, &manifest.inputs)?)?;
    let source_matches = provenance["sources"]
        .as_array()
        .ok_or("missing native sources")?
        .iter()
        .filter(|source| {
            source["family"] == profile.data().family
                && source["native_mjb"]["sha256"] == profile.data().native_mjb_sha256
        })
        .count();
    if source_matches != 1 {
        return Err("native oracle is not bound to this profile's actual MJB".into());
    }
    let mut mask_cases = 0usize;
    let mut mask_pairs = 0usize;
    let mut family_pairs = 0usize;
    let mut native_contacts = 0usize;
    let mut cases = Vec::new();
    for entry in manifest.results {
        let is_mask = entry.name.starts_with("mask_");
        let is_family = entry
            .name
            .starts_with(&format!("{}_", profile.data().family));
        if !is_mask && !is_family {
            continue;
        }
        let fixture: NativeFixture = serde_json::from_slice(&artifact_bytes(root, &entry.result)?)?;
        if fixture.schema != "native_mujoco_collision_eligibility_v1"
            || fixture.name != entry.name
            || fixture.physics_integrations != 0
            || fixture.policy_inferences != 0
        {
            return Err("native eligibility fixture scope or identity changed".into());
        }
        let mut actual_allowed = 0usize;
        for pair in &fixture.all_pairs {
            let [first, second] = pair.pair;
            let computed = if is_mask {
                let first = usize::try_from(first)?;
                let second = usize::try_from(second)?;
                masks_allow(
                    GeomMasks {
                        contype: *fixture
                            .native_attributes
                            .geom_contype
                            .get(first)
                            .ok_or("native mask index")?,
                        conaffinity: *fixture
                            .native_attributes
                            .geom_conaffinity
                            .get(first)
                            .ok_or("native affinity index")?,
                    },
                    GeomMasks {
                        contype: *fixture
                            .native_attributes
                            .geom_contype
                            .get(second)
                            .ok_or("native mask index")?,
                        conaffinity: *fixture
                            .native_attributes
                            .geom_conaffinity
                            .get(second)
                            .ok_or("native affinity index")?,
                    },
                )
            } else if first == -1 || second == -1 {
                let [contype, conaffinity] = fixture
                    .world_fixture_masks
                    .ok_or("native world masks missing")?;
                let geom = usize::try_from(if first == -1 { second } else { first })?;
                profile.environment_masks_allow(
                    geom,
                    GeomMasks {
                        contype,
                        conaffinity,
                    },
                )?
            } else {
                profile
                    .source_geom_pair_eligibility(
                        usize::try_from(first)?,
                        usize::try_from(second)?,
                    )?
                    .is_eligible()
            };
            if computed != pair.native_contact_generated {
                return Err(format!(
                    "native eligibility mismatch: {} {:?}; native={}, computed={}",
                    fixture.name, pair.pair, pair.native_contact_generated, computed
                )
                .into());
            }
            actual_allowed += usize::from(pair.native_contact_generated);
        }
        if actual_allowed != fixture.actual_ncon {
            return Err("native fixture ncon does not cover its pair identity set".into());
        }
        if is_mask {
            mask_cases += 1;
            mask_pairs += fixture.all_pairs.len();
        } else {
            family_pairs += fixture.all_pairs.len();
            native_contacts += fixture.actual_ncon;
            cases.push(json!({"name":fixture.name,"pairs":fixture.all_pairs.len(),"native_ncon":fixture.actual_ncon}));
        }
    }
    let expected_family_pairs = match profile.data().family.as_str() {
        "leg_allcollisions" => 13203,
        "roller_allcollisions" => 15931,
        _ => return Err("unsupported profile family".into()),
    };
    if mask_cases != 256
        || mask_pairs != 256
        || cases.len() != 4
        || family_pairs != expected_family_pairs
    {
        return Err("independent native oracle comparison did not cover all expected pairs".into());
    }
    Ok(
        json!({"mask_cases":mask_cases,"mask_pair_comparisons":mask_pairs,
        "family_cases":cases,"family_pair_comparisons":family_pairs,"family_native_contacts":native_contacts,
        "total_pair_comparisons":mask_pairs+family_pairs,"all_native_pairs_equal":true,
        "nominal_same_weld_and_disabled_parent_not_admitted_as_profiles":true}),
    )
}

fn negative_loaders(
    profile: &SourceCollisionProfile,
    definition: &RobotDefinition,
    output: &Path,
    ron_text: &str,
) -> CheckResult<Value> {
    fs::create_dir(output)?;
    let base = serde_json::to_value(profile.data())?;
    let mut candidates: Vec<(String, Value)> = Vec::new();
    for name in base
        .as_object()
        .ok_or("profile data must be object")?
        .keys()
    {
        let mut changed = base.clone();
        changed.as_object_mut().unwrap().remove(name);
        candidates.push((format!("missing_{name}"), changed));
    }
    for name in base["counts"]
        .as_object()
        .ok_or("counts must be object")?
        .keys()
    {
        let mut changed = base.clone();
        changed["counts"].as_object_mut().unwrap().remove(name);
        candidates.push((format!("missing_count_{name}"), changed));
    }
    for name in [
        "body_parentid",
        "body_weldid",
        "jnt_type",
        "jnt_bodyid",
        "jnt_qposadr",
        "jnt_dofadr",
        "geom_bodyid",
        "geom_type",
        "geom_contype",
        "geom_conaffinity",
    ] {
        let mut changed = base.clone();
        changed[name].as_array_mut().unwrap().pop();
        candidates.push((format!("short_{name}"), changed));
        let mut changed = base.clone();
        let value = changed[name][0].as_u64().ok_or("array scalar")?;
        changed[name][0] = json!(value + 1);
        candidates.push((format!("changed_{name}"), changed));
    }
    for (name, value) in [
        ("schema", json!("unknown_profile")),
        ("family", json!("new_robot_family")),
        ("definition_file_sha256", json!("0".repeat(64))),
        ("native_mjb_sha256", json!("0".repeat(64))),
        ("mujoco_version", json!("other")),
        ("native_runtime_sha256", json!("0".repeat(64))),
        ("disableflags", json!(1024)),
        ("enableflags", json!(16)),
        ("callback_absent", json!(false)),
        ("exclude_signature", json!([1])),
        ("pair_geom1", json!([0])),
        ("pair_geom2", json!([1])),
        ("eq_type", json!([1])),
        ("unknown_field", json!(true)),
    ] {
        let mut changed = base.clone();
        changed[name] = value;
        candidates.push((format!("unsupported_{name}"), changed));
    }
    for name in ["nq", "nbody", "ngeom", "npair", "nexclude", "neq", "nflex"] {
        let mut changed = base.clone();
        changed["counts"][name] = json!(base["counts"][name].as_u64().unwrap() + 1);
        candidates.push((format!("changed_count_{name}"), changed));
    }
    let mut changed = base.clone();
    changed["body_weldid"][2] = json!(1);
    candidates.push(("nominal_nonidentity_weld_not_supported".into(), changed));
    let mut evidence = Vec::new();
    for (name, value) in candidates {
        let bytes = serde_json::to_vec_pretty(&value)?;
        let path = output.join(format!("{name}.json"));
        write_new(&path, &bytes)?;
        let rejection = SourceCollisionProfile::load_json(&path, &hash(&bytes), definition).err();
        let rejection = rejection.ok_or_else(|| format!("malformed profile admitted: {name}"))?;
        evidence.push(json!({"name":name,"format":"json","file_sha256":hash(&bytes),"rejection":rejection.to_string()}));
    }
    let valid_json = serde_json::to_vec_pretty(profile.data())?;
    let hash_path = output.join("valid_data_wrong_expected_hash.json");
    write_new(&hash_path, &valid_json)?;
    for expected in ["0".repeat(64), "bad_hash".into()] {
        let error = SourceCollisionProfile::load_json(&hash_path, &expected, definition)
            .err()
            .ok_or("wrong expected file hash admitted")?;
        evidence.push(json!({"name":"file_hash_rejection","format":"json","expected":expected,"rejection":error.to_string()}));
    }
    let mut ron_candidates = Vec::new();
    for name in [
        "schema",
        "native_mjb_sha256",
        "disableflags",
        "callback_absent",
    ] {
        let changed = ron_text
            .lines()
            .filter(|line| !line.trim_start().starts_with(&format!("{name}:")))
            .collect::<Vec<_>>()
            .join("\n");
        if changed.len() == ron_text.len() {
            return Err("RON negative mutation did not change source".into());
        }
        ron_candidates.push((format!("ron_missing_{name}"), changed));
    }
    for (name, old, replacement) in [
        (
            "ron_wrong_native_hash",
            profile.data().native_mjb_sha256.as_str(),
            "0000000000000000000000000000000000000000000000000000000000000000",
        ),
        (
            "ron_disabled_parent",
            "disableflags: 0",
            "disableflags: 1024",
        ),
        ("ron_nonempty_pair", "pair_geom1: []", "pair_geom1: [0]"),
    ] {
        let changed = ron_text.replace(old, replacement);
        if changed == ron_text {
            return Err("RON negative mutation did not change source".into());
        }
        ron_candidates.push((name.into(), changed));
    }
    for (name, text) in ron_candidates {
        let path = output.join(format!("{name}.ron"));
        write_new(&path, text.as_bytes())?;
        let error = SourceCollisionProfile::load_ron(&path, &hash(text.as_bytes()), definition)
            .err()
            .ok_or_else(|| format!("malformed RON profile admitted: {name}"))?;
        evidence.push(json!({"name":name,"format":"ron","file_sha256":hash(text.as_bytes()),"rejection":error.to_string()}));
    }
    if profile.geom_masks(usize::MAX).is_ok()
        || profile.source_geom_pair_eligibility(0, usize::MAX).is_ok()
        || profile
            .environment_masks_allow(
                usize::MAX,
                GeomMasks {
                    contype: 1,
                    conaffinity: 1,
                },
            )
            .is_ok()
    {
        return Err("out of range geom eligibility was admitted".into());
    }
    Ok(
        json!({"actual_loader_refusals":evidence.len(),"all_rejected":true,"cases":evidence,
        "index_boundary_refusals":3,"nominal_weld_and_disabled_parent_rejected":true,
        "physics_integrations":0,"policy_inferences":0}),
    )
}

fn coedited_native_identity_refusals(
    profile: &SourceCollisionProfile,
    definition: &RobotDefinition,
    output: &Path,
) -> CheckResult<Value> {
    fs::create_dir(output)?;
    let mut model = serde_json::to_value(definition.model())?;
    let original_profile = serde_json::to_value(profile.data())?;
    let mut evidence = Vec::new();
    for (field, index, replacement) in [
        ("body_parentid", 2, 0),
        ("geom_contype", 0, 1),
        ("geom_conaffinity", 0, 1),
    ] {
        let original = model["fields"][field][index].clone();
        if original == json!(replacement) {
            return Err("co-edited native identity negative did not change semantics".into());
        }
        model["fields"][field][index] = json!(replacement);
        let definition_bytes = serde_json::to_vec(&model)?;
        let definition_sha = hash(&definition_bytes);
        let definition_path = output.join(format!("coedited_{field}_definition.json"));
        write_new(&definition_path, &definition_bytes)?;
        // The definition itself must really be admitted with its new correct
        // hash: a separate definition failure cannot stand in for this gate.
        let changed_definition = RobotDefinition::load_json(&definition_path, &definition_sha)?;
        let mut changed_profile = original_profile.clone();
        changed_profile["definition_file_sha256"] = json!(definition_sha);
        changed_profile[field][index] = json!(replacement);
        if changed_profile[field] != model["fields"][field] {
            return Err("co-edited profile and admitted definition arrays differ".into());
        }
        let profile_bytes = serde_json::to_vec_pretty(&changed_profile)?;
        let profile_sha = hash(&profile_bytes);
        let profile_path = output.join(format!("coedited_{field}_profile.json"));
        write_new(&profile_path, &profile_bytes)?;
        let rejection =
            SourceCollisionProfile::load_json(&profile_path, &profile_sha, &changed_definition)
                .err()
                .ok_or_else(|| format!("co-edited definition/profile admitted: {field}"))?;
        if !rejection
            .to_string()
            .contains("semantic digest does not match audited native MJB")
        {
            return Err("co-edited data failed before the native semantic identity gate".into());
        }
        evidence.push(json!({"field":field,"index":index,"original":original,"replacement":replacement,
            "definition_path":definition_path,"definition_sha256":definition_sha,
            "profile_path":profile_path,"profile_sha256":profile_sha,
            "actual_definition_loader_accepted":true,"both_expected_file_hashes_correct":true,
            "definition_and_profile_arrays_equal":true,"native_mjb_sha256_unchanged":profile.data().native_mjb_sha256,
            "native_semantic_digest_rejected":true,"rejection":rejection.to_string()}));
        model["fields"][field][index] = original;
    }
    Ok(
        json!({"actual_loader_refusals":evidence.len(),"all_rejected":true,"cases":evidence,
        "physics_integrations":0,"policy_inferences":0}),
    )
}

fn execute(args: &HashMap<String, String>) -> CheckResult<Value> {
    let required = |name: &str| {
        args.get(name)
            .map(String::as_str)
            .ok_or_else(|| format!("missing {name}"))
    };
    let definition_path = Path::new(required("--definition")?);
    let definition = match definition_path.extension().and_then(|value| value.to_str()) {
        Some("json") => RobotDefinition::load_json(definition_path, required("--definition-sha")?)?,
        Some("ron") => RobotDefinition::load_ron(definition_path, required("--definition-sha")?)?,
        _ => return Err("definition must be actual JSON or RON".into()),
    };
    let profile = load_profile(
        Path::new(required("--profile")?),
        required("--profile-sha")?,
        &definition,
    )?;
    let ron = profile.to_ron()?;
    let ron_path = Path::new(required("--emit-ron")?);
    if ron_path.extension().and_then(|value| value.to_str()) != Some("ron") {
        return Err("output must have .ron extension".into());
    }
    write_new(ron_path, ron.as_bytes())?;
    let reloaded = SourceCollisionProfile::load_ron(ron_path, &hash(ron.as_bytes()), &definition)?;
    if reloaded.data() != profile.data() {
        return Err("actual RON round trip changed metadata".into());
    }
    let oracle = compare_oracle(
        &profile,
        Path::new(required("--oracle")?),
        required("--oracle-sha")?,
    )?;
    let refusals = negative_loaders(
        &profile,
        &definition,
        Path::new(required("--negative-output")?),
        &ron,
    )?;
    let coedited_refusals = coedited_native_identity_refusals(
        &profile,
        &definition,
        &Path::new(required("--negative-output")?).join("coedited_native_identity"),
    )?;
    Ok(
        json!({"scope":"typed_source_collision_metadata_and_pure_eligibility_only","passed":true,
        "family":profile.data().family,"profile_file_sha256":profile.file_sha256(),
        "definition_file_sha256":definition.file_sha256(),"native_mjb_sha256":profile.data().native_mjb_sha256,
        "native_semantic_sha256":profile.native_semantic_sha256(),
        "native_runtime_sha256":profile.data().native_runtime_sha256,"oracle_manifest_sha256":required("--oracle-sha")?,
        "ron":{"path":ron_path,"sha256":reloaded.file_sha256(),"actual_typed_round_trip_equal":true},
        "oracle":oracle,"refusals":refusals,"coedited_identity_refusals":coedited_refusals,
        "physics_integrations":0,"policy_inferences":0,
        "runtime_registry_or_hooks_qualified":false,"ccd_qualified":false,"force_or_bam_qualified":false,
        "full_robot_collision_qualified":false,"terrain_or_prop_masks_decided":false}),
    )
}

fn main() -> CheckResult<()> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let keys = [
        "--definition",
        "--definition-sha",
        "--profile",
        "--profile-sha",
        "--oracle",
        "--oracle-sha",
        "--emit-ron",
        "--negative-output",
        "--output",
    ];
    if raw.len() != keys.len() * 2 {
        return Err("expected all nine named path/SHA/output options".into());
    }
    let mut args = HashMap::new();
    for pair in raw.chunks_exact(2) {
        if !keys.contains(&pair[0].as_str())
            || args.insert(pair[0].clone(), pair[1].clone()).is_some()
        {
            return Err("unknown or duplicate option".into());
        }
    }
    let output = Path::new(args.get("--output").ok_or("missing output")?);
    match execute(&args) {
        Ok(report) => {
            write_new(output, &serde_json::to_vec_pretty(&report)?)?;
            println!("{}", serde_json::to_string(&report)?);
            Ok(())
        }
        Err(error) => {
            let report = json!({"scope":"typed_source_collision_check_failure","passed":false,"error":error.to_string(),
                "physics_integrations":0,"policy_inferences":0,"runtime_filter_or_physics_qualified":false});
            write_new(output, &serde_json::to_vec_pretty(&report)?)?;
            Err(error)
        }
    }
}
