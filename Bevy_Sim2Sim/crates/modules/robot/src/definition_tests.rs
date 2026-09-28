use super::{
    CompiledModel, ModelCounts, ModelFields, ModelNames, RobotDefinition, SOURCE_JNT_SOLIMP_WIDTH,
    SOURCE_JNT_SOLREF_WIDTH, validate,
};
use crate::{ACTION_DIMENSION, contract::ACTUATOR_ORDER};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashSet},
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

const NJNT: usize = 15;
const NBODY: usize = 16;
const NV: usize = 20;
const NQ: usize = 21;
const LIMIT_FIELDS: [&str; 4] = ["jnt_solref", "jnt_solimp", "jnt_margin", "dof_invweight0"];
static NEXT_TEMP_FILE: AtomicU64 = AtomicU64::new(0);

struct TempFile {
    path: PathBuf,
    sha256: String,
}

impl TempFile {
    fn write(bytes: &[u8], extension: &str) -> Self {
        let sha256 = format!("{:x}", Sha256::digest(bytes));
        let serial = NEXT_TEMP_FILE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "microduck_limit_fields_{}_{}_{sha256}.{extension}",
            std::process::id(),
            serial
        ));
        fs::write(&path, bytes).unwrap();
        Self { path, sha256 }
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn qpos_address(joint: usize) -> usize {
    if joint == 0 { 0 } else { joint + 6 }
}

fn dof_address(joint: usize) -> usize {
    if joint == 0 { 0 } else { joint + 5 }
}

fn compiled_model(
    solref: Option<Vec<[f64; SOURCE_JNT_SOLREF_WIDTH]>>,
    solimp: Option<Vec<[f64; SOURCE_JNT_SOLIMP_WIDTH]>>,
    margin: Option<Vec<f64>>,
) -> CompiledModel {
    let inverse_weight =
        (solref.is_some() && solimp.is_some() && margin.is_some()).then(|| vec![1.0; NV]);
    let mut dof_jntid = vec![0; NV];
    let mut dof_bodyid = vec![0; NV];
    for joint in 0..NJNT {
        let width = if joint == 0 { 6 } else { 1 };
        for dof in dof_address(joint)..dof_address(joint) + width {
            dof_jntid[dof] = joint;
            dof_bodyid[dof] = joint + 1;
        }
    }
    let mut joint_names = vec!["trunk_base_freejoint".to_string()];
    joint_names.extend(ACTUATOR_ORDER.iter().map(|name| (*name).to_string()));
    let mut other = BTreeMap::new();
    other.insert("jnt_stiffness".to_string(), json!(vec![0.0; NJNT]));
    let model = CompiledModel {
        schema: "microduck_compiled_v1".to_string(),
        family: "leg_allcollisions".to_string(),
        counts: ModelCounts {
            nq: NQ,
            nv: NV,
            nu: ACTION_DIMENSION,
            nbody: NBODY,
            njnt: NJNT,
            ngeom: 0,
            nmesh: 0,
            nsensor: 0,
            nkey: 1,
            neq: 0,
        },
        names: ModelNames {
            body: std::iter::once("world".to_string())
                .chain((1..NBODY).map(|body| format!("body_{body}")))
                .map(Some)
                .collect(),
            jnt: joint_names.into_iter().map(Some).collect(),
            geom: Vec::new(),
            site: Vec::new(),
            sensor: Vec::new(),
            actuator: ACTUATOR_ORDER.map(|name| Some(name.to_string())).to_vec(),
            mesh: Vec::new(),
            key: vec![Some("home".to_string())],
        },
        fields: ModelFields {
            body_parentid: (0..NBODY).map(|body| body.saturating_sub(1)).collect(),
            body_pos: vec![[0.0; 3]; NBODY],
            body_quat: vec![[1.0, 0.0, 0.0, 0.0]; NBODY],
            body_mass: (0..NBODY)
                .map(|body| if body == 0 { 0.0 } else { 1.0 })
                .collect(),
            body_inertia: (0..NBODY)
                .map(|body| if body == 0 { [0.0; 3] } else { [1.0; 3] })
                .collect(),
            body_ipos: vec![[0.0; 3]; NBODY],
            body_iquat: vec![[1.0, 0.0, 0.0, 0.0]; NBODY],
            jnt_type: (0..NJNT)
                .map(|joint| if joint == 0 { 0 } else { 3 })
                .collect(),
            jnt_bodyid: (0..NJNT).map(|joint| joint + 1).collect(),
            jnt_qposadr: (0..NJNT).map(qpos_address).collect(),
            jnt_dofadr: (0..NJNT).map(dof_address).collect(),
            jnt_pos: vec![[0.0; 3]; NJNT],
            jnt_axis: vec![[0.0, 0.0, 1.0]; NJNT],
            jnt_range: vec![[-1.0, 1.0]; NJNT],
            jnt_limited: vec![false; NJNT],
            jnt_solref: solref,
            jnt_solimp: solimp,
            jnt_margin: margin,
            dof_bodyid,
            dof_jntid,
            dof_armature: vec![0.0; NV],
            dof_damping: vec![0.0; NV],
            dof_frictionloss: vec![0.0; NV],
            dof_invweight0: inverse_weight,
            geom_type: Vec::new(),
            geom_bodyid: Vec::new(),
            geom_pos: Vec::new(),
            geom_quat: Vec::new(),
            geom_size: Vec::new(),
            geom_dataid: Vec::new(),
            geom_contype: Vec::new(),
            geom_conaffinity: Vec::new(),
            geom_condim: Vec::new(),
            geom_friction: Vec::new(),
            mesh_vertadr: Vec::new(),
            mesh_vertnum: Vec::new(),
            mesh_faceadr: Vec::new(),
            mesh_facenum: Vec::new(),
            mesh_vert: Vec::new(),
            mesh_face: Vec::new(),
            mesh_pos: Vec::new(),
            mesh_quat: Vec::new(),
            mesh_scale: Vec::new(),
            mesh_graphadr: None,
            mesh_graph: None,
            site_bodyid: Vec::new(),
            site_pos: Vec::new(),
            site_quat: Vec::new(),
            actuator_trntype: vec![0; ACTION_DIMENSION],
            actuator_trnid: (0..ACTION_DIMENSION)
                .map(|index| [(index + 1) as i32, 0])
                .collect(),
            actuator_gear: vec![[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]; ACTION_DIMENSION],
            actuator_forcerange: vec![[-1.0, 1.0]; ACTION_DIMENSION],
            actuator_forcelimited: vec![true; ACTION_DIMENSION],
            key_qpos: vec![vec![0.0; NQ]],
            qpos0: vec![0.0; NQ],
            other,
        },
        home: [0.0; ACTION_DIMENSION],
        joint_order: ACTUATOR_ORDER.map(|name| name.to_string()),
        bam: Vec::new(),
        timing_plan: json!({}),
        missing_optional_fields: Vec::new(),
        per_step_mutable_fields: Vec::new(),
        note: "synthetic compiled model for source limit field compatibility".to_string(),
        units: "SI; MuJoCo Z-up; quaternions wxyz; principal inertia in inertial frame".to_string(),
        sha256: "a".repeat(64),
    };
    validate(&model).expect("synthetic compiled model");
    model
}

fn source_limits() -> (
    Vec<[f64; SOURCE_JNT_SOLREF_WIDTH]>,
    Vec<[f64; SOURCE_JNT_SOLIMP_WIDTH]>,
    Vec<f64>,
) {
    let mut solref = vec![[0.02, 1.0]; NJNT];
    solref[0] = [-10.0, 0.5];
    let solimp = vec![[0.9, 0.95, 0.001, 0.5, 2.0]; NJNT];
    let mut margin = vec![0.0; NJNT];
    margin[1] = -0.001;
    (solref, solimp, margin)
}

fn field_keys(value: &Value) -> HashSet<String> {
    value["fields"]
        .as_object()
        .unwrap()
        .keys()
        .cloned()
        .collect()
}

fn round_trip(model: &CompiledModel) -> (CompiledModel, String) {
    let bytes = serde_json::to_vec(model).unwrap();
    let json_file = TempFile::write(&bytes, "json");
    let from_json = RobotDefinition::load_json(&json_file.path, &json_file.sha256).unwrap();
    assert_eq!(
        serde_json::to_value(from_json.model()).unwrap(),
        serde_json::to_value(model).unwrap()
    );
    let ron = from_json.to_ron().unwrap();
    let ron_file = TempFile::write(ron.as_bytes(), "ron");
    let from_ron = RobotDefinition::load_ron(&ron_file.path, &ron_file.sha256).unwrap();
    assert_eq!(
        serde_json::to_value(from_ron.model()).unwrap(),
        serde_json::to_value(model).unwrap()
    );
    (from_ron.model().clone(), ron)
}

#[test]
fn legacy_export_round_trip_omits_source_limit_parameters() {
    let model = compiled_model(None, None, None);
    let value = serde_json::to_value(&model).unwrap();
    let keys = field_keys(&value);
    for name in LIMIT_FIELDS {
        assert!(!keys.contains(name), "{name} was invented");
        assert!(!model.fields.other.contains_key(name));
    }
    assert!(keys.contains("jnt_stiffness"));
    let (loaded, ron) = round_trip(&model);
    for name in LIMIT_FIELDS {
        assert!(!ron.contains(name), "{name} appeared in RON");
        assert!(!field_keys(&serde_json::to_value(&loaded).unwrap()).contains(name));
    }
    assert!(loaded.fields.jnt_solref.is_none());
    assert!(loaded.fields.jnt_solimp.is_none());
    assert!(loaded.fields.jnt_margin.is_none());
    assert!(loaded.fields.dof_invweight0.is_none());
    assert_eq!(
        loaded.fields.other.get("jnt_stiffness").unwrap(),
        &json!(vec![0.0; NJNT])
    );

    let mut with_nulls = value;
    for name in LIMIT_FIELDS {
        with_nulls["fields"][name] = Value::Null;
    }
    let bytes = serde_json::to_vec(&with_nulls).unwrap();
    let file = TempFile::write(&bytes, "json");
    let loaded = RobotDefinition::load_json(&file.path, &file.sha256).unwrap();
    let again = serde_json::to_value(loaded.model()).unwrap();
    for name in LIMIT_FIELDS {
        assert!(again["fields"].get(name).is_none());
    }
}

#[test]
fn present_source_limit_parameters_keep_source_dimensions() {
    let (solref, solimp, margin) = source_limits();
    let model = compiled_model(
        Some(solref.clone()),
        Some(solimp.clone()),
        Some(margin.clone()),
    );
    let (loaded, _) = round_trip(&model);
    assert_eq!(loaded.fields.jnt_solref.unwrap(), solref);
    assert_eq!(loaded.fields.jnt_solimp.unwrap(), solimp);
    assert_eq!(loaded.fields.jnt_margin.unwrap(), margin);
    assert_eq!(loaded.fields.dof_invweight0.unwrap(), vec![1.0; NV]);
    for name in LIMIT_FIELDS {
        assert!(!loaded.fields.other.contains_key(name));
    }
    assert_eq!(solref[0].len(), SOURCE_JNT_SOLREF_WIDTH);
    assert_eq!(solimp[0].len(), SOURCE_JNT_SOLIMP_WIDTH);
}

#[test]
fn incomplete_source_limit_set_is_rejected() {
    let mut model = compiled_model(None, None, None);
    model.fields.jnt_margin = Some(vec![0.0; NJNT]);
    assert!(
        validate(&model)
            .unwrap_err()
            .to_string()
            .contains("incomplete")
    );
}

#[test]
fn rejects_source_limit_count_and_non_finite_values() {
    let (solref, solimp, margin) = source_limits();
    let cases = [
        (
            Some(solref[..NJNT - 1].to_vec()),
            Some(solimp.clone()),
            Some(margin.clone()),
            "jnt_solref length must equal njnt",
        ),
        (
            Some(solref.clone()),
            Some(solimp[..NJNT - 1].to_vec()),
            Some(margin.clone()),
            "jnt_solimp length must equal njnt",
        ),
        (
            Some(solref.clone()),
            Some(solimp.clone()),
            Some(margin[..NJNT - 1].to_vec()),
            "jnt_margin length must equal njnt",
        ),
    ];
    for (solref, solimp, margin, expected) in cases {
        let mut model = compiled_model(None, None, None);
        model.fields.jnt_solref = solref;
        model.fields.jnt_solimp = solimp;
        model.fields.jnt_margin = margin;
        model.fields.dof_invweight0 = Some(vec![1.0; NV]);
        let error = validate(&model).unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
    }
    let mut model = compiled_model(None, None, None);
    let (solref, solimp, margin) = source_limits();
    model.fields.jnt_solref = Some(solref.clone());
    model.fields.jnt_solimp = Some(solimp.clone());
    model.fields.jnt_margin = Some(margin.clone());
    model.fields.dof_invweight0 = Some(vec![1.0; NV - 1]);
    assert!(
        validate(&model)
            .unwrap_err()
            .to_string()
            .contains("dof_invweight0 length")
    );
    model.fields.dof_invweight0 = Some(vec![0.0; NV]);
    assert!(
        validate(&model)
            .unwrap_err()
            .to_string()
            .contains("dof_invweight0 values")
    );
    let non_finite = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
    for value in non_finite {
        let mut model = compiled_model(
            Some(solref.clone()),
            Some(solimp.clone()),
            Some(margin.clone()),
        );
        let mut bad_ref = solref.clone();
        bad_ref[2][1] = value;
        model.fields.jnt_solref = Some(bad_ref);
        let error = validate(&model).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("jnt_solref values must be finite"),
            "{error}"
        );

        let mut bad_imp = solimp.clone();
        bad_imp[2][4] = value;
        model.fields.jnt_solref = Some(solref.clone());
        model.fields.jnt_solimp = Some(bad_imp);
        let error = validate(&model).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("jnt_solimp values must be finite"),
            "{error}"
        );

        let mut bad_margin = margin.clone();
        bad_margin[2] = value;
        model.fields.jnt_solimp = Some(solimp.clone());
        model.fields.jnt_margin = Some(bad_margin);
        let error = validate(&model).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("jnt_margin values must be finite"),
            "{error}"
        );
    }
}

#[test]
fn rejects_source_limit_arrays_with_wrong_width() {
    let (solref, solimp, margin) = source_limits();
    let model = compiled_model(Some(solref), Some(solimp), Some(margin));
    let mut value = serde_json::to_value(&model).unwrap();
    let malformed = [
        (
            "jnt_solref",
            json!([[0.02]]),
            "expected an array of length 2",
        ),
        (
            "jnt_solimp",
            json!([[0.9, 0.95, 0.001, 0.5]]),
            "expected an array of length 5",
        ),
        ("jnt_margin", json!([[0.0]]), "expected f64"),
    ];
    for (name, replacement, expected) in malformed {
        let mut broken = value.clone();
        broken["fields"][name] = replacement;
        let bytes = serde_json::to_vec(&broken).unwrap();
        let file = TempFile::write(&bytes, "json");
        let error = match RobotDefinition::load_json(&file.path, &file.sha256) {
            Err(error) => error,
            Ok(_) => panic!("{name} width was accepted"),
        };
        assert!(error.to_string().contains(expected), "{name}: {error}");
    }
    value["fields"]["jnt_solref"] = json!(
        (0..NJNT * SOURCE_JNT_SOLREF_WIDTH)
            .map(|index| index as f64)
            .collect::<Vec<_>>()
    );
    let bytes = serde_json::to_vec(&value).unwrap();
    let file = TempFile::write(&bytes, "json");
    let error = match RobotDefinition::load_json(&file.path, &file.sha256) {
        Err(error) => error,
        Ok(_) => panic!("flat jnt_solref was accepted"),
    };
    assert!(
        error.to_string().contains("expected an array of length 2"),
        "{error}"
    );
}
