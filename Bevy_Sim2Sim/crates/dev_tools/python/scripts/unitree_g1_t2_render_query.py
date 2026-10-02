"""Read actual released scene photometric inputs; never alter render settings."""

import json
import math
from pathlib import Path

from unitree_g1_task_asset_query import digest


def encode(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"nonfinite_schema_value": str(value)}
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    if hasattr(value, "resolvedPath"):
        return {"asset_path": value.path, "resolved_path": value.resolvedPath}
    try:
        return [encode(x) for x in value]
    except TypeError:
        return {"usd_value": str(value)}


def query_render_inputs(raw, path):
    import carb
    import omni.usd
    from pxr import Usd, UsdGeom, UsdLux, UsdShade

    stage = omni.usd.get_context().get_stage()
    transforms = UsdGeom.XformCache()
    lights, robot_meshes, cameras, render_scopes = [], [], [], []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Camera):
            cameras.append({"path": str(prim.GetPath()), "applied_schemas": list(prim.GetAppliedSchemas()),
                "attributes": {a.GetName(): {"value": encode(a.Get()),
                    "authored": a.HasAuthoredValueOpinion()}
                    for a in prim.GetAttributes()},
                "relationships": {r.GetName(): encode(r.GetTargets()) for r in prim.GetRelationships()}})
        if prim.GetTypeName() in ('RenderSettings', 'RenderProduct'):
            render_scopes.append({"path": str(prim.GetPath()), "type": prim.GetTypeName(),
                "attributes": {a.GetName(): {"value": encode(a.Get()),
                    "authored": a.HasAuthoredValueOpinion()}
                    for a in prim.GetAttributes()},
                "relationships": {r.GetName(): encode(r.GetTargets()) for r in prim.GetRelationships()}})
        if prim.HasAPI(UsdLux.LightAPI):
            lights.append({"path": str(prim.GetPath()), "type": prim.GetTypeName(),
                "world_transform": encode(transforms.GetLocalToWorldTransform(prim)),
                "attributes": {a.GetName(): {"value": encode(a.Get()),
                    "authored": a.HasAuthoredValueOpinion()}
                    for a in prim.GetAttributes()}})
        if str(prim.GetPath()).startswith('/World/envs/env_0/Robot/') and prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
            shaders = [] if not material else [UsdShade.Shader(p) for p in Usd.PrimRange(material.GetPrim())
                                               if p.IsA(UsdShade.Shader)]
            robot_meshes.append({"path": str(prim.GetPath()), "orientation": mesh.GetOrientationAttr().Get(),
                "visible": UsdGeom.Imageable(prim).ComputeVisibility(), "purpose": UsdGeom.Imageable(prim).ComputePurpose(),
                "normal_interpolation": mesh.GetNormalsInterpolation(),
                "normal_count": len(mesh.GetNormalsAttr().Get() or []),
                "point_count": len(mesh.GetPointsAttr().Get() or []),
                "face_corner_count": len(mesh.GetFaceVertexIndicesAttr().Get() or []),
                "material": str(material.GetPath()) if material else None,
                "shaders": [{"path": str(s.GetPath()),
                    "attributes": {a.GetName(): encode(a.Get()) for a in s.GetPrim().GetAttributes()
                                   if a.GetName().startswith(('inputs:', 'info:'))}}
                    for s in shaders]})
    settings = carb.settings.get_settings()
    paths = ('/rtx/post/tonemap/op', '/rtx/post/tonemap/cameraIso',
             '/rtx/post/tonemap/whiteScale', '/rtx/post/tonemap/cm2Factor',
             '/rtx/post/tonemap/filmIso', '/rtx/post/tonemap/fNumber',
             '/rtx/post/tonemap/exposureTime', '/rtx/post/tonemap/responsivity',
             '/rtx/post/tonemap/whitepoint', '/rtx/post/tonemap/wrapValue',
             '/rtx/post/tonemap/dither', '/rtx/post/tonemap/ocio/enabled',
             '/rtx/post/tonemap/shutter', '/rtx/post/tonemap/colorMode',
             '/rtx/post/tonemap/enableSrgbToGamma', '/rtx/post/histogram/enabled',
             '/rtx/post/histogram/whiteScale', '/rtx/post/aa/op',
             '/rtx/post/dlss/execMode', '/rtx/raytracing/fractionalCutoutOpacity',
             '/rtx/ambientLightIntensity', '/rtx/ambientLightColor',
             '/rtx/sceneDb/ambientLightIntensity', '/rtx/sceneDb/ambientLightColor')
    result = {"schema": "g1_released_t2_actual_render_inputs_v1", "qualified": False,
        "query_sha256": digest(Path(__file__)), "runtime_build": Path('/isaac-sim/VERSION').read_text().strip(),
        "physics_integrations": 0, "usd_writes": 0, "render_setting_writes": 0,
        "lights": lights, "actual_robot_meshes": robot_meshes,
        "actual_cameras": cameras, "render_scopes": render_scopes,
        "queried_render_settings": {p: encode(settings.get(p)) for p in paths},
        "render_configuration": encode(raw.cfg.sim.render.to_dict()),
        "scope": "read actual original stage/settings only; no implied RTX/Bevy energy or shading equivalence; null settings are unavailable"}
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return {"path": str(path), "sha256": digest(path), "physics_integrations": 0,
            "usd_writes": 0, "render_setting_writes": 0, "light_count": len(lights),
            "robot_mesh_count": len(robot_meshes)}


def probe_legacy_light_inputs(raw, camera, directory):
    """Four fixed zero-step renders distinguish legacy and modern USD light inputs.

    Only the eight source RectLight intensities are overridden in the temporary
    session layer. No pose, material, physics input or model is modified.
    """
    import numpy as np
    import carb
    import omni.kit.app
    import omni.usd
    from PIL import Image
    from pxr import Usd, UsdLux

    stage = omni.usd.get_context().get_stage()
    lights = [p for p in stage.Traverse() if p.HasAPI(UsdLux.LightAPI)]
    if len(lights) != 8 or any(p.GetTypeName() != 'RectLight'
            or p.GetAttribute('intensity').Get() != 200000.0
            or p.GetAttribute('inputs:intensity').Get() != 1.0 for p in lights):
        raise ValueError('Legacy light probe requires the unchanged eight original RectLights')
    session = stage.GetSessionLayer()
    original = {(str(p.GetPath()), name): session.GetPropertyAtPath(p.GetPath().AppendProperty(name))
                for p in lights for name in ('intensity', 'inputs:intensity')}
    if any(original.values()):
        raise ValueError('Original light probe session layer already overrides intensity')
    settings = carb.settings.get_settings()
    previous = settings.get('/app/player/playSimulations')
    images = {}
    writes = 0
    try:
        raw.sim.set_setting('/app/player/playSimulations', False)
        with Usd.EditContext(stage, session):
            for condition in ('baseline', 'modern_zero', 'legacy_zero', 'restored'):
                if condition == 'modern_zero':
                    for p in lights: p.GetAttribute('inputs:intensity').Set(0.0); writes += 1
                elif condition == 'legacy_zero':
                    for p in lights:
                        p.GetAttribute('inputs:intensity').Clear(); writes += 1
                        p.GetAttribute('intensity').Set(0.0); writes += 1
                elif condition == 'restored':
                    for p in lights: p.GetAttribute('intensity').Clear(); writes += 1
                for _ in range(6):
                    omni.kit.app.get_app().update()
                camera.reset()
                rgb = camera.data.output['rgb'][0].detach().cpu().numpy()[..., :3].copy()
                if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
                    raise ValueError('Source light probe returned a foreign camera buffer')
                images[condition] = rgb
                Image.fromarray(rgb).save(directory / f'light_{condition}.png')
    finally:
        with Usd.EditContext(stage, session):
            for p in lights:
                for name in ('intensity', 'inputs:intensity'):
                    p.GetAttribute(name).Clear()
        raw.sim.set_setting('/app/player/playSimulations', previous)
    if any(p.GetAttribute('intensity').Get() != 200000.0
            or p.GetAttribute('inputs:intensity').Get() != 1.0 for p in lights):
        raise ValueError('Source light probe failed to restore original values')
    baseline = images['baseline'].astype(np.float64)
    result = {'schema': 'g1_source_legacy_light_causal_probe_v1', 'qualified': False,
        'physics_integrations': 0, 'model_calls': 0, 'render_only_pumps': 24,
        'temporary_usd_light_intensity_writes': writes,
        'light_attributes_restored': True, 'conditions': {},
        'scope': 'source render-only positive control; no native task or cross-renderer photometric equivalence'}
    for condition, rgb in images.items():
        result['conditions'][condition] = {'rgb_sha256': digest(directory / f'light_{condition}.png'),
            'mean_rgb': rgb.mean(axis=(0, 1)).tolist(),
            'mean_absolute_rgb_difference_from_baseline': float(np.abs(rgb.astype(np.float64)-baseline).mean())}
    path = directory / 'light_causal_probe.json'
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return {'path': str(path), 'sha256': digest(path), **result}
