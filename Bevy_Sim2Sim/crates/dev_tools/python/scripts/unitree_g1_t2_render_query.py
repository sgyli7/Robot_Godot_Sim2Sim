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
    lights, robot_meshes = [], []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdLux.LightAPI):
            lights.append({"path": str(prim.GetPath()), "type": prim.GetTypeName(),
                "world_transform": encode(transforms.GetLocalToWorldTransform(prim)),
                "attributes": {a.GetName(): {"value": encode(a.Get()),
                    "authored": a.HasAuthoredValueOpinion()}
                    for a in prim.GetAttributes() if a.GetName().startswith("inputs:")}})
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
        "queried_render_settings": {p: encode(settings.get(p)) for p in paths},
        "render_configuration": encode(raw.cfg.sim.render.to_dict()),
        "scope": "read actual original stage/settings only; no implied RTX/Bevy energy or shading equivalence; null settings are unavailable"}
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return {"path": str(path), "sha256": digest(path), "physics_integrations": 0,
            "usd_writes": 0, "render_setting_writes": 0, "light_count": len(lights),
            "robot_mesh_count": len(robot_meshes)}
