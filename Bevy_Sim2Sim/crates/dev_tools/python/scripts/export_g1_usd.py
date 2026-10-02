#!/usr/bin/env python3
"""Export the frozen Arena G1 USD using CPU OpenUSD, without loading Isaac Sim.

Geometry stays in source Z-up metres. The Rust boundary performs the sole basis
change. Mass defaults on four shape-less fixed sensors are surfaced explicitly;
this export is a diagnostic plant, not proof of PhysX/Rapier equivalence.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

USD_SHA256 = "a7a2bab76981d19a1d76adecdfffec9b52afa34df9ba8e288ccedf410d3ce6bd"
ARENA_COMMIT = "7d75c95934c51a0318c957a8831e862ca43c53b5"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def vector(v):
    return [float(x) for x in v]


def quaternion(q):
    return [float(q.GetReal()), *vector(q.GetImaginary())]


def pose(matrix):
    t = Gf.Transform(matrix)
    scale = vector(t.GetScale())
    if any(abs(x - 1.0) > 1e-5 for x in scale):
        raise ValueError(f"non-rigid body/joint pose scale {scale}")
    return {"position": vector(t.GetTranslation()), "rotation_wxyz": quaternion(t.GetRotation().GetQuat())}


def bound_visual_material(prim):
    """Preserve the actual G1 bound shader color, not displayColor fallback."""
    material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
    if not material:
        raise ValueError(f"source G1 visual has no bound material: {prim.GetPath()}")
    shaders = [UsdShade.Shader(p) for p in Usd.PrimRange(material.GetPrim())
               if p.IsA(UsdShade.Shader)]
    if len(shaders) != 1 or shaders[0].GetIdAttr().Get() != "UsdPreviewSurface":
        raise ValueError(f"unsupported source G1 visual shader: {material.GetPath()}")
    shader = shaders[0]
    source_input = "diffuse_color_constant"
    color = shader.GetInput(source_input).Get()
    if color is None or any(not math.isfinite(v) or not 0 <= v <= 1 for v in color):
        raise ValueError(f"invalid source G1 diffuse color: {material.GetPath()}")
    return {"path": str(material.GetPath()), "shader_id": "UsdPreviewSurface",
            "source_input": source_input, "diffuse_color_linear": vector(color)}


def inertia_oracle(stage, destination):
    """Independent f64 tensor calculation from authored USD, not Rust's basis helper."""
    def matmul(a, b):
        return [[sum(a[r][k] * b[k][c] for k in range(3)) for c in range(3)] for r in range(3)]
    def transpose(a):
        return [list(row) for row in zip(*a)]
    change = [[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]]
    rows = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        mass = UsdPhysics.MassAPI(prim)
        diagonal = vector(mass.GetDiagonalInertiaAttr().Get())
        source_axes = quaternion(mass.GetPrincipalAxesAttr().Get())
        axes_sentinel = source_axes == [0.] * 4
        if axes_sentinel:
            if max(diagonal) != min(diagonal):
                raise ValueError("undefined principal axes with anisotropic inertia")
            q = [1., 0., 0., 0.]
        else:
            norm = math.sqrt(sum(v * v for v in source_axes))
            q = [v / norm for v in source_axes]
        w, x, y, z = q
        rotation = [[1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)],
                    [2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)],
                    [2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)]]
        diagonal_matrix = [[diagonal[r] if r == c else 0. for c in range(3)] for r in range(3)]
        tensor_source = matmul(matmul(rotation, diagonal_matrix), transpose(rotation))
        tensor_engine = matmul(matmul(change, tensor_source), transpose(change))
        rows.append({"body": len(rows), "name": prim.GetName(), "mass": float(mass.GetMassAttr().Get()),
                     "principal_axes_sentinel": axes_sentinel,
                     "source_tensor_row_major": tensor_source, "engine_tensor_row_major": tensor_engine})
    def quat_matrix(value):
        w, x, y, z = quaternion(value)
        norm = math.sqrt(w*w + x*x + y*y + z*z)
        w, x, y, z = (v / norm for v in (w, x, y, z))
        return [[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]]
    def matvec(matrix, value):
        return [sum(matrix[r][c] * value[c] for c in range(3)) for r in range(3)]
    def add(a, b):
        return [x+y for x,y in zip(a,b)]
    identity = [[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]
    rigid = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    ids = {str(p.GetPath()): i for i,p in enumerate(rigid)}
    poses = {0: (identity, [0.,0.,3.])}
    connections, positions, axes = [], {}, []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        p = ids[str(joint.GetBody0Rel().GetTargets()[0])]
        c = ids[str(joint.GetBody1Rel().GetTargets()[0])]
        r0, r1 = quat_matrix(joint.GetLocalRot0Attr().Get()), quat_matrix(joint.GetLocalRot1Attr().Get())
        p0, p1 = vector(joint.GetLocalPos0Attr().Get()), vector(joint.GetLocalPos1Attr().Get())
        if prim.IsA(UsdPhysics.RevoluteJoint):
            hinge = UsdPhysics.RevoluteJoint(prim)
            theta = math.radians(0.63 * hinge.GetLowerLimitAttr().Get() + 0.37 * hinge.GetUpperLimitAttr().Get())
            positions[prim.GetName()] = theta
            axis = "XYZ".index(str(hinge.GetAxisAttr().Get()))
            basis = [1. if k == axis else 0. for k in range(3)]
            x,y,z = basis
            cross = [[0.,-z,y],[z,0.,-x],[-y,x,0.]]
            ct, st = math.cos(theta), math.sin(theta)
            rotation = [[ct * identity[r][col] + (1-ct) * basis[r] * basis[col] + st * cross[r][col] for col in range(3)] for r in range(3)]
        else:
            rotation, basis = identity, None
        local_r = matmul(matmul(r0, rotation), transpose(r1))
        local_p = add(p0, [-v for v in matvec(local_r,p1)])
        connections.append((prim.GetName(),p,c,local_r,local_p,r0,basis))
    pending = list(connections)
    while pending:
        ready = [j for j in pending if j[1] in poses]
        if not ready:
            raise ValueError("cyclic or disconnected USD source joints")
        for name,p,c,lr,lp,r0,basis in ready:
            parent_r,parent_p = poses[p]
            poses[c] = (matmul(parent_r,lr),add(parent_p,matvec(parent_r,lp)))
            if basis is not None:
                axes.append({"name":name,"engine_world_axis":matvec(change,matvec(matmul(parent_r,r0),basis))})
        pending = [j for j in pending if j not in ready]
    native_case = {"root_source_position":[0.,0.,3.],"joint_positions_by_name":positions,
                   "bodies":[{"body":i,"position_engine":matvec(change,poses[i][1]),
                              "rotation_engine_row_major":matmul(matmul(change,poses[i][0]),transpose(change))} for i in range(len(rigid))],
                   "joint_axes":axes}
    result = {"schema": "g1_inertia_oracle_v1", "usd_sha256": USD_SHA256, "exporter_sha256": sha(__file__),
              "formula": "I_engine = C * (R_source * diag(I_source) * transpose(R_source)) * transpose(C)",
              "source_axes_normalization": "normalize copied USD quaternion in f64; zero sentinel only accepted for isotropic sensor tensor",
              "C_row_major": change, "bodies": rows, "independent_joint_frame_case": native_case}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")
    print(json.dumps({"oracle": str(destination), "sha256": sha(destination), "bodies": len(rows)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--inertia-oracle-output", type=Path, help="independent zero-integration tensor oracle; do not regenerate geometry cache")
    parser.add_argument("--visual-output", type=Path)
    args = parser.parse_args()
    if sha(args.usd) != USD_SHA256:
        raise ValueError("USD bytes differ from the frozen Arena asset")
    stage = Usd.Stage.Open(str(args.usd))
    if UsdGeom.GetStageMetersPerUnit(stage) != 1.0 or UsdGeom.GetStageUpAxis(stage) != "Z":
        raise ValueError("unexpected USD units or up axis")
    if args.inertia_oracle_output:
        inertia_oracle(stage, args.inertia_oracle_output)
        return
    if args.output is None:
        parser.error("--output is required when not exporting the inertia oracle")
    cache = UsdGeom.XformCache()
    bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    ids = {str(p.GetPath()): i for i, p in enumerate(bodies)}
    if len(ids) != 53:
        raise ValueError("G1 body count changed")
    warnings, body_data, joints, collisions, visuals = [], [], [], [], []
    for prim in bodies:
        mass = UsdPhysics.MassAPI(prim)
        com = vector(mass.GetCenterOfMassAttr().Get())
        axes = quaternion(mass.GetPrincipalAxesAttr().Get())
        defaults = []
        if not all(math.isfinite(x) for x in com):
            if com != [-math.inf] * 3:
                raise ValueError("unknown center-of-mass sentinel")
            com, defaults = [0., 0., 0.], ["centerOfMass=(-inf,-inf,-inf): diagnostic origin fallback for shape-less fixed sensor"]
        if axes == [0.] * 4:
            axes = [1., 0., 0., 0.]
            defaults.append("principalAxes=(0,0,0,0): diagnostic identity fallback for shape-less fixed sensor")
        if defaults:
            warnings.append({"body": prim.GetName(), "interpretation": defaults})
        body_data.append({"name": prim.GetName(), "path": str(prim.GetPath()),
                          "rest_pose": pose(cache.GetLocalToWorldTransform(prim)),
                          "mass": float(mass.GetMassAttr().Get()), "center_of_mass": com,
                          "principal_inertia": vector(mass.GetDiagonalInertiaAttr().Get()),
                          "principal_axes_wxyz": axes, "diagnostic_mass_defaults": bool(defaults)})
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            parents, children = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
            if len(parents) != 1 or len(children) != 1:
                raise ValueError("a G1 joint does not connect two physical bodies")
            data = {"name": prim.GetName(), "parent": ids[str(parents[0])], "child": ids[str(children[0])],
                    "frame_parent": {"position": vector(joint.GetLocalPos0Attr().Get()), "rotation_wxyz": quaternion(joint.GetLocalRot0Attr().Get())},
                    "frame_child": {"position": vector(joint.GetLocalPos1Attr().Get()), "rotation_wxyz": quaternion(joint.GetLocalRot1Attr().Get())}}
            if prim.IsA(UsdPhysics.RevoluteJoint):
                hinge = UsdPhysics.RevoluteJoint(prim)
                data.update(kind="revolute", axis=str(hinge.GetAxisAttr().Get()),
                            limits=[math.radians(hinge.GetLowerLimitAttr().Get()), math.radians(hinge.GetUpperLimitAttr().Get())])
            elif prim.IsA(UsdPhysics.FixedJoint):
                data.update(kind="fixed", axis="X", limits=[0., 0.])
            else:
                raise ValueError(f"unsupported G1 joint: {prim.GetPath()}")
            joints.append(data)
        if not prim.HasAPI(UsdPhysics.CollisionAPI) and not (args.visual_output and prim.IsA(UsdGeom.Mesh)):
            continue
        owner = prim
        while owner and str(owner.GetPath()) not in ids:
            owner = owner.GetParent()
        if not owner:
            raise ValueError(f"shape outside any body: {prim.GetPath()}")
        body = ids[str(owner.GetPath())]
        relative = cache.GetLocalToWorldTransform(prim) * cache.GetLocalToWorldTransform(owner).GetInverse()
        data = {"body": body, "path": str(prim.GetPath())}
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            # Bake complete affine local transform into points once, including scale.
            points = [vector(relative.Transform(Gf.Vec3d(p))) for p in mesh.GetPointsAttr().Get()]
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                if UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() != "convexHull":
                    raise ValueError("unsupported mesh collision cooking mode")
                data.update(shape="convex_hull", points=[list(p) for p in dict.fromkeys(map(tuple, points))])
            else:
                indices = list(mesh.GetFaceVertexIndicesAttr().Get())
                triangles, offset = [], 0
                for count in mesh.GetFaceVertexCountsAttr().Get():
                    if count != 3:
                        raise ValueError("source visual is not triangulated; avoid changing topology")
                    triangles.append(indices[offset:offset + count])
                    offset += count
                colors = mesh.GetDisplayColorAttr().Get()
                visuals.append({**data, "points": points, "triangles": triangles,
                                "display_color": vector(colors[0]) if colors else [0.65, 0.65, 0.68],
                                "bound_material": bound_visual_material(prim)})
                continue
        else:
            transform = Gf.Transform(relative)
            scale = vector(transform.GetScale())
            local = {"position": vector(transform.GetTranslation()), "rotation_wxyz": quaternion(transform.GetRotation().GetQuat())}
            if prim.IsA(UsdGeom.Sphere):
                if max(scale) - min(scale) > 1e-5:
                    raise ValueError("nonuniform sphere scale")
                data.update(shape="sphere", radius=float(UsdGeom.Sphere(prim).GetRadiusAttr().Get()) * scale[0], local_pose=local)
            elif prim.IsA(UsdGeom.Capsule):
                capsule = UsdGeom.Capsule(prim)
                if max(scale) - min(scale) > 1e-5:
                    raise ValueError("nonuniform capsule scale")
                data.update(shape="capsule", radius=float(capsule.GetRadiusAttr().Get()) * scale[0],
                            half_height=float(capsule.GetHeightAttr().Get()) * scale[0] / 2., axis=str(capsule.GetAxisAttr().Get()), local_pose=local)
            elif prim.IsA(UsdGeom.Cube):
                size = float(UsdGeom.Cube(prim).GetSizeAttr().Get())
                data.update(shape="box", half_extents=[s * size / 2. for s in scale], local_pose=local)
            else:
                raise ValueError(f"unsupported collider {prim.GetTypeName()}")
        collisions.append(data)
    if len(joints) != 52 or len(collisions) != 52:
        raise ValueError("G1 joint/collider count changed")
    for w in warnings:
        body = next(i for i, b in enumerate(body_data) if b["name"] == w["body"])
        if any(c["body"] == body for c in collisions) or not any(j["child"] == body and j["kind"] == "fixed" for j in joints):
            raise ValueError("mass sentinel occurs outside a shape-less fixed sensor")
    data = {"schema": "arena_g1_usd_v1", "usd_sha256": USD_SHA256, "arena_commit": ARENA_COMMIT,
            "exporter_sha256": sha(__file__), "openusd_version": list(Usd.GetVersion()),
            "units": "metres_kilograms_radians_z_up", "bodies": body_data, "joints": joints,
            "collisions": collisions, "diagnostic_notes": warnings}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n")
    receipt = {"output": str(args.output), "sha256": sha(args.output), "body_count": len(bodies),
               "joint_kinds": dict(Counter(j["kind"] for j in joints)),
               "collision_shapes": dict(Counter(c["shape"] for c in collisions)), "mass_default_bodies": [w["body"] for w in warnings]}
    if args.visual_output:
        args.visual_output.parent.mkdir(parents=True, exist_ok=True)
        args.visual_output.write_text(json.dumps({"usd_sha256": USD_SHA256, "body_names": [b["name"] for b in body_data], "visuals": visuals}, separators=(",", ":"), allow_nan=False) + "\n")
        receipt.update(visual_output=str(args.visual_output), visual_sha256=sha(args.visual_output), visual_count=len(visuals))
    args.output.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
