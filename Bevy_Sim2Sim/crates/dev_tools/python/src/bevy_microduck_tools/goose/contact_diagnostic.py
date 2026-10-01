"""Unqualified single-patch Kelvin contact mapped into MuJoCo3.10 native rows.

This bounded diagnostic changes no frozen Goose candidate. Step1 prepares the
native contact constraints, the normal rows receive backward-Euler Kelvin
regularization/reference coefficients, and step2 integrates implicit exactly
once. Contact force remains in the native solver; no post-reset qpos writes.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from .artifacts import DT, sha256, write_json

K = 23357.0304
C = 2.0
TRAVEL = .0015
FAIL_TRAVEL = .00155


def fixture(shape, mass):
    root = ET.Element("mujoco")
    ET.SubElement(root, "option", timestep=str(DT), gravity="0 0 0", integrator="implicit", cone="elliptic", iterations="100", tolerance="1e-12")
    world = ET.SubElement(root,"worldbody")
    ET.SubElement(world,"geom",name="floor",type="plane",size="1 1 .1",friction="0 0 0",condim="1")
    # One micrometre initial penetration activates native unilateral contact.
    body = ET.SubElement(world,"body",name="patch",pos="0 0 .000749")
    ET.SubElement(body,"joint",name="z",type="slide",axis="0 0 1")
    ET.SubElement(body,"geom",name="patch",type=shape,size=".00075" if shape=="sphere" else ".0025 .012 .00075",mass=str(mass),friction="0 0 0",condim="1")
    return mujoco.MjModel.from_xml_string(ET.tostring(root,encoding="unicode"))


def kelvin_native_step(model,data,force):
    before = float(data.time)
    mujoco.mj_step1(model,data)
    # A single patch can generate four box/plane points. Distribute the
    # physical patch K and C across active normal contact rows each tick.
    normal_rows = [int(c.efc_address) for c in data.contact if c.efc_address >= 0]
    if any(int(c.dim) != 1 for c in data.contact if c.efc_address >= 0):
        raise RuntimeError("Diagnostic supports only frictionless normal rows")
    n = len(normal_rows)
    if n:
        kp, cp = K/n, C/n
        denominator = DT*(cp+DT*kp)
        for row in normal_rows:
            data.efc_R[row] = 1./denominator
            data.efc_D[row] = denominator
            # mj_referenceConstraint: -b*v-k*impedance*(pos-margin).
            data.efc_KBIP[row] = [kp/denominator,1./DT,1.,0.]
        mujoco.mj_referenceConstraint(model,data)
        # Rebuild projected constraint inertia from the new R when this
        # solver path uses it, then gather modified fields for native islands.
        mujoco.mj_projectConstraint(model,data)
        if data.nisland:
            for row in normal_rows:
                island_row = int(data.map_efc2iefc[row])
                if island_row < 0:
                    raise RuntimeError("Active row missing native island mapping")
                data.iefc_R[island_row] = data.efc_R[row]
                data.iefc_D[island_row] = data.efc_D[row]
                data.iefc_aref[island_row] = data.efc_aref[row]
    # +z joint and downward fixture force; never add a contact force manually.
    data.qfrc_applied[0] = -force
    mujoco.mj_step2(model,data)
    if abs(data.time-before-DT)>1e-12:
        raise RuntimeError("Native split step changed single-tick timing")
    return n,float(sum(data.efc_force[row] for row in normal_rows))


def one_case(shape,mass,force,schedule,impact_velocity=0.):
    model = fixture(shape,mass)
    data = mujoco.MjData(model)
    data.qvel[0] = -impact_velocity  # Initial condition, before first integration.
    rows = []
    reason = None
    previous_energy = .5*mass*impact_velocity**2+.5*K*(1e-6)**2
    max_energy_increment_unforced = 0.
    for tick in range(300):
        load = force
        if schedule=="ramp": load *= min((tick+1)*DT,1.)
        if schedule=="release": load = force if tick<150 else 0.
        if schedule=="impact": load=0.
        try:
            points,native_normal_force = kelvin_native_step(model,data,load)
        except (RuntimeError,mujoco.FatalError) as exc:
            reason=str(exc)
            break
        compression = 1e-6-float(data.qpos[0])
        velocity = float(data.qvel[0])
        spring_energy=.5*K*max(compression,0.)**2
        energy=.5*mass*velocity**2+spring_energy
        if not load:
            max_energy_increment_unforced=max(max_energy_increment_unforced,energy-previous_energy)
        previous_energy=energy
        warnings={mujoco.mjtWarning(i).name:int(w.number) for i,w in enumerate(data.warning) if w.number}
        rows.append({"tick":tick,"time_s":float(data.time),"force_n":load,"contact_points":points,
                     "compression_m":compression,"velocity_m_s":velocity,"native_normal_force_n":native_normal_force,
                     "energy_j":energy,"warnings":warnings})
        if warnings or not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            reason="nonfinite_or_solver_warning"
            break
        if compression>FAIL_TRAVEL:
            reason="travel_exceeded_1_55mm"
            break
    static_rows=rows[-50:] if schedule in ("step","ramp") else []
    compression_error=max((abs(r["compression_m"]-force/K) for r in static_rows),default=0.)
    force_error=max((abs(r["native_normal_force_n"]-force) for r in static_rows),default=0.)
    energy_ok=max_energy_increment_unforced<=1e-12
    if reason is None and not energy_ok: reason="unforced_energy_increased"
    if reason is None and (compression_error>.00005 or force_error>force*.1): reason="static_curve_error"
    return {"shape":shape,"mass_kg":mass,"requested_force_n":force,"schedule":schedule,"impact_velocity_m_s":impact_velocity,
            "status":"passed" if reason is None and len(rows)==300 else "failed","failure_reason":reason,
            "ticks_completed":len(rows),"max_compression_m":max((r["compression_m"] for r in rows),default=0.),
            "static_compression_error_m":compression_error,"static_force_error_n":force_error,
            "max_energy_increment_unforced_j":max_energy_increment_unforced,"rows":rows}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    output=args.output.resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    os.chdir(output.parent)
    if mujoco.__version__ != "3.10.0": raise RuntimeError("Native row/island ABI diagnostic pinned to MuJoCo3.10.0")
    started=time.perf_counter()
    cases=[]
    for shape in ("sphere","box"):
        for mass in (.0002232,.29298065203309465,1.):
            for force in (5.,10.,20.,30.):
                for schedule in ("step","ramp","release"):
                    cases.append(one_case(shape,mass,force,schedule))
            for velocity in (.02,.05,.1,.5):
                cases.append(one_case(shape,mass,0.,"impact",velocity))
    receipt={"schema":"goose_kelvin_contact_native_diagnostic_v1","engine":"mujoco_cpu","engine_version":mujoco.__version__,
             "code_sha256":sha256(Path(__file__)),"physics_dt_s":DT,"integrations_per_tick":1,"substeps":0,
             "normal_law":"perpointK=K/n,C=C/n;R=1/[dt(C+dtK)];aref=-v/dt-Kr/[dt(C+dtK)]",
             "stiffness_n_m":K,"damping_n_s_m":C,"travel_m":TRAVEL,"failure_travel_m":FAIL_TRAVEL,
             "initial_condition":"patch starts1umcompressed; only pre-integration initialqvel forimpact",
             "contact_force_source":"nativeefc_force,notexternalcontactpenalty","qpos_writes_after_reset":0,
             "integrator":"nativeimplicitvia mj_step1+mj_step2","friction":0,"cases":cases,
             "passed_cases":sum(c["status"]=="passed" for c in cases),"failed_cases":sum(c["status"]=="failed" for c in cases),
             "elapsed_wall_s":time.perf_counter()-started,"qualified":False,"optimizer_updates":0,
             "limits":["nativeMuJoCo3.10internalconstraintarrayABI","noGoosesceneorjawqualification","noGPUWarpimplementation",
                       "friction/torsion/tangentrowsnotimplemented","noactual1.5mmtravelendhardening","impactwithflightgapnotcovered"]}
    write_json(output,receipt)
    print(f"STATUS DIAGNOSTIC_ONLY passed={receipt['passed_cases']} failed={receipt['failed_cases']} {output}")


if __name__=="__main__":main()
