"""Isolate the delivered jaw mass/hinges/equalities, without changing frozen v1."""
from __future__ import annotations
import argparse, copy, json, os
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from .artifacts import DT, sha256, write_json


def _values(values):return ' '.join(format(float(v),'.17g') for v in values)


def make_fixture(model_path,contract,output):
    original=ET.parse(model_path).getroot()
    head=next(b for b in original.iter('body') if b.get('name')=='head_roll')
    root=ET.Element('mujoco',model='goose_isolated_original_jaw50')
    ET.SubElement(root,'compiler',angle='radian',inertiafromgeom='false')
    ET.SubElement(root,'option',timestep=str(DT),gravity='0 0 0',integrator='implicit',iterations=original.find('option').get('iterations','100'),cone='elliptic')
    world=ET.SubElement(root,'worldbody')
    head_origin=next(j['pivot_world_at_zero_m'] for j in contract['joints'] if j['name']=='head_roll')
    fixed=ET.SubElement(world,'body',name='head_roll',pos=_values(head_origin))
    fixed.append(copy.deepcopy(head.find('inertial')))
    for name in ('beak_hinge','beak_input_rotor'):
        source=next(b for b in head.findall('body') if b.get('name')==name)
        body=copy.deepcopy(source)
        for element in body.iter('body'):
            for child in list(element):
                if child.tag not in ('inertial','joint','body'):element.remove(child)
        fixed.append(body)
    transmission=contract['beak_transmission']
    radius,phase=transmission['crank_radius_m'],transmission['closed_crank_angle_in_xz_rad']
    delta=radius*np.array([np.cos(phase),0.,np.sin(phase)])
    jaw=next(b for b in fixed.iter('body') if b.get('name')=='beak_hinge')
    coupler=next(b for b in fixed.iter('body') if b.get('name')=='beak_coupler_link')
    ET.SubElement(jaw,'site',name='jaw_output_pin',pos=_values(delta))
    ET.SubElement(coupler,'site',name='coupler_output_pin',pos=_values(np.array(transmission['jaw_axis_world_m'])-transmission['drive_axis_world_m']))
    ET.SubElement(jaw,'site',name='jaw_grip_load_point',pos=_values(np.array(contract['sites']['grip']['world_at_zero_m'])-transmission['jaw_axis_world_m']))
    actuators=ET.SubElement(root,'actuator')
    actuators.append(copy.deepcopy(next(a for a in original.find('actuator') if a.get('joint')=='beak_input_rotor')))
    root.append(copy.deepcopy(original.find('equality')))
    tree=ET.ElementTree(root);ET.indent(tree,space='  ');tree.write(output,encoding='unicode')
    return mujoco.MjModel.from_xml_path(str(output))


def set_initial_pose(model,data,q):
    mujoco.mj_resetData(model,data)
    for name,value in (('beak_hinge',q),('beak_input_rotor',q),('beak_coupler_link',-q)):
        data.qpos[int(model.joint(name).qposadr[0])]=value
    mujoco.mj_forward(model,data)


def state(model,data):
    q={name:float(data.qpos[int(model.joint(name).qposadr[0])]) for name in ('beak_hinge','beak_input_rotor','beak_coupler_link')}
    pin=data.site_xpos[model.site('jaw_output_pin').id]
    coupler=data.site_xpos[model.site('coupler_output_pin').id]
    forces=[]
    for row in range(data.nefc):
        kind=int(data.efc_type[row]);eid=int(data.efc_id[row])
        forces.append({'type':mujoco.mjtConstraint(kind).name,'id':eid,'force':float(data.efc_force[row]),
                       'equality_name':mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_EQUALITY,eid) if kind==int(mujoco.mjtConstraint.mjCNSTR_EQUALITY) else None})
    return {'time_s':float(data.time),'angles_rad':q,'rotor_jaw_error_rad':q['beak_input_rotor']-q['beak_hinge'],
            'coupler_jaw_error_rad':q['beak_coupler_link']+q['beak_hinge'],
            'output_pin_distance_m':float(np.linalg.norm(pin-coupler)),
            'jaw_output_pin_world_m':pin.tolist(),'coupler_output_pin_world_m':coupler.tolist(),
            'native_constraint_rows':forces,'qfrc_constraint_nm':data.qfrc_constraint.tolist(),
            'warnings':{mujoco.mjtWarning(i).name:int(w.number) for i,w in enumerate(data.warning) if w.number}}


def dynamic_case(model,initial,load_n=0.,load_site='jaw_grip_load_point'):
    data=mujoco.MjData(model);set_initial_pose(model,data,initial)
    rows=[]
    for tick in range(25):
        data.ctrl[0]=.24
        data.qfrc_applied[:]=0.
        force=np.zeros(3)
        if load_n:
            point=data.site_xpos[model.site(load_site).id].copy()
            jid=model.joint('beak_hinge').id
            lever=point-data.xanchor[jid]
            tangent=np.cross(data.xaxis[jid],lever)
            force=-load_n*tangent/np.linalg.norm(tangent)
            mujoco.mj_applyFT(model,data,force,np.zeros(3),point,model.body('beak_hinge').id,data.qfrc_applied)
        external=data.qfrc_applied.copy()
        before=float(data.time)
        mujoco.mj_step(model,data)
        # Recompute pose-derived evidence after integration, no dynamics step.
        mujoco.mj_kinematics(model,data)
        row=state(model,data);row.update(tick=tick,input_torque_nm=.24,load_force_world_n=force.tolist(),external_generalized_torque_nm=external.tolist())
        rows.append(row)
        if abs(data.time-before-DT)>1e-12:raise RuntimeError('Jaw diagnostic violated single20msintegration')
        if row['warnings']:break
    return {'initial_jaw_rad':initial,'load_n':load_n,'load_site':load_site,'integrations':len(rows),
            'all_finite':all(np.isfinite(list(r['angles_rad'].values())).all() for r in rows),
            'max_angle_relation_error_rad':max(max(abs(r['rotor_jaw_error_rad']),abs(r['coupler_jaw_error_rad'])) for r in rows),
            'max_output_pin_distance_m':max(r['output_pin_distance_m'] for r in rows),'rows':rows}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True);parser.add_argument('--contract',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv);model_path,contract_path,output=(p.resolve() for p in (args.model,args.contract,args.output))
    output.parent.mkdir(parents=True,exist_ok=True);os.chdir(output.parent)
    contract=json.loads(contract_path.read_text())
    if sha256(model_path)!=contract['model_sha256']:raise ValueError('Frozen jawsourceidentitymismatch')
    fixture_path=output.parent/'isolated_original_jaw50.xml';model=make_fixture(model_path,contract,fixture_path)
    initial_geometry=[]
    for q in np.linspace(0.,.55,12):
        data=mujoco.MjData(model);set_initial_pose(model,data,float(q));initial_geometry.append(state(model,data))
    cases=[dynamic_case(model,0.),dynamic_case(model,.275)]
    for site in ('jaw_output_pin','jaw_grip_load_point'):
        for load in (5.,10.,20.):cases.append(dynamic_case(model,.275,load,site))
    report={'schema':'goose_isolated_original_jaw50_diagnostic_v1','engine':'mujoco_cpu','engine_version':mujoco.__version__,
            'source_model_sha256':sha256(model_path),'source_contract_sha256':sha256(contract_path),'fixture_sha256':sha256(fixture_path),
            'diagnostic_code_sha256':sha256(Path(__file__)),'physics_dt_s':DT,'integrations_per_tick':1,'substeps':0,
            'preserved':['threeoriginalhinges','bodymass/COM/fullinertia','inputarmature/frictionloss','joint-equalitysolref/solimp','jointlimits'],
            'fixed_head_parent':True,'gravity_m_s2':[0,0,0],'contacts_enabled':False,'source_solref_s':.002,
            'refsafe_enabled':not bool(model.opt.disableflags&int(mujoco.mjtDisableBit.mjDSBL_REFSAFE)),
            'effective_minimum_timeconstant_s':2*DT,'qpos_writes_after_initialization':0,'initial_geometry':initial_geometry,'dynamic_cases':cases,
            'load_definition':'resistanceoppositepositivejawopeningvelocityatactual12mmoutputpinorcontractgripmappedintomovingjawframe',
            'qualified':False,'optimizer_updates':0,'limits':['prescribedheadsupport','nobody/ground/selfcontact','noGooseM0qualification','nosourceconstraintretuning']}
    write_json(output,report);print(f'STATUS JAW_DIAGNOSTIC_ONLY {output}')


if __name__=='__main__':main()
