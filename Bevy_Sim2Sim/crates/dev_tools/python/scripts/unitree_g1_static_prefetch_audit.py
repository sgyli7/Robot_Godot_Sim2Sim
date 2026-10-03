#!/usr/bin/env python3
"""Audit actual native T1 prefetch images/actions/clock, including partial runs.

Object truth is read after execution by the placement auditor alone. Original
paused suite scores remain unchanged. No Qwen or full task qualification.
"""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
from unitree_g1_static_capture_audit import digest


def audit(case,seed_offset=0):
    case=Path(case);native=case/'native'
    config=json.loads((case/'config.json').read_text())
    execution=json.loads((case/'execution.json').read_text())
    receipt=json.loads((native/'capture_receipt.json').read_text())
    startup_ticks=60 if config.get('static_startup',False) else 0
    if (digest(case/'config.json')!=execution['config_sha256']
            or config['policy'].get('prefetch_after_ticks')!=10
            or config['policy']['max_calls']!=8
            or receipt['pauses_for_camera_and_policy']
            or receipt['diagnostic_prefetch_after_ticks']!=10):
        raise ValueError('Actual run is not the bounded startup/static prefetch profile')
    rows=[json.loads(line) for line in (native/'owner_steps.jsonl').read_text().splitlines()]
    episode=config['runner']['body']['static_agile']['episode_id'];sequences={}
    for tick,row in enumerate(rows,1):
        body=row['body']['static_agile'];step=body['step_configuration']
        if (row['episode_id']!=episode or row['owner_episode_integrations']!=tick
                or body['integration_count']!=tick or step['physics_hz']!=50
                or np.float32(step['dt'])!=np.float32(.02)
                or step['num_internal_pgs_iterations']!=16
                or step['num_solver_iterations']!=1 or step['max_ccd_substeps']!=1
                or step['additional_solver_iterations_max']!=0):
            raise ValueError('Owner violated original integration or episode contract')
        if tick<=startup_ticks:
            if row['execution']['startup']['ticks']!=tick:
                raise ValueError('Initialization timeline is incomplete')
            continue
        applied=row['execution']['original_vla'] if startup_ticks else row['execution']
        sequence=1+(tick-startup_ticks-1)//40
        if (applied['sequence_id']!=sequence or applied['admitted_chunks']!=sequence
                or applied['frame_index']!=(tick-startup_ticks-1)%40
                or applied['execution_start_sim_ns']!=(startup_ticks+(sequence-1)*40)*20_000_000
                or applied['observation_age_ns']!=(tick-1)*20_000_000-applied['observation']['sim_time_ns']):
            raise ValueError('Original actions were restarted, skipped or restamped')
        sequences.setdefault(sequence,[]).append(applied)
    images=[];latencies=[]
    for sample in sorted((case/'policy_captures').iterdir()):
        model=json.loads((sample/'receipt.json').read_text());sequence=model['sequence_id']
        stamp=json.loads((native/f'live_stamp_{sequence:04}.json').read_text())
        image_tick=stamp['source_ticks'][0];start_tick=startup_ticks+(sequence-1)*40
        if (stamp['source_ticks']!=[image_tick,image_tick]
                or stamp['episode_id']!=episode
                or stamp['sim_time_ns']!=image_tick*20_000_000
                or model['sampling_seed']!=seed_offset+sequence
                or model['seed_offset']!=seed_offset
                or (sequence==1 and image_tick!=startup_ticks)
                or (sequence>1 and not start_tick-30<=image_tick<start_tick)):
            raise ValueError('Actual prefetch image or sampling identity is invalid')
        with np.load(sample/'observation.npz',allow_pickle=False) as observation:
            pixels=np.asarray(Image.open(native/f'live_ego_{sequence:04}.png').convert('RGB'))
            if not np.array_equal(observation['ego_view'][0],pixels):
                raise ValueError('Model input did not contain actual captured RGB')
            q=stamp['native_state']['measured_joints']['positions']
            for group,indices in [('left_arm',range(15,22)),('right_arm',range(29,36)),
                    ('left_hand',range(22,29)),('right_hand',range(36,43)),('waist',range(12,15))]:
                if not np.array_equal(observation[group][0],np.asarray([q[i] for i in indices],dtype=np.float32)):
                    raise ValueError('Model self state does not match its actual image Tick')
        reply_path=native/f'live_reply_{sequence:04}.json'
        if reply_path.exists():
            reply=json.loads(reply_path.read_text())
            if reply['observation']!=model['stamp']:
                raise ValueError('Reply observation identity changed')
            with np.load(sample/'actions.npz',allow_pickle=False) as actions:
                for group in ['left_arm','right_arm','left_hand','right_hand','waist','base_height_command','navigate_command']:
                    values=[([f['base_height_m']] if group=='base_height_command' else
                        f['navigate_mps_rps'] if group=='navigate_command' else f[group]) for f in reply['frames']]
                    if not np.array_equal(actions[group][0],np.asarray(values,dtype=np.float32)):
                        raise ValueError('Original model actions were modified before submission')
        if sequence in sequences and sequences[sequence][0]['observation']!=model['stamp']:
            raise ValueError('Actuated actions used a different observation')
        images.append({'sequence':sequence,'actual_image_tick':image_tick,
            'fixed_start_tick':start_tick,'actual_rgb_sha256':digest(native/f'live_ego_{sequence:04}.png'),
            'executed_frames':len(sequences.get(sequence,[]))})
        latencies.append(model['inference_seconds'])
    if (receipt['actual_integrations']!=len(rows)
            or receipt['actual_model_successes']!=len(rows)
            or receipt['owner_step_records']!=len(rows)
            or not receipt['owner_step_trace_complete'] or receipt['owner_step_trace_dropped']):
        raise ValueError('Actual owner/WBC/evidence counters disagree')
    task_rows=rows[startup_ticks:]
    task_wall=(task_rows[-1]['last_boundary_wall_ms']-task_rows[0]['last_boundary_wall_ms'])/1000 if len(task_rows)>1 else None
    ratio=(len(task_rows)-1)*.02/task_wall if task_wall and task_wall>0 else None
    complete=(len(rows)==startup_ticks+320 and len(images)==8 and all(len(sequences.get(s,[]))==40 for s in range(1,9)))
    placement_path=case/'placement_geometry_audit.json'
    placement=json.loads(placement_path.read_text()) if placement_path.exists() else None
    if placement and placement['input_sha256']['trace']!=digest(native/'owner_steps.jsonl'):
        raise ValueError('Placement auditor consumed a different trace')
    return {'schema':'g1_native_static_prefetch_audit_v1','protocol_verified':True,
        'complete':complete,'actual_integrations':len(rows),'startup_integrations':min(len(rows),startup_ticks),
        'original_task_integrations':len(task_rows),'actual_captured_model_inputs':len(images),
        'images':images,'task_boundary_wall_seconds':task_wall,'task_boundary_sim_wall_ratio':ratio,
        'task_boundary_ratio_in_accepted_range':bool(complete and ratio is not None and .98<=ratio<=1.02),
        'inference_p50_seconds':float(np.median(latencies)) if latencies else None,
        'inference_p95_seconds':float(np.percentile(latencies,95)) if latencies else None,
        'pending_ticks_final':receipt['pending_ticks'],
        'maximum_observed_pending_ticks':max((r['pending_ticks'] for r in rows),default=0),
        'control_deadlines_missed':receipt['control_deadlines_missed'],
        'failure_reason':receipt['failure_reason'],
        'strict_placement_passed':bool(placement and placement['diagnostic_release_window_passed']),
        'standing_all_ticks':bool(placement and all(s['standing'] for s in placement['samples'])),
        'continuous_full_task_qualified':False,'qwen_calls':0,'full_task_qualified':False,
        'source_commit':execution['code_commit'],'binary_sha256':execution['binary_sha256'],
        'trace_sha256':digest(native/'owner_steps.jsonl')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed-offset',type=int,default=0);args=p.parse_args();result=audit(args.case,args.seed_offset)
    with args.output.open('x') as file:json.dump(result,file,indent=2,allow_nan=False);file.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='images'},indent=2))
