"""Current native tracking bridge must see the actual11 physical leaves."""
import json
import os
from pathlib import Path

import mujoco
import numpy as np
import pytest
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

from bevy_microduck_tools.goose.native_tracking import (
    GROUND_SENSOR, make_native_tracking_teacher_cfg)


@pytest.fixture
def cfg():
    root = os.environ.get('GOOSE_CONTACT_SEQUENCE_EXPERIMENT')
    if root is None:
        pytest.skip('Requires the frozen actual Goose two-step reference')
    root = Path(root)
    inputs = json.loads((root/'protocol.json').read_text())['inputs']
    return make_native_tracking_teacher_cfg(Path(inputs['model']),Path(inputs['contract']),
        root/'own_actual_two_steps_motion.npz',
        cold_qpos=np.load(inputs['births'])['qpos'][0],num_envs=1)


def test_tracking_ground_sensor_covers_both_soles_and_all_physical_leaves(cfg):
    sensor = next(s for s in cfg.scene.sensors if s.name == GROUND_SENSOR)
    names = sensor.primary.pattern
    assert len(names) == 11
    assert 'right_flexible_sole' in names and 'left_flexible_sole' in names
    assert 'torso_envelope' in names and 'head_upper_bill_envelope' in names
    assert len(set(names)) == len(names)
    assert sensor.secondary.pattern == 'ground'
    assert sensor.global_frame


def test_tracking_adoption_keeps_native_rewards_and_public_interface(cfg):
    original = make_tracking_env_cfg()
    assert cfg.rewards.keys() == original.rewards.keys()
    for name,term in cfg.rewards.items():
        expected = original.rewards[name]
        assert (term.func,term.weight,term.params) == (expected.func,expected.weight,expected.params)
    assert cfg.decimation == 1 and cfg.sim.mujoco.timestep == .02
    assert not cfg.auto_reset
    assert 'public_actor' in cfg.observations
    assert set(cfg.observations['public_actor'].terms) == {'base'}
    assert cfg.commands['motion'].sampling_mode == 'start'
