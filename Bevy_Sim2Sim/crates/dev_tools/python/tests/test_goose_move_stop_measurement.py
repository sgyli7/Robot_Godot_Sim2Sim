from types import SimpleNamespace

import pytest
import torch

from bevy_microduck_tools.goose.motion_tracking import FilteredCommandStop
from bevy_microduck_tools.goose.move_stop_measurement import (
    CurrentCOMCommandStop,
    with_current_com_stop_measurement,
)


def make_env():
    goal = torch.zeros(1, 3)
    sensor = SimpleNamespace(data=torch.zeros(1, 3))
    data = SimpleNamespace(
        root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        root_link_ang_vel_b=torch.zeros(1, 3),
    )
    env = SimpleNamespace(
        step_dt=0.02, num_envs=1, device="cpu", common_step_counter=0,
        scene={"robot": SimpleNamespace(data=data),
               "robot/whole_robot_com_velocity": sensor},
        command_manager=SimpleNamespace(get_command=lambda name: goal),
    )
    cfg = SimpleNamespace(params={"tracking_tau_s": 0.2,
        "velocity_sensor_name": "robot/whole_robot_com_velocity"})
    return env, goal, sensor, cfg


def evaluate(term, env):
    return term(env, "velocity", 0.035, 0.2, "robot/whole_robot_com_velocity")


def test_alternating_speed_cannot_cancel_in_stop_reward():
    env, goal, sensor, cfg = make_env()
    filtered = FilteredCommandStop(cfg, env)
    current = CurrentCOMCommandStop(cfg, env)
    for tick in range(200):
        env.common_step_counter = tick
        sensor.data[0, 0] = 0.1 if tick % 2 else -0.1
        old = evaluate(filtered, env)
        new = evaluate(current, env)
    assert old.item() > 0.95
    assert new.item() < 0.001
    assert abs(sensor.data[0, 0]) > 0.04


def test_current_stop_preserves_native_formula_and_command_gate():
    assert CurrentCOMCommandStop.__call__ is FilteredCommandStop.__call__
    env, goal, sensor, cfg = make_env()
    current = CurrentCOMCommandStop(cfg, env)
    assert evaluate(current, env).item() == 1.0
    goal[0, 2] = 0.3
    assert evaluate(current, env).item() == 0.0


def test_current_stop_does_not_advance_or_reset_shared_tracking_history():
    env, goal, sensor, cfg = make_env()
    filtered = FilteredCommandStop(cfg, env)
    sensor.data[0, 0] = 0.1
    evaluate(filtered, env)
    state = filtered.state.mean.clone()
    ready = filtered.state.ready.clone()
    step = filtered.state.last_step
    current = CurrentCOMCommandStop(cfg, env)
    sensor.data.zero_()
    assert evaluate(current, env).item() == 1.0
    current.reset(torch.tensor([0]))
    assert env._goose_motion_tracking_state is filtered.state
    assert torch.equal(filtered.state.mean, state)
    assert torch.equal(filtered.state.ready, ready)
    assert filtered.state.last_step == step


def test_configuration_is_opt_in_and_preserves_other_terms_and_parameters():
    _, _, _, term_cfg = make_env()
    term_cfg.func = FilteredCommandStop
    term_cfg.weight = 2.0
    term_cfg.params.update(command_name="velocity", vel_std=0.035)
    cfg = SimpleNamespace(rewards={"command_stop": term_cfg,
        "track_linear_velocity": SimpleNamespace(weight=5.0)},
        observations={"actor": (65, 18)}, decimation=4)
    result = with_current_com_stop_measurement(cfg)
    assert cfg.rewards["command_stop"].func is FilteredCommandStop
    assert result.rewards["command_stop"].func is CurrentCOMCommandStop
    assert result.rewards["command_stop"].params == term_cfg.params
    assert result.rewards["command_stop"].weight == 2.0
    assert result.rewards["track_linear_velocity"] == cfg.rewards["track_linear_velocity"]
    assert result.observations == cfg.observations
    assert result.decimation == cfg.decimation
    with pytest.raises(ValueError, match="original filtered"):
        with_current_com_stop_measurement(result)
