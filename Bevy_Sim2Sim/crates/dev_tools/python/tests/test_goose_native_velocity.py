"""Keep the velocity task's reward math and numeric lineage upstream-owned."""
from pathlib import Path
from types import SimpleNamespace

import torch
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.tasks.velocity.mdp.rewards import track_linear_velocity

from bevy_microduck_tools.goose.native_velocity import (
    make_native_velocity_cfg, track_root_com_linear_velocity)


def test_native_rewards_are_not_replaced_by_the_legacy_guidance(monkeypatch):
    import bevy_microduck_tools.goose.native_velocity as module
    cfg = SimpleNamespace(observations={"actor": SimpleNamespace(
        terms={"base": SimpleNamespace(func=None)})}, scene=SimpleNamespace(sensors=()),
        metrics={})
    monkeypatch.setattr(module, "make_development_env_cfg", lambda *a, **kw: cfg)
    monkeypatch.setattr(module, "add_foot_sensors", lambda c, p: c)
    actual = make_native_velocity_cfg(Path("unused.xml"), Path("unused.json"))
    stock = unitree_g1_flat_env_cfg()
    assert actual.rewards.keys() == stock.rewards.keys()
    for name, term in actual.rewards.items():
        assert term.weight == stock.rewards[name].weight
        expected = (track_root_com_linear_velocity if name == "track_linear_velocity"
            else stock.rewards[name].func)
        assert term.func is expected
        for key, value in term.params.items():
            if key not in ("asset_cfg", "command_name", "sensor_name", "height_sensor_name",
                           "std_walking", "std_running"):
                assert value == stock.rewards[name].params[key]
    ranges = actual.commands["velocity"].ranges
    assert ranges.lin_vel_x == (-.15, .3)
    assert ranges.lin_vel_y == (-.1, .1)
    assert ranges.ang_vel_z == (-.6, .6)
    assert actual.terminations["contact_domain_failure"].params == {"max_depth_m": .05}

    # Exercise the installed class with the actual task parameters. Its
    # function default is .5, but the task explicitly supplies .05.
    from bevy_microduck_tools.goose.artifacts import JOINT_ORDER
    commands = torch.tensor([[0., 0., 0.], [.3, 0., 0.], [-.15, 0., 0.],
        [0., .1, 0.], [0., -.1, 0.], [0., 0., .6]])
    joints = torch.zeros((6, 18))
    joints[:, 8] = .1
    asset = SimpleNamespace(data=SimpleNamespace(default_joint_pos=torch.zeros_like(joints),
        joint_pos=joints), find_joints=lambda _: (list(range(18)), JOINT_ORDER))
    env = SimpleNamespace(scene={"robot": asset}, device="cpu",
        command_manager=SimpleNamespace(get_command=lambda _: commands))
    term = actual.rewards["pose"]
    pose = term.func(term, env)
    configured = pose(env, **term.params)
    assert configured[1:].min() > configured[0]
    no_explicit_threshold = {k: v for k, v in term.params.items() if k != "walking_threshold"}
    default_result = pose(env, **no_explicit_threshold)
    torch.testing.assert_close(default_result[1:5], default_result[0].expand(4), atol=0, rtol=0)


def test_com_point_adapter_uses_the_original_reward_including_vertical_velocity():
    original_link = torch.tensor([[9., 8., 7.], [-9., -8., -7.]])
    com = torch.tensor([[.1, -.05, .2], [-.1, .05, -.2]])
    commands = torch.tensor([[.15, -.1, 0.], [-.15, .1, 0.]])
    data = SimpleNamespace(root_link_lin_vel_b=original_link, root_com_lin_vel_b=com)
    env = SimpleNamespace(scene={"robot": SimpleNamespace(data=data)},
        command_manager=SimpleNamespace(get_command=lambda _: commands))
    expected = SimpleNamespace(scene={"robot": SimpleNamespace(
        data=SimpleNamespace(root_link_lin_vel_b=com))}, command_manager=env.command_manager)
    before = original_link.clone(), com.clone()
    torch.testing.assert_close(track_root_com_linear_velocity(env, std=.5, command_name="velocity"),
        track_linear_velocity(expected, std=.5, command_name="velocity"), atol=0, rtol=0)
    assert torch.equal(data.root_link_lin_vel_b, before[0])
    assert torch.equal(data.root_com_lin_vel_b, before[1])
