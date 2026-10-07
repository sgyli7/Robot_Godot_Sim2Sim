"""Keep the velocity task's reward math and numeric lineage upstream-owned."""
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.tasks.velocity.mdp.rewards import track_linear_velocity

from bevy_microduck_tools.goose.native_velocity import (
    GooseSoleWalkRunCommandCfg, make_sole_walk_run_task_cfg,
    make_native_velocity_cfg, make_native_walk_run_cfg, track_root_com_linear_velocity)


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


@pytest.mark.parametrize("course_start_step", [0, 512*24])
def test_walk_run_course_uses_native_schedule_without_changing_rewards(
        monkeypatch, course_start_step):
    import bevy_microduck_tools.goose.native_velocity as module
    cfg = SimpleNamespace(observations={"actor": SimpleNamespace(
        terms={"base": SimpleNamespace(func=None)})}, scene=SimpleNamespace(sensors=()),
        metrics={})
    monkeypatch.setattr(module, "make_development_env_cfg", lambda *a, **kw: cfg)
    monkeypatch.setattr(module, "add_foot_sensors", lambda c, p: c)
    actual = make_native_walk_run_cfg(Path("unused.xml"), Path("unused.json"),
        course_start_step=course_start_step)
    stock = unitree_g1_flat_env_cfg()
    for name, term in actual.rewards.items():
        assert term.weight == stock.rewards[name].weight
        expected = (track_root_com_linear_velocity if name == "track_linear_velocity"
            else stock.rewards[name].func)
        assert term.func is expected
        if "std" in term.params:
            assert term.params["std"] == stock.rewards[name].params["std"]
    command = actual.commands["velocity"]
    assert command.init_velocity_prob == 0.
    assert command.rel_standing_envs == .2
    curriculum = actual.curriculum["walk_run_commands"]
    term = SimpleNamespace(cfg=command)
    env = SimpleNamespace(common_step_counter=course_start_step,
        command_manager=SimpleNamespace(get_term=lambda _: term))
    curriculum.func(env, None, **curriculum.params)
    assert command.ranges.lin_vel_x == (0., .4)
    env.common_step_counter = course_start_step+192*24-1
    curriculum.func(env, None, **curriculum.params)
    assert command.ranges.lin_vel_x == (0., .4)
    env.common_step_counter += 1
    curriculum.func(env, None, **curriculum.params)
    assert command.ranges.lin_vel_x == (-.15, .7)
    assert command.ranges.lin_vel_y == (-.1, .1)
    assert command.ranges.ang_vel_z == (-.6, .6)


def test_sole_command_real_prefix_reset_and_restored_stage_boundaries():
    cfg = GooseSoleWalkRunCommandCfg(entity_name="robot", course_start_step=12288,
        resampling_time_range=(3., 8.), heading_command=False,
        rel_standing_envs=0., rel_heading_envs=0., rel_world_envs=0.,
        rel_forward_envs=0., init_velocity_prob=0.,
        ranges=GooseSoleWalkRunCommandCfg.Ranges(
            lin_vel_x=(-.15, .7), lin_vel_y=(-.1, .1), ang_vel_z=(-.6, .6)))
    # The robot deliberately has no physical-state write API.
    env = SimpleNamespace(num_envs=4, device="cpu", step_dt=.02,
        common_step_counter=12288, episode_length_buf=torch.zeros(4, dtype=torch.long),
        scene={"robot": SimpleNamespace(data=SimpleNamespace(
            root_link_lin_vel_b=torch.zeros(4, 3), root_link_ang_vel_b=torch.zeros(4, 3)))})
    term = cfg.build(env)
    ids = torch.arange(4)
    term.reset(ids)
    assert term.course_stage == 0
    sampled = term.sampled_commands.clone()
    for age_before in range(50):
        assert torch.count_nonzero(term.command) == 0
        assert term.warm_prefix_active.all()
        env.episode_length_buf += 1  # Exactly one completed physics Tick.
        env.common_step_counter += 1
        term.compute(.02)
    torch.testing.assert_close(term.command, sampled, atol=0, rtol=0)
    assert not term.warm_prefix_active.any()
    # A finite episode reset restarts only that world's standing prefix.
    env.episode_length_buf[1] = 0
    term.reset(torch.tensor([1]))
    assert torch.count_nonzero(term.command[1]) == 0
    torch.testing.assert_close(term.command[[0, 2, 3]], sampled[[0, 2, 3]], atol=0, rtol=0)
    # New upstream reset calls _update_command(ids). Preserve the other
    # worlds' timers and command state on that scoped path as well.
    timers = term.time_left.clone()
    commands = term.command.clone()
    flags = term.warm_prefix_active.clone()
    term._update_command(torch.tensor([1]))
    torch.testing.assert_close(term.time_left, timers, atol=0, rtol=0)
    torch.testing.assert_close(term.command[[0, 2, 3]], commands[[0, 2, 3]], atol=0, rtol=0)
    torch.testing.assert_close(term.warm_prefix_active[[0, 2, 3]], flags[[0, 2, 3]], atol=0, rtol=0)
    # All-stage transitions use the restored absolute clock, not runner iter.
    for clock, stage, maximum in ((13055, 0, .06), (13056, 1, .4),
                                  (13823, 1, .4), (13824, 2, .7)):
        env.common_step_counter = clock
        term.compute(0.)
        assert term.course_stage == stage
        assert float(term._templates()[:, 0].max()) == pytest.approx(maximum)
        assert float(term.sampled_commands[:, 0].max()) <= maximum+1e-7
    assert (term.time_left >= 3.).all()  # Stage resampling renews native timer.
    assert not term.is_world_env.any() and not term.is_heading_env.any()


def test_sole_task_pairs_native_swing_and_goose_posture_without_interface_change(monkeypatch):
    import bevy_microduck_tools.goose.native_velocity as module
    cfg = SimpleNamespace(observations={"actor": SimpleNamespace(
        terms={"base": SimpleNamespace(func=None)})}, scene=SimpleNamespace(sensors=()),
        metrics={}, events={})
    monkeypatch.setattr(module, "make_development_env_cfg", lambda *a, **kw: cfg)
    monkeypatch.setattr(module, "add_foot_sensors", lambda c, p: c)
    actual = make_sole_walk_run_task_cfg(Path("unused.xml"), Path("unused.json"),
        course_start_step=12288)
    stock = unitree_g1_flat_env_cfg()
    assert isinstance(actual.commands["velocity"], GooseSoleWalkRunCommandCfg)
    assert actual.commands["velocity"].standing_ticks == 50
    assert actual.curriculum == {}
    assert actual.terminations["contact_domain_failure"].params == {"max_depth_m": .05}
    for name, term in actual.rewards.items():
        expected = (track_root_com_linear_velocity if name == "track_linear_velocity"
            else stock.rewards[name].func)
        assert term.func is expected
        assert term.weight == (1. if name == "air_time" else stock.rewards[name].weight)
    assert actual.rewards["air_time"].params["command_threshold"] == .01
    assert actual.rewards["air_time"].params["threshold_min"] == .05
    assert actual.rewards["air_time"].params["threshold_max"] == .5
    for name in ("foot_clearance", "foot_swing_height"):
        assert actual.rewards[name].params["target_height"] == .02
        assert actual.rewards[name].params["command_threshold"] == .01
    from bevy_microduck_tools.goose.artifacts import JOINT_ORDER
    commands = torch.tensor([[0., 0., 0.], [.06, 0., 0.], [.4, 0., 0.], [.7, 0., 0.]])
    joints = torch.zeros(4, 18); joints[:, 8] = .1
    asset = SimpleNamespace(data=SimpleNamespace(default_joint_pos=torch.zeros_like(joints),
        joint_pos=joints), find_joints=lambda _: (list(range(18)), JOINT_ORDER))
    env = SimpleNamespace(scene={"robot": asset}, device="cpu",
        command_manager=SimpleNamespace(get_command=lambda _: commands))
    pose_cfg = actual.rewards["pose"]
    reward = pose_cfg.func(pose_cfg, env)(env, **pose_cfg.params)
    assert reward[1] > reward[0]
    torch.testing.assert_close(reward[1:], reward[1].expand(3), atol=0, rtol=0)
    from bevy_microduck_tools.goose.foot_curriculum import CONTACT_SENSOR
    env.scene[CONTACT_SENSOR] = SimpleNamespace(data=SimpleNamespace(
        current_air_time=torch.tensor([[.1, 0.], [.1, 0.], [.1, 0.], [.6, 0.]])))
    env.extras = {"log": {}}
    air_cfg = actual.rewards["air_time"]
    air = air_cfg.func(env, **air_cfg.params)
    torch.testing.assert_close(air, torch.tensor([0., 1., 1., 0.]), atol=0, rtol=0)


def test_sole_landing_memory_reset_only_affects_the_reset_worlds():
    from bevy_microduck_tools.goose.native_velocity import reset_sole_landing_history
    peaks = torch.tensor([[.02, .03], [.04, .05], [.06, .07]])
    landing = SimpleNamespace(peak_heights=peaks)
    env = SimpleNamespace(reward_manager=SimpleNamespace(
        get_term_cfg=lambda _: SimpleNamespace(func=landing)))
    reset_sole_landing_history(env, torch.tensor([1]))
    torch.testing.assert_close(peaks, torch.tensor([[.02, .03], [0., 0.], [.06, .07]]),
        atol=0, rtol=0)


def test_native_units_profile_preserves_task_and_delegates_original_penalty(monkeypatch, tmp_path):
    import json
    import bevy_microduck_tools.goose.native_velocity as module
    from bevy_microduck_tools.goose.artifacts import JOINT_ORDER
    from bevy_microduck_tools.goose.mature_training import native_physical_action_std
    from mjlab.envs import mdp

    def base_cfg(*args, **kwargs):
        return SimpleNamespace(observations={"actor": SimpleNamespace(
            terms={"base": SimpleNamespace(func=None)})}, scene=SimpleNamespace(sensors=()),
            metrics={}, events={})

    monkeypatch.setattr(module, "make_development_env_cfg", base_cfg)
    monkeypatch.setattr(module, "add_foot_sensors", lambda cfg, _: cfg)
    contract = {"joint_order": list(JOINT_ORDER), "joints": [
        {"name": name, "torque_peak_limit_nm": 4.+i,
         "kp_nm_rad": 20.+i, "action_scale_rad": .1+.01*i}
        for i, name in enumerate(JOINT_ORDER)]}
    path = tmp_path/"contract.json"
    path.write_text(json.dumps(contract))
    original = module.make_sole_walk_run_task_cfg(Path("unused.xml"), path, course_start_step=0)
    corrected = module.make_sole_walk_run_native_units_cfg(Path("unused.xml"), path,
        course_start_step=0)
    assert corrected.commands == original.commands
    assert corrected.events == original.events
    assert corrected.terminations == original.terminations
    for name, term in corrected.rewards.items():
        before = original.rewards[name]
        assert term.weight == before.weight
        if name != "action_rate_l2":
            assert (term.func, term.params) == (before.func, before.params)
    native = torch.arange(36, dtype=torch.float64).reshape(2, 18)/20
    previous = native.flip(-1)/2
    scale = native.new_tensor(native_physical_action_std(contract))
    original_view = SimpleNamespace(action_manager=SimpleNamespace(
        action=native, prev_action=previous))
    public_view = SimpleNamespace(action_manager=SimpleNamespace(
        action=native*scale, prev_action=previous*scale))
    term = corrected.rewards["action_rate_l2"]
    torch.testing.assert_close(term.func(public_view, **term.params), mdp.action_rate_l2(original_view))
    torch.testing.assert_close(public_view.action_manager.action, native*scale, atol=0, rtol=0)
