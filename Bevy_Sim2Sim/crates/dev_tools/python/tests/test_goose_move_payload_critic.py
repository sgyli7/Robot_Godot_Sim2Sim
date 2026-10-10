from math import sqrt
from types import SimpleNamespace

import pytest
import torch
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg

from bevy_microduck_tools.goose.move_payload_critic import (
    payload_state_observation, with_move_payload_critic)


def data(pos, quat, linear, angular):
    return SimpleNamespace(**{name: torch.tensor(values, dtype=torch.float64)
        for name, values in dict(root_link_pos_w=pos, root_link_quat_w=quat,
            root_link_lin_vel_w=linear, root_link_ang_vel_w=angular).items()})


def make_env():
    robot = data([[1., 2., 3.], [0., 0., 0.]],
                 [[sqrt(.5), 0., 0., sqrt(.5)], [1., 0., 0., 0.]],
                 [[.3, 0., 0.], [0., 0., 0.]], [[0., 0., 2.], [0., 0., 0.]])
    # World offset Y is robot-frame X. A fixed point on the rotating robot has
    # velocity omega cross offset; it must have zero frame-relative derivative.
    payload = data([[1., 3., 3.], [0., 0., 1.]],
                   [[-sqrt(.5), 0., 0., -sqrt(.5)], [1., 0., 0., 0.]],
                   [[-1.7, 0., 0.], [.1, .2, .3]], [[0., 0., 2.], [.4, .5, .6]])
    return SimpleNamespace(num_envs=2, device="cpu", scene={
        "robot": SimpleNamespace(data=robot),
        "payload": SimpleNamespace(data=payload, indexing=SimpleNamespace(root_body_id=1))},
        sim=SimpleNamespace(model=SimpleNamespace(body_mass=torch.tensor([[10., .1], [10., .3]]))))


def test_rotating_root_frame_uses_link_origin_and_transport_velocity():
    observation = payload_state_observation(make_env())
    assert observation.shape == (2, 14) and observation.dtype == torch.float32
    assert torch.allclose(observation[0, :3], torch.tensor([1., 0., 0.]), atol=1e-6)
    assert torch.allclose(observation[0, 3:7], torch.tensor([1., 0., 0., 0.]), atol=1e-6)
    assert observation[0, 7:13].abs().max() < 1e-6
    assert torch.allclose(observation[1, 7:13], torch.tensor([.1, .2, .3, .4, .5, .6]))
    assert torch.equal(observation[:, -1], torch.tensor([.1, .3]))


def test_read_never_advances_or_mutates_native_state_and_quaternion_double_cover():
    env = make_env()
    before = {entity: {name: value.clone() for name, value in vars(item.data).items()}
              for entity, item in env.scene.items()}
    expected = payload_state_observation(env)
    for _ in range(3):
        out = payload_state_observation(env)
        assert torch.equal(out, expected)
        out[:] = -100.
    for entity, states in before.items():
        for name, value in states.items():
            assert torch.equal(getattr(env.scene[entity].data, name), value)
    env.scene["payload"].data.root_link_quat_w *= -1
    assert torch.equal(payload_state_observation(env), expected)


def test_same_actor_state_can_have_different_payload_state():
    env = make_env()
    env.scene["robot"].data = data([[0., 0., 0.]]*2, [[1., 0., 0., 0.]]*2,
                                 [[0., 0., 0.]]*2, [[0., 0., 0.]]*2)
    assert not torch.equal(payload_state_observation(env)[0], payload_state_observation(env)[1])


def actor_signal(env):
    return torch.zeros(env.num_envs, 65)


def test_opt_in_preserves_actor_physics_rewards_and_original_profile():
    cfg = SimpleNamespace(scene=SimpleNamespace(entities={"robot": {}, "payload": {}}),
        observations={group: ObservationGroupCfg(terms={"base": ObservationTermCfg(func=actor_signal)})
                      for group in ("actor", "critic")}, rewards={"tracking": 1.},
        commands=[.4, .7], control=(50, 200, 18))
    result = with_move_payload_critic(cfg)
    assert result.observations["actor"] == cfg.observations["actor"]
    assert "free_payload_state" not in cfg.observations["critic"].terms
    assert result.scene == cfg.scene and result.rewards == cfg.rewards
    assert result.commands == cfg.commands and result.control == cfg.control
    assert result.observations["critic"].terms["free_payload_state"].func is payload_state_observation
    with pytest.raises(ValueError, match="twice"):
        with_move_payload_critic(result)
    del cfg.scene.entities["payload"]
    with pytest.raises(ValueError, match="real robot"):
        with_move_payload_critic(cfg)
