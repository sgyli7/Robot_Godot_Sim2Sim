"""Independent native-controller and analytical checks for the Torch adapter."""
import copy

import mujoco
import numpy as np
import pytest
import torch

from bevy_microduck_tools.goose.batch_drive import BatchedGooseDrive, BatchedNominalNeckGravity
from test_goose_50hz import fixture_runtime


@pytest.fixture(autouse=True)
def native_logs_stay_in_test_temp_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def nominal_graph(contract, mass=0., passive=False):
    """Add an explicitly synthetic neck graph to the zero-FF controller rig."""
    result = copy.deepcopy(contract)
    result["root_origin_at_zero_m"] = [0., 0., 0.]
    result["bodies"] = []
    parent = "torso"
    for i, joint in enumerate(result["joints"][:6]):
        joint.update(parent=parent, pivot_world_at_zero_m=[.1*(i+1), 0., 0.], axis_parent=[0., 1., 0.])
        result["bodies"].append(dict(name=joint["name"], mass_kg=mass, com_local_m=[.05, 0., 0.]))
        parent = joint["name"]
    if passive:
        result["passive_linkage_joints"] = [dict(name="coupler", parent=parent,
            pivot_world_at_zero_m=[.7, 0., 0.], axis_parent=[0., 1., 0.],
            mimic_joint=parent, mimic_multiplier=-1., mimic_offset_rad=0.)]
        result["bodies"].append(dict(name="coupler", mass_kg=mass, com_local_m=[.05, 0., 0.]))
    return result


@pytest.mark.parametrize("mode", ["nominal", "delay", "derated", "heat", "power", "speed"])
def test_batch_effort_and_history_match_independent_native_ticks(tmp_path, mode):
    references = []
    for i in range(2):
        directory = tmp_path / str(i)
        directory.mkdir()
        reference, _ = fixture_runtime(directory)
        references.append(reference)
    contract = nominal_graph(references[0].contract)
    if mode == "power":
        contract["positive_mechanical_power_limit_w"] = .01  # Exercise scaling with this small rig.
    drive = BatchedGooseDrive(contract, 2)
    rng = np.random.default_rng(941)
    for i, reference in enumerate(references):
        reference.contract["positive_mechanical_power_limit_w"] = contract["positive_mechanical_power_limit_w"]
        reference.strength = .7 if mode in ("derated", "heat") else 1.
        reference.delay = int(mode == "delay")
        reference.target[:] = .4 if mode in ("heat", "power") else 0.
        reference.thermal[:] = 1.1 if mode == "heat" else 0.
        reference.data.qpos[reference.qidx] = rng.uniform(-.2, .2, 18)
        reference.data.qvel[reference.vidx] = 2. if mode == "speed" else rng.uniform(-.6, .6, 18)
        reference.data.qvel[:6] = rng.uniform(-.2, .2, 6)
        reference.commands = rng.uniform(-.3, .3, 3)
        mujoco.mj_forward(reference.model, reference.data)
        drive.strength[i] = reference.strength
        drive.delay[i] = bool(reference.delay)
        drive.target[i] = torch.from_numpy(reference.target)
        drive.thermal[i] = torch.from_numpy(reference.thermal)
    for tick in range(20):
        actions = rng.uniform(-2, 2, (2, 18))  # Includes clipped inputs and reversals.
        if tick < 2:
            actions[:] = 1 if tick == 0 else -1
        q = np.stack([r.data.qpos[r.qidx] for r in references])
        qd = np.stack([r.data.qvel[r.vidx] for r in references])
        quats = np.stack([r.data.qpos[3:7] for r in references])
        effort = drive.prepare(actions, q, qd, quats)
        if mode == "power":
            positive_power = np.where(effort.numpy()*qd > 0, effort.numpy()*qd, 0).sum(axis=1)
            assert np.all(positive_power <= .01+1e-14)
        if mode == "speed" and tick == 0:
            assert not np.any(effort.numpy())
        for i, reference in enumerate(references):
            reference.step(actions[i])  # Real integration, not a replacement formula.
            np.testing.assert_allclose(effort[i], reference.last_tau, atol=2e-14, rtol=2e-14)
            np.testing.assert_allclose(drive.target[i], reference.target, atol=2e-14, rtol=2e-14)
            np.testing.assert_allclose(drive.thermal[i], reference.thermal, atol=2e-14, rtol=2e-14)
            assert reference.physics_integrations == tick+1
        drive.commit()
        observed = drive.observations(
            np.stack([r.data.qpos[r.qidx] for r in references]),
            np.stack([r.data.qvel[r.vidx] for r in references]),
            np.stack([r.data.qvel[3:6] for r in references]),
            np.stack([r.data.xmat[r.torso].reshape(3, 3) for r in references]),
            np.stack([r.commands for r in references]))
        np.testing.assert_array_equal(observed.numpy(), np.stack([r.observations() for r in references]))
        np.testing.assert_allclose(drive.phase, [r.phase for r in references], atol=2e-14)
        assert drive.completed_ticks.tolist() == drive.prepared_ticks.tolist() == [tick+1]*2


def test_nominal_neck_weight_has_correct_lever_arm_imu_sign_and_passive_mass(tmp_path):
    reference, _ = fixture_runtime(tmp_path)
    gravity = BatchedNominalNeckGravity(nominal_graph(reference.contract, mass=1., passive=True))
    # Straight horizontal chain, 6 one-kg neck bodies and a one-kg passive coupler.
    # Each y joint must counter gravity at all downstream nominal COM lever arms.
    expected = np.array([-9.81*(sum(.1*(j-i)+.05 for j in range(i, 6))+.1*(6-i)+.05)
                         for i in range(5)])
    efforts = gravity(np.zeros((3, 6)), np.array([[1., 0., 0., 0.], [-2., 0., 0., 0.], [0., 1., 0., 0.]]))
    np.testing.assert_allclose(efforts.numpy(), [expected, expected, -expected], atol=1e-13)
    assert efforts.shape == (3, 5)  # No beak or leg FF output.
    with pytest.raises(ValueError, match="Zero IMU"):
        gravity(np.zeros((1, 6)), np.zeros((1, 4)))


def test_pending_tick_cannot_double_prepare_commit_or_silently_reset(tmp_path):
    reference, _ = fixture_runtime(tmp_path)
    drive = BatchedGooseDrive(nominal_graph(reference.contract), 1)
    args = (np.ones((1, 18)), np.zeros((1, 18)), np.zeros((1, 18)), np.array([[1., 0., 0., 0.]]))
    with pytest.raises(RuntimeError, match="No prepared"):
        drive.commit()
    drive.prepare(*args)
    before = drive.target.clone(), drive.thermal.clone()
    with pytest.raises(RuntimeError, match="One drive preparation"):
        drive.prepare(*args)
    with pytest.raises(RuntimeError, match="pending Tick"):
        drive.reset()
    assert torch.equal(drive.target, before[0]) and torch.equal(drive.thermal, before[1])
    assert drive.completed_ticks.tolist() == [0]
    assert drive.phase.tolist() == [0.] and not bool(drive.actions.any())


def test_selective_episode_reset_does_not_erase_other_world_history(tmp_path):
    reference, _ = fixture_runtime(tmp_path)
    drive = BatchedGooseDrive(nominal_graph(reference.contract), 2)
    drive.prepare(np.ones((2, 18)), np.zeros((2, 18)), np.zeros((2, 18)), np.tile([1., 0., 0., 0.], (2, 1)))
    drive.commit()  # Drive-history unit test; no physics qualification is claimed.
    second = drive.target[1].clone(), drive.thermal[1].clone(), drive.phase[1].clone()
    drive.reset([0])
    assert drive.completed_ticks.tolist() == [0, 1]
    assert not bool(drive.actions[0].any()) and bool(drive.actions[1].all())
    assert torch.equal(drive.target[1], second[0]) and torch.equal(drive.thermal[1], second[1])
    assert torch.equal(drive.phase[1], second[2])


def test_batch_input_rejection_and_observation_clipping_match_native_contract(tmp_path):
    reference, _ = fixture_runtime(tmp_path)
    contract = nominal_graph(reference.contract)
    drive = BatchedGooseDrive(contract, 1)
    reference.commands[:] = 1000.
    reference.data.qpos[reference.qidx] = 1000.
    reference.data.qvel[reference.vidx] = 1000.
    reference.data.qvel[3:6] = 1000.
    obs = drive.observations(reference.data.qpos[reference.qidx][None], reference.data.qvel[reference.vidx][None],
        reference.data.qvel[3:6][None], reference.data.xmat[reference.torso].reshape(1, 3, 3), reference.commands[None])
    np.testing.assert_array_equal(obs.numpy()[0], reference.observations())
    assert obs.dtype == torch.float32 and obs.shape == (1, 65)
    with pytest.raises(ValueError, match="nonfinite"):
        drive.prepare(np.full((1, 18), np.nan), np.zeros((1, 18)), np.zeros((1, 18)), [[1., 0., 0., 0.]])
    assert not bool(drive.prepared_ticks.any()) and not bool(drive.target.any())
    bad = copy.deepcopy(contract)
    bad["physics_dt_s"] = .005
    with pytest.raises(ValueError, match="50 Hz"):
        BatchedGooseDrive(bad, 1)
    source = tmp_path / "src/sai_agent/goose/stage_one_gravity.py"
    source.write_text(source.read_text()+"\n# changed source identity\n")
    with pytest.raises(ValueError, match="source identity"):
        BatchedGooseDrive(contract, 1)
