"""Torch batching of the existing Goose drive and 65-value actor observation.

This robot adapter neither integrates physics nor runs a policy/optimizer.
Prepare once before the upstream step, then commit history once after a
successful integration. Keep failed preparations for diagnosis; never reset
them implicitly. All control arithmetic uses float64, as the NumPy reference
does; casting motor effort to the physics backend is the caller's boundary.
"""
from __future__ import annotations

from pathlib import Path

import torch

from .artifacts import DT, JOINT_ORDER, sha256

REVISION = "goose_batch_drive_v1"


def _quat_matrix(wxyz):
    norm = torch.linalg.vector_norm(wxyz, dim=-1, keepdim=True)
    if bool((norm == 0).any()):
        raise ValueError("Zero IMU quaternion")
    w, x, y, z = (wxyz / norm).unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(-1, 3, 3)


class BatchedNominalNeckGravity:
    """Same nominal encoder/IMU calculation as SHA-bound stage_one_gravity.

Only six neck/beak encoders and the IMU quaternion are inputs. Passive links
use the inherited nominal mimic approximation; actual passive coordinates,
randomized masses, root translation and contacts are deliberately absent.
"""

    def __init__(self, contract, *, device="cpu"):
        source = contract["source_checkpoint"]
        relative = "src/sai_agent/goose/stage_one_gravity.py"
        if sha256(Path(source["source_root"]) / relative) != source["source_module_sha256"][relative]:
            raise ValueError("Nominal gravity source identity mismatch")
        self.device = torch.device(device)
        tensor = lambda value: torch.as_tensor(value, dtype=torch.float64, device=self.device)
        joints = contract["joints"][:6]
        passive = contract.get("passive_linkage_joints", [])
        bodies = {b["name"]: b for b in contract["bodies"]}
        pivots = {"torso": tensor(contract["root_origin_at_zero_m"])}
        pivots.update({j["name"]: tensor(j["pivot_world_at_zero_m"]) for j in [*joints, *passive]})
        indices = {"torso": 0}
        active_indices = {j["name"]: i for i, j in enumerate(joints)}
        self.nodes = []
        for i, joint in enumerate([*joints, *passive]):
            name, parent = joint["name"], joint["parent"]
            axis = tensor(joint["axis_parent"])
            x, y, z = axis.unbind()
            zero = torch.zeros((), dtype=torch.float64, device=self.device)
            skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
            body = bodies[name]
            self.nodes.append((indices[parent], pivots[name]-pivots[parent], axis,
                skew, skew @ skew, tensor(body["com_local_m"]),
                tensor([0., 0., -9.81*body["mass_kg"]]),
                i if i < 6 else active_indices[joint["mimic_joint"]],
                1. if i < 6 else joint["mimic_multiplier"],
                0. if i < 6 else joint["mimic_offset_rad"]))
            indices[name] = i+1
        self.identity = torch.eye(3, dtype=torch.float64, device=self.device)
        self.descendants = torch.cat((torch.triu(torch.ones((5, 6), dtype=torch.float64, device=self.device)),
            torch.ones((5, len(passive)), dtype=torch.float64, device=self.device)), dim=1)

    @torch.no_grad()
    def __call__(self, q, imu_wxyz):
        q = torch.as_tensor(q, dtype=torch.float64, device=self.device)
        imu_wxyz = torch.as_tensor(imu_wxyz, dtype=torch.float64, device=self.device)
        if (q.ndim != 2 or q.shape[1] != 6 or imu_wxyz.shape != (q.shape[0], 4)
                or not bool(torch.isfinite(q).all() & torch.isfinite(imu_wxyz).all())):
            raise ValueError("Nominal gravity requires finite batched 6 encoders and 4 IMU values")
        positions = [torch.zeros((q.shape[0], 3), dtype=torch.float64, device=self.device)]
        rotations = [_quat_matrix(imu_wxyz)]
        axes, centers, forces = [], [], []
        for i, (parent, offset, axis, skew, skew2, com, force, source, multiplier, bias) in enumerate(self.nodes):
            pr = rotations[parent]
            positions.append(positions[parent] + pr @ offset)
            if i < 6:
                axes.append(pr @ axis)
            angle = multiplier*q[:, source]+bias
            rotations.append(pr @ (self.identity + angle.sin()[:, None, None]*skew
                                     + (1-angle.cos())[:, None, None]*skew2))
            centers.append(positions[-1] + rotations[-1] @ com)
            forces.append(force)
        diff = torch.stack(centers, dim=1)[:, None] - torch.stack(positions[1:6], dim=1)[:, :, None]
        moment = torch.linalg.cross(diff, torch.stack(forces)[None, None].expand_as(diff), dim=-1)
        moment *= self.descendants[None, :, :, None]
        return -(moment.sum(dim=2)*torch.stack(axes[:5], dim=1)).sum(dim=-1)


class BatchedGooseDrive:
    """18 ordered position targets, PD, nominal FF, caps and history at 50 Hz."""

    def __init__(self, contract, num_envs, *, device="cpu"):
        if (num_envs < 1 or contract["joint_order"] != list(JOINT_ORDER)
                or [j["name"] for j in contract["joints"]] != list(JOINT_ORDER)
                or any(contract[k] != DT for k in ("physics_dt_s", "torque_dt_s", "policy_dt_s"))
                or contract["physics_steps_per_tick"] != 1):
            raise ValueError("Incorrect Goose batch size, axis order or 50 Hz timing")
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.gravity = BatchedNominalNeckGravity(contract, device=self.device)
        joints = contract["joints"]
        for name, field in (("kp", "kp_nm_rad"), ("kd", "kd_nm_s_rad"),
                ("peak", "torque_peak_limit_nm"), ("cont", "continuous_design_limit_nm"),
                ("speed", "speed_limit_rad_s"), ("scale", "action_scale_rad"),
                ("neutral", "q_neutral_rad"), ("ranges", "range_rad")):
            setattr(self, name, torch.tensor([j[field] for j in joints], dtype=torch.float64, device=self.device))
        self.power_limit = contract["positive_mechanical_power_limit_w"]
        self.phase_increment = 2*torch.pi*contract["phase_frequency_hz"]*DT
        self.actions = self.neutral.new_zeros((num_envs, 18))
        self.target = self.neutral.repeat(num_envs, 1)
        self.thermal = torch.zeros_like(self.actions)
        self.last_tau = torch.zeros_like(self.actions)
        self.strength = self.neutral.new_ones((num_envs, 1))
        self.delay = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.phase = self.neutral.new_zeros(num_envs)
        self.prepared_ticks = torch.zeros(num_envs, dtype=torch.int64, device=self.device)
        self.completed_ticks = torch.zeros_like(self.prepared_ticks)
        self._pending_action = None

    def _values(self, value, shape):
        value = torch.as_tensor(value, dtype=torch.float64, device=self.device)
        if value.shape != (self.num_envs, *shape) or not bool(torch.isfinite(value).all()):
            raise ValueError("Wrong-size/nonfinite Goose batch input")
        return value

    @torch.no_grad()
    def reset(self, env_ids=None):
        """Explicit episode reset only; a failed/pending Tick must be preserved."""
        if self._pending_action is not None:
            raise RuntimeError("Preserve the pending Tick before resetting its drive")
        ids = slice(None) if env_ids is None else torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self.actions[ids] = 0
        self.target[ids] = self.neutral
        self.thermal[ids] = 0
        self.last_tau[ids] = 0
        self.strength[ids] = 1
        self.delay[ids] = False
        self.phase[ids] = 0
        self.prepared_ticks[ids] = 0
        self.completed_ticks[ids] = 0

    @torch.no_grad()
    def prepare(self, action, q, qd, imu_wxyz):
        """One effort preparation; q/qd use beak_hinge, motor 6 drives rotor."""
        if self._pending_action is not None:
            raise RuntimeError("One drive preparation per pending Tick")
        action, q, qd = (self._values(v, (18,)) for v in (action, q, qd))
        imu_wxyz = self._values(imu_wxyz, (4,))
        strength = self._values(self.strength, (1,))
        if bool((strength <= 0).any()):
            raise ValueError("Goose drive strength must be positive")
        ff = self.gravity(q[:, :6], imu_wxyz)
        action = action.clamp(-1, 1)
        command = torch.where(self.delay[:, None], self.actions, action)
        desired = (self.neutral+self.scale*command).clamp(self.ranges[:, 0], self.ranges[:, 1])
        self.target += (desired-self.target).clamp(-self.speed*DT, self.speed*DT)
        torque = self.kp*(self.target-q)-self.kd*qd
        torque[:, :5] += ff
        cap = self.peak*strength*(1-qd.abs()/(self.speed*1.3)).clamp(0, 1)
        cap = torch.minimum(cap, torch.where(self.thermal > (self.cont*strength)**2,
                                              self.cont, self.peak)*strength)
        torque = torque.clamp(-cap, cap)
        positive = torque*qd > 0
        power = torch.where(positive, torque*qd, 0).sum(dim=1, keepdim=True)
        factor = self.power_limit/power.clamp(min=self.power_limit)
        torque = torch.where(positive, torque*factor, torque)
        self.thermal += DT/2*(torque.square()-self.thermal)
        self.last_tau.copy_(torque)
        self.prepared_ticks += 1
        self._pending_action = action.clone()
        return torque.clone()

    @torch.no_grad()
    def commit(self):
        """Commit action/phase after the caller verifies one successful step.

        This counter is drive history, not proof that physics was integrated.
        The upstream environment must record its actual step/time separately.
        """
        if self._pending_action is None:
            raise RuntimeError("No prepared Tick to commit")
        self.phase.copy_((self.phase+self.phase_increment) % (2*torch.pi))
        self.actions.copy_(self._pending_action)
        self.completed_ticks += 1
        self._pending_action = None

    @torch.no_grad()
    def observations(self, q, qd, base_angular_velocity, torso_rotation_world, commands):
        """Same 65 entries as runtime; rotation must be from the completed pose."""
        q, qd = (self._values(v, (18,)) for v in (q, qd))
        angular, rotation, commands = (self._values(v, shape) for v, shape in (
            (base_angular_velocity, (3,)), (torso_rotation_world, (3, 3)), (commands, (3,))))
        gravity = -rotation[:, 2, :]
        result = torch.cat((angular*.25, gravity, commands, q-self.neutral, qd*.1,
            self.actions, self.phase.sin()[:, None], self.phase.cos()[:, None]), dim=1)
        return result.to(torch.float32).clamp(-20, 20)


class BatchedRigidNativeDrive(BatchedGooseDrive):
    """Ordered native position/motor commands, caps and actual-force history."""

    @torch.no_grad()
    def prepare(self, action, q, qd, imu_wxyz, *, motor_velocity):
        if self._pending_action is not None:
            raise RuntimeError("One native drive preparation per pending Tick")
        action, q, qd, motor_velocity = (self._values(v, (18,))
            for v in (action, q, qd, motor_velocity))
        imu_wxyz = self._values(imu_wxyz, (4,))
        strength = self._values(self.strength, (1,))
        if bool((strength <= 0).any()):
            raise ValueError("Native drive strength must be positive")
        action = action.clamp(-1, 1)
        command = torch.where(self.delay[:, None], self.actions, action)
        desired = (self.neutral+self.scale*command).clamp(self.ranges[:, 0], self.ranges[:, 1])
        self.target += (desired-self.target).clamp(-self.speed*DT, self.speed*DT)
        feed = torch.zeros_like(self.target)
        feed[:, :5] = self.gravity(q[:, :6], imu_wxyz)
        cap = self.peak*strength*(1-qd.abs()/(self.speed*1.3)).clamp(0, 1)
        cap = torch.minimum(cap, torch.where(self.thermal > (self.cont*strength)**2,
            self.cont, self.peak)*strength)
        bound = (cap*motor_velocity.abs()).sum(dim=1, keepdim=True)
        self.cap = cap*(self.power_limit/bound.clamp(min=self.power_limit))
        control = self.target+feed/self.kp
        control[:, 5] = (self.kp[5]*(self.target[:, 5]-q[:, 5])-self.kd[5]*qd[:, 5]).clamp(
            -self.cap[:, 5], self.cap[:, 5])
        self.prepared_ticks += 1
        self._pending_action = action.clone()
        return control

    @torch.no_grad()
    def commit(self, actual_force):
        actual_force = self._values(actual_force, (18,))
        if self._pending_action is None or bool((actual_force.abs() > self.cap+2e-6).any()):
            raise RuntimeError("Native force commit must follow a bounded prepared Tick")
        self.last_tau.copy_(actual_force)
        self.thermal += DT/2*(actual_force.square()-self.thermal)
        super().commit()
