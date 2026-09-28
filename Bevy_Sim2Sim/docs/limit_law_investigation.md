# 冻结 MicroDuck 关节限位法则调查（诊断，不开放 BAM）

更新日期：2026-09-28。本调查只读原始编译 MuJoCo 模型，并复用 `paired_force_probe` 已保存的 Rapier 单步追踪；没有运行 `SourceCollisionWorld`，也没有消耗其 128 次积分预算。两侧仍以 `1/60 s` 单步、相同初态和零 actuator torque 比较。此处不声称限位或完整 Sim2Sim 等价。

## 输入身份和真实源参数

| 本体 | 冻结 `.mjb` SHA256 | 超限 `qpos` SHA256 | `head_roll` joint / qpos / DOF |
| --- | --- | --- | --- |
| 腿式 | `832e1f08a1e328d8498be565874691348b89bc13041ff0d9eebf3f06c6bdc47c` | `e16f8e7ddf1d1cf2061b4c760faa1cbf78536d1cd8e6b992ca415eef10ae21bf` | `9 / 15 / 14` |
| 轮式 | `ea54c7c2b0aa8fbc3e9a0e3a431214650cbd4bdbd934d6707bb0aa52ac475191` | `df50ef2c31243ef01c6419f18238008c9a2d8d3e444073b82c61cb27b3fb7773` | `11 / 17 / 16` |

本机路径分别为 `.scratch/root_robot_assembly_mujoco_v1/{leg,roller}_native_model.mjb` 和 `.scratch/paired_force_v1/{leg,roller}_head_roll_limit_qpos.json`。两模型的 `head_roll` 上限均为 `0.4363323129985735 rad`，下限约 `-0.436332313 rad`，`jnt_margin=0`，`jnt_solref=[0.02, 1]`，`jnt_solimp=[0.9, 0.95, 0.001, 0.5, 2]`。MuJoCo 3.10.0 的该冻结模型设置 `opt.timestep=1/60 s`、Euler 积分、Newton 求解器、`iterations=100`、`tolerance=1e-8`，`disableflags=0`。

`solref` 的正值格式代表时间常数和阻尼比。由于默认 `refsafe` 开启，实际时间常数被抬到 `max(0.02, 2/60)=1/30 s`；违反深度大于 `solimp` 的 `width=0.001 rad` 时，阻抗 `d=d_w=0.95`。MuJoCo 的[官方求解参数公式](https://mujoco.readthedocs.io/en/latest/modeling.html#solver-parameters)于是给出 `b=2/(d_w τ)=63.1578947 s⁻¹`，`k=d/(d_w² τ² ζ²)=947.368421 s⁻²`，参考加速度 `a_ref=-b(Jv)-kr`。对上限超出 `0.02 rad` 且初速为零，源残差 `r=-0.02 rad`，`Jv=0`，故 `a_ref=18.947368421 rad/s²`。`solimp` 宽度以内的阻抗随违反深度变化，不能把该常数延伸到阈值附近。

MuJoCo 的[单约束无约束极小值公式](https://mujoco.readthedocs.io/en/latest/computation/#parameters)为 `f⁺=(A+R)⁻¹(a_ref-a₀)`；这里 `A=JM⁻¹Jᵀ`，`R` 是源求解器该行正则项，`a₀` 是该行的无约束加速度。此状态 `ncon=0`、`nefc=1`，唯一行类型为关节限位，`J[head_roll]=-1`，且该行最优力为正。因此以下数值可直接由冻结 `MjData` 验证，而非从 Rapier 的中途冲量猜测：

| 本体 | `A` | `R=efc_R` | `a_ref=efc_aref` | `a₀` | `efc_force` | `qfrc_constraint[head_roll]` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 腿式 | `512.660431016` | `27.015219028` | `18.947368421` | ≈ `0` | `0.035108807335 N·m` | `-0.035108807335 N·m` |
| 轮式 | `512.632030976` | `27.012833597` | `18.947368421` | `0` | `0.035110810210 N·m` | `-0.035110810210 N·m` |

现有 Rapier 行追踪 `.scratch/paired_force_v1/{leg,roller}_limit_trace_v1.json` 的 `rhs / biased_impulse` 分别为 `512.660449725` 和 `512.631945169`，与源 `A` 相差约 `1.9e-5` / `8.6e-5`。这说明在此姿态下关节有效逆质量高度接近；**并不**使两种限位力相等。Rapier 默认 joint softness 为 `1e6 Hz, ζ=1`，其限位行采用超界角度乘 `erp_inv_dt` 的速度偏差；当前 `0.02 rad` 案例的带偏差行 `rhs≈1.1999776 rad/s`、冲量 `≈0.0023407 N·m·s`，随后无偏差阶段将行累计冲量归零，最终角度钳到上限。源为软限位，步末仍约 `0.4513326 rad`、`-0.299982 rad/s`。原生中途冲量不能直接除以 `dt` 充当源 `qfrc_constraint`。

## 最小可复现实验矩阵

以各自冻结 `qpos` 为基准，仅覆盖 `head_roll` 为 `upper + depth`；所有其他位置不变，`qvel` 除该 DOF 外为零，`ctrl` 为零。深度选 `-0.001, 0, +0.0002, +0.0005, +0.001, +0.005, +0.02 rad`，该 DOF 速度选 `0, +0.3, -0.3 rad/s`；两种本体各 21 组，每组新建 `MjData`。在 `mj_forward` 后记录 `ncon/nefc`、`efc_type/id/J/pos/vel/aref/R/force`、`qfrc_constraint`，再做恰好一次 `mj_step`，在再次调用 `mj_forward` **之前**记录步末位置、速度和该步 `qfrc_constraint`。每组 native 侧由全新直连 `SimulationWorld` 与同一 `qpos/qvel`、零 actuator torque 做单步；记录全部原生限位阶段、最终 q/v 和原生力项，不经过 `SourceCollisionWorld`。

源参考值中的判别点：

| 深度 / 初速 | 腿式 `qfrc_constraint` | 轮式 `qfrc_constraint` | 现象 |
| --- | ---: | ---: | --- |
| `-0.001 / +0.3` | `0` | `0` | 步前未触发，虽本步将越界 |
| `0 / +0.3` | `0` | `0` | 精确边界步前尚无活动行 |
| `+0.0002 / 0` | `-0.000317881` | `-0.000317900` | `solimp` 过渡带 |
| `+0.0005 / 0` | `-0.000832075` | `-0.000832123` | `solimp` 过渡带 |
| `+0.001 / 0` | `-0.001755374` | `-0.001755474` | 达到宽度上沿 |
| `+0.02 / 0` | `-0.035108807` | `-0.035110810` | 软限位，仍残留约 `0.015 rad` 越界 |
| `+0.02 / +0.3` | `-0.070217428` | `-0.070221435` | 向外速度增加阻尼力 |
| `+0.02 / -0.3` | `0` | `0` | 向内脱离，无拉力 |

从 `Bevy_Sim2Sim` 目录复核源侧全部 42 组，可用安装了 MuJoCo 3.10.0 的 Python 执行下列只读脚本；本机环境是 `/home/ethan/Projects/Sai_Lab/upstream/pollen-robotics-microduck_rl/.venv/bin/python`。脚本先校验输入 SHA，再输出每组求解前的行参数和 `mj_step` 的求解结果：

同一矩阵也已落为仓库脚本 `crates/dev_tools/python/scripts/limit_law_matrix.py`。它对每组重新创建 `MjData`，核对唯一活动行的类型/DOF/Jacobian，记录原生 `efc_aref`、`efc_R`、`J M⁻¹ Jᵀ`、求解外力与步末 q/v；输出 JSON 明确标记目标限位/BAM 未验收。两种本体各 21 组、其中各 15 组步前有活动行。本机原始报告 `.scratch/paired_force_v1/leg_limit_matrix.json` 与 `roller_limit_matrix.json` 的 SHA256 分别为 `c30cfd248770f4eb761672616f6417ead877337624f590f8602459475b730a9e`、`48b23a63c220f9713f38a022a9254d920aad67fd2411ca87119f83fd970f027d`。

独立的只读复算还从**编译模型和初态**重建了全部 42 组单行结果，而不把 `efc_R` 或 `efc_force` 当输入。此冻结模型的二次 `solimp` 过渡给出深度 `x=clamp(depth/0.001,0,1)`、`s(x)=2x²`（`x≤0.5`）或 `1−2(1−x)²`，以及阻抗 `d=0.9+0.05s(x)`。`refsafe` 将 `solref` 的时间常数从 `0.02` 提到 `2/60=1/30 s`；以 `d_width=0.95`，阻尼系数 `b=2/(d_width·τ)`、刚度系数 `k=d/(d_width²·τ²)`，故 `aref=−b·Jv−k·r`。正则项必须是 `R=(1−d)/d·dof_invweight0[head_roll]`，即用编译时的逆权重，不是当前状态的精确逆质量 `A=JM⁻¹Jᵀ`。自由加速度 `a₀=J M⁻¹(qfrc_passive+qfrc_actuator+qfrc_applied−qfrc_bias)`；此单关节行 `Jdot·v=0`。单边力为 `f=max(0,(aref−a₀)/(A+R))`，源关节约束力矩为 `−f`。

源端每种本体的 21 案中，15 案有活动行，其中 10 案有正力、5 案单边投影为零；其余 6 案无行。重算的最大力误差腿式 `1.804e−16 N·m`、轮式 `1.249e−16 N·m`。误用实时 `A` 构造 `R` 则最大误差分别为 `4.306e−6` 与 `4.190e−6 N·m`。只读脚本、摘要与结果在 `.scratch/limit_formula_validation_v1/`，结果 SHA256 为 `4280a2d12e34bbac119c6aa28fa7a5b5cbb2d6218b04aecde7adc41d05432f80`。这仅证明冻结源端**单行**公式；Rapier 的约束前自由加速度、质量、实际限位行和步末状态仍须分别比对。

```python
import hashlib
import json
from pathlib import Path

import mujoco

inputs = {
    "leg": (
        "832e1f08a1e328d8498be565874691348b89bc13041ff0d9eebf3f06c6bdc47c",
        "e16f8e7ddf1d1cf2061b4c760faa1cbf78536d1cd8e6b992ca415eef10ae21bf",
    ),
    "roller": (
        "ea54c7c2b0aa8fbc3e9a0e3a431214650cbd4bdbd934d6707bb0aa52ac475191",
        "df50ef2c31243ef01c6419f18238008c9a2d8d3e444073b82c61cb27b3fb7773",
    ),
}
for family, (mjb_sha, qpos_sha) in inputs.items():
    mjb = Path(f".scratch/root_robot_assembly_mujoco_v1/{family}_native_model.mjb")
    qpos_file = Path(f".scratch/paired_force_v1/{family}_head_roll_limit_qpos.json")
    assert hashlib.sha256(mjb.read_bytes()).hexdigest() == mjb_sha
    qpos_bytes = qpos_file.read_bytes()
    assert hashlib.sha256(qpos_bytes).hexdigest() == qpos_sha
    base = json.loads(qpos_bytes)
    model = mujoco.MjModel.from_binary_path(str(mjb))
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "head_roll")
    q, dof = model.jnt_qposadr[joint], model.jnt_dofadr[joint]
    upper = model.jnt_range[joint, 1]
    for depth in (-0.001, 0, 0.0002, 0.0005, 0.001, 0.005, 0.02):
        for velocity in (0, 0.3, -0.3):
            data = mujoco.MjData(model)
            data.qpos[:] = base
            data.qpos[q] = upper + depth
            data.qvel[:] = 0
            data.qvel[dof] = velocity
            data.ctrl[:] = 0
            mujoco.mj_forward(model, data)
            row = (
                data.nefc,
                data.efc_pos[: data.nefc].copy().tolist(),
                data.efc_vel[: data.nefc].copy().tolist(),
                data.efc_aref[: data.nefc].copy().tolist(),
                data.efc_R[: data.nefc].copy().tolist(),
            )
            mujoco.mj_step(model, data)
            print(family, depth, velocity, row,
                  data.qfrc_constraint[dof], data.qpos[q], data.qvel[dof])
```

第一道诊断分解：在**探针中**对 head_roll 暂时禁用 Rapier 硬限位，仅把上表源侧同状态 `qfrc_constraint` 作为一对等大反向物体力矩，经已有 `RobotAssembly::actuator_body_torques` 输入同一单步。原 actuator torque 仍为零，源力矩只是隔离积分/映射误差的 oracle，不是实现或等价证书。若 `+0.02 / 0` 的 native 终态接近上表源终态，便能将差异定位在限位定律而非惯量或力矩轴。之后再从源参数独立计算并接入专门限位行，必须用整张矩阵的**行激活、约束力和终态**一起验收；尤其不能只调一个状态的角度或末帧冲量。

该首项 oracle 已按上述方式在 scratch 定义中执行。两种本体都只将对应 `jnt_limited[head_roll]` 改为 false，冻结正式定义未更动。注入精确源力后，腿式步末位置/速度差为 `9.15e-9 rad` / `4.28e-7 rad/s`，轮式为 `1.74e-8 rad` / `5.35e-7 rad/s`；报告 SHA 和变体身份见 [同力矩单步对照](paired_actuation_comparison.md)。这把当前单点的主要不一致定位在限位法则；42 组矩阵和多约束情况仍未通过。

编译导出器现已接入 `jnt_solref`、`jnt_solimp`、`jnt_margin` 和 `dof_invweight0`，Rust 定义将四者作为完整集合校验长度与有限值；旧冻结 JSON/RON 仍可加载，且重新序列化不凭空补字段。用两份源 `.mjb` 生成的 scratch 增强定义分别通过真实 `compiled_definition` 校验，但**正式冻结定义未重导出或晋升**，因此生产入口还不能凭它计算限位。`robot_builder.rs` 仍仅将 `jnt_range` 变成 Rapier 硬限位。源相容实现还要固定 `refsafe`/步长，比较目标端约束前自由加速度和质量，再验实际行的力与终态；不能硬编码本页数值。Rapier 的 `GenericJoint.softness` 是整条关节的共同设置，也影响锁定轴，不能当成独立的 MuJoCo 限位旋钮。即使单行实验通过，接触/多限位耦合求解与 BAM `previous_solve_load` 仍需单独验收。
