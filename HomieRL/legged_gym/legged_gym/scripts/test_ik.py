#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import pinocchio as pin
import numpy as np
from pathlib import Path

URDF_PATH = "/home/tzh/OpenHomie/HomieRL/legged_gym/resources/robots/g1_description/g1.urdf"
EE_NAME   = "right_wrist_yaw_link"

TARGET_POS = np.array([0.1, -0.35, 0.5])
TARGET_ROT = np.eye(3)

model = pin.buildModelFromUrdf(URDF_PATH)
data  = model.createData()

fixed_joint=[]
for idx, jname in enumerate(model.names):
    if idx > 0 and idx < 21:
        fixed_joint.append(model.getJointId(jname))
print("fixed_joint",fixed_joint)
# for name in fixed_joint
# print("joint name:", model.getJointId(name)) 

# ====== 3. 计算 full 模型的 neutral q 和连杆位置 ======
q_full_init = pin.neutral(model)
pin.normalize(model, q_full_init)
data_full = model.createData()
pin.framesForwardKinematics(model, data_full, q_full_init)

# 收集每个关节原点在世界坐标下的位置（data_full.oMi）
joint_positions = np.array([data_full.oMi[j].translation for j in range(model.njoints)])


# ====== 3. 初始配置 ======
q = pin.neutral(model)
pin.normalize(model, q)

reduced_model = pin.buildReducedModel(
    model,
    fixed_joint,           
    q                      
)
ee_id = reduced_model.getFrameId(EE_NAME)
reduced_q = pin.neutral(reduced_model)
reduced_data = reduced_model.createData()
pin.normalize(reduced_model, reduced_q)

print("reduced_model:",reduced_model)
print("reduced_q:",reduced_q)
print("reduced_data:",reduced_data)

pin.framesForwardKinematics(reduced_model, reduced_data, reduced_q)
oMf_init = data.oMf[ee_id]  # 当前“neutral”姿态下的末端 SE3
print("start----q:",reduced_q)
print("=== 当前 SE3 (oMf_init) ===",oMf_init)             # 4×4 矩阵形式


# ====== 4. 打印目标 SE3 ======
target_SE3 = pin.SE3(TARGET_ROT, TARGET_POS)
print("=== 目标 SE3 (target_SE3) ===",target_SE3)           # 4×4 矩阵形式



# ====== 5. 完整 SE(3) IK 迭代 （Jlog6 + 阻尼伪逆） ======
IT_MAX = 100
EPS    = 1e-4
DAMP   = 1e-12

print("开始 SE(3) IK 迭代 …")
for i in range(1, IT_MAX + 1):
    pin.framesForwardKinematics(reduced_model, reduced_data, reduced_q)
    oMf = reduced_data.oMf[ee_id]

    iMd = oMf.actInv(target_SE3)
    err = pin.log(iMd).vector
    err_norm = float(np.linalg.norm(err))

    if err_norm < EPS:
        print(f"✔ 收敛于第 {i} 步，‖err‖ = {err_norm:.4e}")
        break

    J6   = pin.computeFrameJacobian(reduced_model, reduced_data, reduced_q, ee_id, pin.LOCAL)
    Jlog = pin.Jlog6(iMd.inverse())
    J    = -Jlog.dot(J6)

    JJt = J.dot(J.T) + DAMP * np.eye(6)
    dq  = -J.T.dot(np.linalg.solve(JJt, err))
    reduced_q   = pin.integrate(reduced_model, reduced_q, dq)

    reduced_q[:] = np.minimum(np.maximum(reduced_q,
                                 reduced_model.lowerPositionLimit),
                     reduced_model.upperPositionLimit)
    pin.normalize(reduced_model, reduced_q)
else:
    print(f"✘ 未收敛 (达 {IT_MAX} 步)，最终 ‖err‖ = {err_norm:.4e}")

# ====== 6. 计算并打印求解后 SE3 ======
pin.framesForwardKinematics(reduced_model, reduced_data, reduced_q)
oMf_final = reduced_data.oMf[ee_id]
print("final----reduced_q:",reduced_q)
print("\n=== 求解后 SE3 (oMf_final) ===")
print(oMf_final)            # 4×4 矩阵形式
print()

# ====== 7. 位姿误差摘要 ======
delta = oMf_final.inverse() * target_SE3
pos_err = np.linalg.norm(delta.translation)
rot_err = np.linalg.norm(pin.log(delta).vector[:3])
print(f"末端位置误差 ‖Δp‖ = {pos_err:.4e} m")
print(f"末端姿态误差 ‖Δθ‖ = {rot_err:.4e} rad\n")

# ====== 11. 可视化工作空间 ======
NUM_SAMPLES = 8000

lower = reduced_model.lowerPositionLimit
upper = reduced_model.upperPositionLimit

ee_positions = np.zeros((NUM_SAMPLES, 3))
rng = np.random.default_rng(42)
for i in range(NUM_SAMPLES):
    q_sample = rng.uniform(lower, upper)
    pin.framesForwardKinematics(reduced_model, reduced_data, q_sample)
    ee_positions[i] = reduced_data.oMf[ee_id].translation

min_xyz = ee_positions.min(axis=0)
max_xyz = ee_positions.max(axis=0)
print("=== 工作空间位置范围（XYZ） ===")
print(f"  X in [{min_xyz[0]:.3f}, {max_xyz[0]:.3f}] m")
print(f"  Y in [{min_xyz[1]:.3f}, {max_xyz[1]:.3f}] m")
print(f"  Z in [{min_xyz[2]:.3f}, {max_xyz[2]:.3f}] m")
print()

# ====== 12. 绘制结果 ======
fig = plt.figure(figsize=(14, 7))

# 12.1 子图1：连杆初始姿态
ax1 = fig.add_subplot(121, projection="3d")
# 绘制所有关节点
ax1.scatter(joint_positions[:, 0],
            joint_positions[:, 1],
            joint_positions[:, 2],
            c='k', s=20, label='Joints')
# 绘制连杆：从每个关节到其父关节
for j in range(1, model.njoints):
    parent = model.parents[j]
    p0 = joint_positions[parent]
    p1 = joint_positions[j]
    ax1.plot([p0[0], p1[0]],
             [p0[1], p1[1]],
             [p0[2], p1[2]],
             c='gray')

# 高亮 base（根关节 idx=0）
ax1.scatter(0, 0, 0, c='red', s=50, label='Base (0)')
ax1.set_title("Full Model Initial Pose")
ax1.set_xlabel("X (m)")
ax1.set_ylabel("Y (m)")
ax1.set_zlabel("Z (m)")
ax1.legend()

# 12.2 子图2：工作空间 & 目标点
ax2 = fig.add_subplot(122, projection="3d")
ax2.scatter(ee_positions[:, 0],
            ee_positions[:, 1],
            ee_positions[:, 2],
            s=1, c="blue", alpha=0.3, label="Workspace Samples")
# 绘制目标点
ax2.scatter(TARGET_POS[0],
            TARGET_POS[1],
            TARGET_POS[2],
            c="blue",
            s=50,
            label="Target Point")
ax2.scatter(0,
            0,
            0,
            c="red",
            s=50,
            label="Target Point")
ax2.set_title("Estimated Workspace of Right Arm (7-DOF)")
ax2.set_xlabel("X (m)")
ax2.set_ylabel("Y (m)")
ax2.set_zlabel("Z (m)")
ax2.legend()

plt.tight_layout()
plt.show()