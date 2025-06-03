#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pinocchio as pin
import numpy as np

def setup_reduced_model(urdf_path):
    model_full = pin.buildModelFromUrdf(urdf_path)
    data_full = model_full.createData()

    fixed_joint_ids_to_lock = []

    if hasattr(model_full, 'njoints'):
        num_joints_total = model_full.njoints
        for i in range(1, 21):  # 迭代潜在的关节ID 1 到 20
            if i < num_joints_total:  # 检查此关节ID是否有效
                fixed_joint_ids_to_lock.append(i)
            else:
                break
    q_full_init = pin.neutral(model_full)
    reduced_model = pin.buildReducedModel(model_full, fixed_joint_ids_to_lock, q_full_init)
    reduced_data = reduced_model.createData()
    reduced_q_init = pin.neutral(reduced_model)
    pin.normalize(reduced_model, reduced_q_init)

    ee_frame_id = reduced_model.getFrameId("right_wrist_yaw_link")
    return reduced_model, reduced_data, ee_frame_id, reduced_q_init

# def solve_right_arm_ik(target_pos, q_init_7dof,
#                        base_SE3=None,
#                        it_max=100, eps=1e-4, damp=1e-12):
   
#     # 默认旋转为单位
#     target_rot = np.eye(3)
#     # URDF 路径硬编码（根据需要修改）
#     urdf_path = "/home/tzh/OpenHomie/HomieRL/legged_gym/resources/robots/g1_description/g1.urdf"
#     # 构建 reduced 模型
#     reduced_model, reduced_data, ee_frame_id, _ = setup_reduced_model(urdf_path)
#     # 初始 q
#     q = np.array(q_init_7dof, dtype=float).copy()
#     pin.normalize(reduced_model, q)
#     if base_SE3 is not None:
#         target_SE3_world = pin.SE3(target_rot, target_pos)
#         target_SE3 = base_SE3.inverse() * target_SE3_world
#     else:
#         target_SE3 = pin.SE3(target_rot, target_pos)
#     # IK 迭代
#     final_i = 0
#     for i in range(1, it_max + 1):
#         final_i = i
#         pin.framesForwardKinematics(reduced_model, reduced_data, q)
#         oMf = reduced_data.oMf[ee_frame_id]
#         iMd = oMf.actInv(target_SE3)
#         err = pin.log(iMd).vector
#         err_norm = float(np.linalg.norm(err))
#         if err_norm < eps:
#             break
#         J6 = pin.computeFrameJacobian(reduced_model, reduced_data, q,
#                                       ee_frame_id, pin.LOCAL)
#         Jlog = pin.Jlog6(iMd.inverse())
#         J = -Jlog.dot(J6)
#         JJt = J.dot(J.T) + damp * np.eye(6)
#         dq = -J.T.dot(np.linalg.solve(JJt, err))
#         q = pin.integrate(reduced_model, q, dq)
#         q[:] = np.minimum(np.maximum(q,
#                                      reduced_model.lowerPositionLimit),
#                            reduced_model.upperPositionLimit)
#         pin.normalize(reduced_model, q)
#         convergence = (err_norm < eps)
#     return convergence, q, err_norm, final_i, reduced_model.lowerPositionLimit, reduced_model.upperPositionLimit

def solve_right_arm_ik(target_pos, q_init_7dof,
                                     base_SE3=None,
                                     it_max=100, eps=1e-4, damp=1e-12):
    # 1. 载入并构建 reduced model
    urdf_path = "/home/tzh/OpenHomie/HomieRL/legged_gym/resources/robots/g1_description/g1.urdf"
    reduced_model, reduced_data, ee_frame_id, _ = setup_reduced_model(urdf_path)

    # 2. 初始关节角
    q = np.array(q_init_7dof, dtype=float).copy()
    pin.normalize(reduced_model, q)

    # 3. 计算「在肩部系下，末端目标的平移」target_p_local
    if base_SE3 is not None:
        # base_SE3：肩部→世界的变换
        # world 里末端目标是 (I, target_pos)，先算成肩部::local
        tmp = base_SE3.inverse() * pin.SE3(np.eye(3), target_pos)
        target_p_local = tmp.translation   # (3,) 向量
    else:
        target_p_local = target_pos.copy()   # 已经是 local 坐标

    final_i = 0
    for i in range(1, it_max + 1):
        final_i = i

        # 3.1 正向运动学，求出末端在肩部系下的当前位置
        pin.framesForwardKinematics(reduced_model, reduced_data, q)
        oMf = reduced_data.oMf[ee_frame_id]
        current_p_local = oMf.translation   # (3,)

        # 3.2 只算平移误差
        err_trans = current_p_local - target_p_local
        err_norm = np.linalg.norm(err_trans)
        if err_norm < eps:
            break

        # 3.3 计算雅可比，只要前三行（平移部分）
        J6 = pin.computeFrameJacobian(reduced_model, reduced_data, q,
                                      ee_frame_id, pin.LOCAL)
        Jpos = J6[:3, :]   # (3 × 7)

        # 3.4 阻尼最小二乘求解 dq
        JJt = Jpos.dot(Jpos.T) + damp * np.eye(3)    # (3×3)
        dq = -Jpos.T.dot(np.linalg.solve(JJt, err_trans))  # (7,)

        # 3.5 更新 q，并裁剪到关节极限
        q = pin.integrate(reduced_model, q, dq)
        q[:] = np.minimum(np.maximum(q,
                                     reduced_model.lowerPositionLimit),
                          reduced_model.upperPositionLimit)
        pin.normalize(reduced_model, q)

    # 最后返回
    convergence = (err_norm < eps)
    return convergence, q, err_norm, final_i, reduced_model.lowerPositionLimit, reduced_model.upperPositionLimit


if __name__ == "__main__":
    from pinocchio import SE3
    # 假设已有底座在世界系下的位置
    # base_pos = np.array([1.0, 0.5, 0.0])  # 仅示例
    # base_rot = np.eye(3)
    # base_SE3 = SE3(base_rot, base_pos)
    
    base_rot = np.array([
        [0.99757,    0.0621323,  0.0315211],
        [-0.0688848, 0.947359,   0.312675],
        [-0.0104346, -0.314087,  0.949337]
    ])
    
    base_pos = np.array([-0.109501, -0.0602609, 1.05168])
    base_SE3 = SE3(base_rot, base_pos)

    target_pos_world = np.array([0.1, -0.25, 1.3]) # 可以尝试修改这个目标点
    # target_pos_world = np.array([2.0, 2.0, 2.0]) # 一个可能无效的目标点

    q_init_7dof = np.zeros(7)
    
    print(f"尝试到达目标点: {target_pos_world}")
    
    _, q_sol, final_error, iterations, lower_limits, upper_limits = solve_right_arm_ik(
                               target_pos_world, q_init_7dof,
                               base_SE3=base_SE3,
                               it_max=100, eps=1e-4, damp=1e-12)
    
    print(f"求解的7DOF关节角（rad）：{q_sol}")
    print(f"最终误差 (err_norm): {final_error:.6f}")
    print(f"迭代次数: {iterations}")

    if iterations >= 100 and final_error > 1e-4 : # it_max = 100, eps = 1e-4
        print("\\n警告: IK求解可能未完全收敛。")
    else:
        print("\\nIK求解成功收敛。")

    # 检查是否有多个关节达到了其极限
    at_lower_limit = np.isclose(q_sol, lower_limits)
    at_upper_limit = np.isclose(q_sol, upper_limits)
    
    num_at_limits = np.sum(at_lower_limit) + np.sum(at_upper_limit)
    if num_at_limits > 0:
        print(f"有 {num_at_limits} 个关节达到了其运动极限:")
        for i in range(len(q_sol)):
            if at_lower_limit[i]:
                print(f"  关节 {i} 达到了下限: {q_sol[i]:.4f} (限制: {lower_limits[i]:.4f})")
            if at_upper_limit[i]:
                print(f"  关节 {i} 达到了上限: {q_sol[i]:.4f} (限制: {upper_limits[i]:.4f})")
        if num_at_limits > len(q_sol) / 2: # 例如，如果超过一半关节在极限位置
             print("较多关节达到极限，目标点可能在工作空间边界。")
