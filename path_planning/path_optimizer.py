# path_optimizer.py

import numpy as np
import math
import random

def _is_collision_free(point1, point2, obstacle_map, delta=1.0):
    """
    检查两点之间的直线路径是否穿过障碍物。
    (This is a utility function, same as in path_smoother.py)
    """
    p1 = np.array(point1, dtype=float)
    p2 = np.array(point2, dtype=float)
    v = p2 - p1
    distance = np.linalg.norm(v)
    if distance < 1e-6:
        return True
    u = v / distance
    num_steps = int(distance / delta)
    current_pos = p1.copy()
    for _ in range(num_steps):
        current_pos += delta * u
        x_idx, y_idx = int(current_pos[0]), int(current_pos[1])
        map_height, map_width = obstacle_map.shape
        if not (0 <= x_idx < map_width and 0 <= y_idx < map_height):
            return False
        if obstacle_map[y_idx, x_idx] >= 60:
            return False
    return True


def _sample_point_in_ellipse(foci1, foci2, max_len):
    """
    在一个由两个焦点和最大路径长度定义的椭圆内随机采样一个点。
    """
    # 椭圆参数计算
    f1, f2 = np.array(foci1), np.array(foci2)
    dist_foci = np.linalg.norm(f1 - f2)

    # 椭圆长轴 a 和短轴 b
    c = dist_foci / 2.0
    a = max_len / 2.0

    # 如果焦点间距离大于最大长度，椭圆不存在，无法采样
    if a <= c:
        return None

    b = math.sqrt(a ** 2 - c ** 2)

    # 椭圆中心和旋转角度
    center = (f1 + f2) / 2.0
    angle = math.atan2(f2[1] - f1[1], f2[0] - f1[0])

    # 在一个单位圆内生成一个均匀分布的随机点
    # 使用 sqrt(random) 来确保在椭圆面积上是均匀的
    r = math.sqrt(random.random())
    phi = random.uniform(0, 2 * math.pi)

    # 将单位圆上的点映射到未旋转的椭圆上
    x_ellipse = a * r * math.cos(phi)
    y_ellipse = b * r * math.sin(phi)

    # 旋转并平移该点到最终位置
    x_rotated = x_ellipse * math.cos(angle) - y_ellipse * math.sin(angle)
    y_rotated = x_ellipse * math.sin(angle) + y_ellipse * math.cos(angle)

    final_point = (center[0] + x_rotated, center[1] + y_rotated)
    return final_point


def optimize_path_elliptical(path, obstacle_map, num_samples=1000, max_passes=1, convergence_threshold=-1):
    """
    使用椭圆采样区域对路径进行精细优化。

    参数:
    - path (list): 经过初步平滑的路径。
    - obstacle_map (np.array): 地形图。
    - num_samples (int): 每个拐点在椭圆区域内的采样次数。
    - max_passes (int): 对整条路径进行优化的最大轮数。
    - convergence_threshold (float): 当一轮优化后路径长度减少小于此值时，认为算法收敛。

    返回:
    - list: 最终优化后的路径。
    """
    min = float('inf')
    new = float('inf')

    if len(path) < 3:
        return path

    optimized_path = list(path)  # 创建一个可修改的副本

    for pass_num in range(max_passes):
        path_changed = False

        # 使用一个“滑动窗口”遍历 (P_{i-1}, P_i, P_{i+1})
        for i in range(1, len(optimized_path) - 1):
            p_prev = optimized_path[i - 1]
            p_curr = optimized_path[i]
            p_next = optimized_path[i + 1]

            # 计算原始路径段的成本（长度）
            original_cost = np.linalg.norm(np.array(p_curr) - np.array(p_prev)) + \
                            np.linalg.norm(np.array(p_next) - np.array(p_curr))

            min_cost = original_cost
            best_point = p_curr

            # 在以 p_prev 和 p_next 为焦点的椭圆内采样
            for _ in range(num_samples):
                # 采样一个候选点
                # 椭圆长轴长度等于原始路径段长度，确保任何内部点都会使路径更短

                sample = _sample_point_in_ellipse(p_prev, p_next, original_cost)
                if sample is None:  # 如果椭圆无法形成，则跳过
                    continue

                # a. 碰撞检测
                if _is_collision_free(p_prev, sample, obstacle_map) and \
                        _is_collision_free(sample, p_next, obstacle_map):

                    # b. 如果无碰撞，计算新成本
                    new_cost = np.linalg.norm(np.array(sample) - np.array(p_prev)) + \
                               np.linalg.norm(np.array(p_next) - np.array(sample))

                    min = min_cost
                    new = new_cost
                    # 寻找最优替代点

                    if new_cost < min_cost:
                        print(f"new_cost:{new_cost}, min_cost:{min_cost}")
                        min_cost = new_cost
                        best_point = sample
                        path_changed = True

            # 节点替换决策
            optimized_path[i] = best_point

        # 全路径迭代收敛检查
        #if new - min < convergence_threshold:
        #    print(f"椭圆优化在第 {pass_num + 1} 轮收敛。")
        #    break

    return optimized_path