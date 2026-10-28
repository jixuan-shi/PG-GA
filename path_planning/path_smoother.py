# path_smoother.py

import numpy as np


def _is_collision_free(point1, point2, obstacle_map, delta=1.0):
    """
    检查两点之间的直线路径是否穿过障碍物。
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


def smooth_path(original_path, obstacle_map):
    """
    对原始路径进行贪心剪枝，以获得更短、更平滑的路径。
    """
    if not original_path or len(original_path) < 2:
        return original_path

    # --- REMOVED: print statement moved to main.py for better reporting ---
    # print("\n--- 开始路径剪枝/平滑 ---") # <-- This line is removed

    smoothed_path = [original_path[0]]
    current_index = 0

    while current_index < len(original_path) - 1:
        lookahead_index = len(original_path) - 1

        shortcut_found = False
        while lookahead_index > current_index:
            point_a = original_path[current_index]
            point_b = original_path[lookahead_index]

            if _is_collision_free(point_a, point_b, obstacle_map):
                smoothed_path.append(point_b)
                current_index = lookahead_index
                shortcut_found = True
                break
            else:
                lookahead_index -= 1

        if not shortcut_found:
            smoothed_path.append(original_path[current_index + 1])
            current_index += 1

    # --- REMOVED: print statement moved to main.py ---
    # print(f"路径剪枝完成：...") # <-- This line is removed
    return smoothed_path