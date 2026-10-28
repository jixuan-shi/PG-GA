# ga_functions.py

import numpy as np
import math
# 新增导入：Numba用于JIT编译加速
from numba import jit
import numba

# -----------------------------------------------------------------------------
# TOP-LEVEL FUNCTION FOR PARALLEL EXECUTION
# -----------------------------------------------------------------------------
FITNESS_CACHE = {}


def parallel_fitness_calculation(args):
    individual, city_coordinates, depot_coordinates, ga_params = args

    # 将数组转换为可哈希的键（例如，字节串）
    perm_key = individual['permutation'].tobytes()
    tasks_key = individual['task_counts'].tobytes()
    chromosome_key = (perm_key, tasks_key)

    # 检查缓存
    if chromosome_key in FITNESS_CACHE:
        return FITNESS_CACHE[chromosome_key]

    # 如果不在缓存中，则计算
    fitness = calculate_fitness_minimize_max_workload(
        individual, city_coordinates, depot_coordinates, ga_params
    )[0]

    # 存入缓存
    FITNESS_CACHE[chromosome_key] = fitness
    return fitness

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS (for segmentation, repair, crossover, and 2-opt)
# -----------------------------------------------------------------------------

@jit(nopython=True)
def _calculate_dist(coord1, coord2):
    """Internal helper: Calculates Euclidean distance between two points (JIT compiled)."""
    return np.linalg.norm(coord1 - coord2)


@jit(nopython=True)
def _two_opt(route, city_coordinates, depot_coordinates):
    """
    Internal helper: Applies 2-Opt to a single route (JIT compiled).
    This version is optimized for Numba and has unified return types.
    """
    # +++ 新增：确保输入类型为int64 +++
    route_64 = route.astype(np.int64)

    num_cities_in_route = len(route_64)
    if num_cities_in_route < 2:
        return route_64  # <--- 修正1: 返回类型统一的route_64

    all_coords = np.zeros((city_coordinates.shape[0] + 1, 2))
    all_coords[0, :] = depot_coordinates
    all_coords[1:, :] = city_coordinates

    path_indices = np.zeros(num_cities_in_route + 2, dtype=np.int64)
    path_indices[0] = 0
    path_indices[-1] = 0
    path_indices[1:-1] = route_64 + 1

    improved = True
    while improved:
        improved = False
        for i in range(1, num_cities_in_route):
            for j in range(i + 1, num_cities_in_route + 1):
                p1_idx, p2_idx = path_indices[i - 1], path_indices[i]
                p3_idx, p4_idx = path_indices[j], path_indices[j + 1]

                current_dist = _calculate_dist(all_coords[p1_idx], all_coords[p2_idx]) + \
                               _calculate_dist(all_coords[p3_idx], all_coords[p4_idx])

                new_dist = _calculate_dist(all_coords[p1_idx], all_coords[p3_idx]) + \
                           _calculate_dist(all_coords[p2_idx], all_coords[p4_idx])

                if new_dist < current_dist - 1e-9:
                    segment_to_reverse = path_indices[i:j + 1].copy()
                    path_indices[i:j + 1] = np.flip(segment_to_reverse)
                    improved = True

    optimized_route = path_indices[1:-1] - 1
    # +++ 确保返回类型为int64 +++
    return optimized_route.astype(np.int64)  # <--- 修正2: 明确返回int64


def _split_ref_segments(task_counts, ref_sort_map):
    segments = [];
    current_pos = 0
    for count in task_counts:
        end_pos = current_pos + count
        segments.append(ref_sort_map[current_pos:min(end_pos, len(ref_sort_map))])
        current_pos = end_pos
    return segments


def _rebuild_permutation_by_ref(original_perm, task_counts, ref_sort_map):
    if np.sum(task_counts) != len(original_perm): return original_perm
    ref_segments = _split_ref_segments(task_counts, ref_sort_map)
    perm_lookup = {city: i for i, city in enumerate(original_perm)}
    new_perm_list = []
    for ref_cities in ref_segments:
        indices_in_orig = sorted([perm_lookup.get(c) for c in ref_cities if c is not None and c in perm_lookup])
        new_perm_list.extend([original_perm[i] for i in indices_in_orig])
    new_perm = np.array(new_perm_list, dtype=int)
    if len(new_perm) != len(original_perm) or len(np.unique(new_perm)) != len(original_perm): return original_perm
    return new_perm


def repair_task_counts(task_counts, ga_params):
    n_salesmen = ga_params.get('n_salesmen', 0);
    n_cities = ga_params.get('n_cities', 0);
    min_cities = ga_params.get('min_cities_per_salesman', 0)
    if n_salesmen == 0: return np.array([], dtype=int)
    if n_cities == 0: return np.zeros(n_salesmen, dtype=int)
    repaired_counts = np.array(task_counts, dtype=int).copy()
    max_iter = 2 * n_cities * n_salesmen + 10
    for _ in range(max_iter):
        v = False
        for i in range(n_salesmen):
            if repaired_counts[i] < min_cities:
                v = True
                for _ in range(min_cities - repaired_counts[i]):
                    d = np.where(repaired_counts > min_cities)[0]
                    if d.size > 0:
                        di = np.random.choice(d);
                        repaired_counts[di] -= 1;
                        repaired_counts[i] += 1
                    else:
                        repaired_counts[i] += 1
        s = n_cities - np.sum(repaired_counts)
        if s != 0:
            v = True
            if s > 0:
                for _ in range(s): repaired_counts[np.random.randint(0, n_salesmen)] += 1
            else:
                for _ in range(abs(s)):
                    e = np.where(repaired_counts > min_cities)[0]
                    if e.size == 0: e = np.where(repaired_counts > 0)[0]
                    if e.size > 0:
                        repaired_counts[np.random.choice(e)] -= 1
                    else:
                        break
        if not v and np.sum(repaired_counts) == n_cities and np.all(
            repaired_counts >= min_cities): return repaired_counts
    while np.sum(repaired_counts) != n_cities:
        d = n_cities - np.sum(repaired_counts);
        if d == 0: break
        if d > 0:
            repaired_counts[np.random.randint(0, n_salesmen)] += 1
        else:
            e = np.where(repaired_counts > 0)[0]
            if e.size > 0:
                repaired_counts[np.random.choice(e)] -= 1
            else:
                break
    return repaired_counts


def get_segment_boundaries(permutation, task_counts, ga_params):
    n_salesmen = ga_params.get('n_salesmen', 0);
    n_cities_in_perm = len(permutation)
    if n_salesmen == 0: return np.array([], dtype=int)
    current_task_counts = np.array(task_counts, dtype=int).copy()
    if current_task_counts.size == 0: current_task_counts = np.zeros(n_salesmen, dtype=int)
    if n_cities_in_perm > 0 and np.sum(current_task_counts) != n_cities_in_perm:
        while np.sum(current_task_counts) > n_cities_in_perm:
            idx_max = np.argmax(current_task_counts)
            if current_task_counts[idx_max] > 0:
                current_task_counts[idx_max] -= 1
            else:
                break
        while np.sum(current_task_counts) < n_cities_in_perm: current_task_counts[np.random.randint(0, n_salesmen)] += 1
    boundaries = np.cumsum(current_task_counts, dtype=int)
    if boundaries.size > 0 and n_cities_in_perm > 0:
        boundaries[-1] = n_cities_in_perm;
        boundaries = np.maximum(0, boundaries);
        boundaries = np.minimum(n_cities_in_perm, boundaries)
        for i in range(1, len(boundaries)): boundaries[i] = max(boundaries[i], boundaries[i - 1])
    elif n_cities_in_perm == 0 and n_salesmen > 0:
        boundaries = np.zeros(n_salesmen, dtype=int)
    return boundaries


def get_city_segments(individual, ga_params):
    n_salesmen = ga_params.get('n_salesmen', 0);
    permutation = individual.get('permutation', np.array([]));
    task_counts = individual.get('task_counts', np.array([]));
    n_cities_in_perm = len(permutation)
    if n_salesmen == 0: return []
    if n_cities_in_perm == 0: return [np.array([], dtype=int) for _ in range(n_salesmen)]
    boundary_params = {'n_salesmen': n_salesmen, 'n_cities': n_cities_in_perm, 'min_cities_per_salesman': 0}
    counts_for_boundaries = repair_task_counts(task_counts, boundary_params)
    boundaries = get_segment_boundaries(permutation, counts_for_boundaries, ga_params)
    segments = [];
    start_idx = 0
    for end_idx in boundaries: segments.append(np.array(permutation[start_idx:end_idx], dtype=int)); start_idx = end_idx
    while len(segments) < n_salesmen: segments.append(np.array([], dtype=int))
    return segments[:n_salesmen]


# -----------------------------------------------------------------------------
# CORE GA FUNCTIONS
# -----------------------------------------------------------------------------

def initialize_population(ga_params, city_angle_sort_map):
    population = [];
    population_size = ga_params['population_size'];
    n_cities = ga_params['n_cities'];
    n_salesmen = ga_params['n_salesmen'];
    min_cities_per_salesman = ga_params['min_cities_per_salesman']
    for i in range(population_size):
        permutation = np.array([], dtype=int);
        task_counts = np.array([], dtype=int)
        if n_cities > 0: permutation = city_angle_sort_map.copy() if i == 0 else np.random.permutation(n_cities)
        if n_salesmen > 0:
            if n_cities == 0:
                task_counts = np.zeros(n_salesmen, dtype=int)
            else:
                task_counts = np.full(n_salesmen, min_cities_per_salesman, dtype=int)
                remaining = n_cities - np.sum(task_counts)
                if remaining < 0: raise ValueError("Init error.")
                for _ in range(remaining): task_counts[np.random.randint(0, n_salesmen)] += 1
        population.append({'permutation': permutation, 'task_counts': task_counts})
    return population


def calculate_fitness_minimize_max_workload(individual, city_coordinates, depot_coordinates, ga_params):
    n_salesmen = ga_params['n_salesmen'];
    n_cities = ga_params['n_cities'];
    task_counts = np.array(individual.get('task_counts', []), dtype=int)
    inf = (float('inf'), float('inf'), float('inf'), np.full(n_salesmen, float('inf')))
    if n_salesmen == 0: return (0, 0, 0, np.array([])) if not np.any(task_counts > 0) else inf
    if task_counts.size != n_salesmen or (n_cities > 0 and np.sum(task_counts) != n_cities): return inf
    segments = get_city_segments(individual, ga_params);
    paths = np.zeros(n_salesmen)
    for s, indices in enumerate(segments):
        dist = 0.0
        if indices.size > 0:
            path_coords = np.vstack([depot_coordinates, city_coordinates[indices], depot_coordinates])
            dist = np.sum(np.linalg.norm(np.diff(path_coords, axis=0), axis=1))
        paths[s] = (dist / 10.0) + (len(indices) * 20.0)
    total = np.sum(paths);
    max_p = np.max(paths) if paths.size > 0 else 0
    penalty = np.std(task_counts) * ga_params['balance_weight_task_count']
    fit = (ga_params['weight_max_path_length'] * max_p) + (ga_params['weight_total_path_length'] * total) + penalty
    if math.isinf(fit) or math.isnan(fit): return inf
    return fit, max_p, total, paths


def tournament_selection(fitness_values, tournament_size):
    pop_size = len(fitness_values);
    if pop_size == 0: raise ValueError("Empty population.")
    actual_size = min(tournament_size, pop_size)
    contenders = np.random.choice(pop_size, actual_size, replace=False)
    winner_idx = np.argmin(fitness_values[contenders])
    return contenders[winner_idx]


def crossover_task_counts(parent1, parent2, ga_params, city_angle_sort_map):
    n_salesmen = ga_params['n_salesmen']
    if n_salesmen <= 1: return parent1['task_counts'].copy(), parent1['permutation'].copy()
    c1 = np.array(parent1['task_counts'], dtype=int);
    c2 = np.array(parent2['task_counts'], dtype=int);
    p1 = parent1['permutation']
    cp = np.random.randint(1, n_salesmen)
    child_tc = np.concatenate([c1[:cp], c2[cp:]])
    child_tc = repair_task_counts(child_tc, ga_params)
    child_p = _rebuild_permutation_by_ref(p1, child_tc, city_angle_sort_map)
    return child_tc, child_p


def crossover_permutation_ox1(parent1_ind, parent2_ind, child_task_counts, ga_params):
    p1 = parent1_ind['permutation'];
    p2 = parent2_ind['permutation']
    if ga_params['n_salesmen'] <= 1 or p1.size == 0: return p1.copy()
    temp_ind = {'permutation': p1, 'task_counts': child_task_counts}
    segments1 = get_city_segments(temp_ind, ga_params)
    p2_lookup = {city: i for i, city in enumerate(p2)}
    new_segments = []
    for seg in segments1:
        current_seg = list(seg)
        if len(current_seg) < 2: new_segments.append(np.array(current_seg)); continue
        start, end = sorted(np.random.choice(len(current_seg), 2, replace=False))
        sub_seg = current_seg[start:end + 1]
        pos_in_p2 = [p2_lookup.get(c) for c in sub_seg]
        if any(p is None for p in pos_in_p2): new_segments.append(np.array(current_seg)); continue
        reordered = [c for _, c in sorted(zip(pos_in_p2, sub_seg))]
        current_seg[start:end + 1] = reordered
        new_segments.append(np.array(current_seg))
    child_perm = np.concatenate(new_segments)
    if len(child_perm) != ga_params['n_cities'] or len(np.unique(child_perm)) != ga_params['n_cities']: return p1.copy()
    return child_perm


def mutate_task_counts_special(individual, ga_params, city_angle_sort_map):
    if np.random.rand() >= ga_params['mutation_rate_tasks'] or ga_params['n_salesmen'] <= 1 or ga_params[
        'n_cities'] == 0: return individual
    original = individual;
    tc = original['task_counts'].copy()
    idx = np.random.randint(0, ga_params['n_salesmen']);
    action = np.random.randint(0, 2)
    new_tc = tc.copy();
    mutated = False
    if action == 1:
        max_p = ga_params['n_cities'] - (ga_params['n_salesmen'] - 1) * ga_params['min_cities_per_salesman']
        if tc[idx] < max_p:
            donors = np.where(tc > ga_params['min_cities_per_salesman'])[0];
            donors = donors[donors != idx]
            if donors.size > 0:
                donor_idx = np.random.choice(donors);
                new_tc[idx] += 1;
                new_tc[donor_idx] -= 1;
                mutated = True
    else:
        if tc[idx] > ga_params['min_cities_per_salesman']:
            receivers = np.arange(ga_params['n_salesmen']);
            receivers = receivers[receivers != idx]
            if receivers.size > 0:
                receiver_idx = np.random.choice(receivers);
                new_tc[idx] -= 1;
                new_tc[receiver_idx] += 1;
                mutated = True
    if not mutated: return original
    new_p = _rebuild_permutation_by_ref(original['permutation'], new_tc, city_angle_sort_map)
    if len(new_p) != ga_params['n_cities'] or len(np.unique(new_p)) != ga_params['n_cities']: return original
    final_tc = repair_task_counts(new_tc, ga_params)
    if np.sum(final_tc) != ga_params['n_cities']: return original
    final_p = _rebuild_permutation_by_ref(original['permutation'], final_tc, city_angle_sort_map) if not np.array_equal(
        new_tc, final_tc) else new_p
    if len(final_p) != ga_params['n_cities']: return original
    return {'permutation': final_p, 'task_counts': final_tc}


def mutate_permutation_segmented(individual, ga_params):
    """
    对每个销售员的路径段应用更强大的变异算子。
    会随机选择“反转变异”或“乱序变异”。
    """
    permutation = individual['permutation'].copy()
    task_counts = individual['task_counts']

    if permutation.size < 3 or ga_params.get('n_salesmen', 0) == 0:
        return permutation

    # 使用您原有的函数获取每个销售员路径的边界
    temp_params = {'n_salesmen': ga_params['n_salesmen'], 'n_cities': len(permutation)}
    boundaries = get_segment_boundaries(permutation, task_counts, temp_params)

    if boundaries.size == 0 and ga_params.get('n_salesmen', 0) > 0:
        return permutation

    start_idx = 0
    for end_idx in boundaries:
        seg_len = end_idx - start_idx

        # 确保段内至少有3个城市，这样变异才有意义
        if seg_len > 2 and np.random.rand() < ga_params['mutation_rate_perm']:

            # --- 1. 在当前段内选择一个子段进行操作 ---
            # 随机选择两个不同的切割点
            cut1, cut2 = np.random.choice(seg_len, 2, replace=False)
            start_cut, end_cut = min(cut1, cut2), max(cut1, cut2)

            # 转换为在完整 permutation 数组中的绝对索引
            abs_start = start_idx + start_cut
            abs_end = start_idx + end_cut

            # --- 2. 随机选择一种变异类型来应用 ---
            mutation_type = np.random.rand()

            if mutation_type < 0.7:  # 70% 的概率执行反转变异 (更倾向于局部优化)
                # --- 反转变异 (Inversion Mutation) ---
                # 提取子段并将其反转
                sub_segment = permutation[abs_start: abs_end + 1]
                permutation[abs_start: abs_end + 1] = sub_segment[::-1]

            else:  # 30% 的概率执行乱序变异 (更倾向于增加多样性)
                # --- 乱序变异 (Scramble Mutation) ---
                # 提取子段并将其内部顺序打乱
                sub_segment = permutation[abs_start: abs_end + 1]
                np.random.shuffle(sub_segment)
                permutation[abs_start: abs_end + 1] = sub_segment

        start_idx = end_idx

    return permutation

def apply_2opt_to_individual(individual, city_coordinates, depot_coordinates, ga_params):
    """
    Applies 2-Opt local search to all paths of an individual.
    """
    if ga_params['n_cities'] < 2 or ga_params['n_salesmen'] < 1:
        return individual

    city_segments = get_city_segments(individual, ga_params)
    optimized_segments = []

    for route in city_segments:
        if len(route) > 1:
            optimized_route = _two_opt(route, city_coordinates, depot_coordinates)
            optimized_segments.append(optimized_route)
        else:
            optimized_segments.append(route)

    new_permutation = np.concatenate(optimized_segments) if optimized_segments else np.array([], dtype=int)

    return {'permutation': new_permutation, 'task_counts': individual['task_counts']}


def calculate_fitness_from_cost_matrix(individual, cost_matrix, depot_distances, ga_params):
    """
    计算个体的适应度，但路径成本从一个预先计算好的成本矩阵中查找。
    这对于迭代优化至关重要。
    """
    n_salesmen = ga_params['n_salesmen']
    n_cities = ga_params['n_cities']
    task_counts = np.array(individual.get('task_counts', []), dtype=int)

    inf_return = (float('inf'), float('inf'), float('inf'), np.full(n_salesmen, float('inf')))

    # --- 有效性检查 ---
    if task_counts.size != n_salesmen or (n_cities > 0 and np.sum(task_counts) != n_cities):
        return inf_return

    city_segments = get_city_segments(individual, ga_params)
    salesman_path_lengths = np.zeros(n_salesmen)

    for s, segment_indices in enumerate(city_segments):
        current_path_cost = 0.0
        if segment_indices.size > 0:
            # 成本：从Depot到第一个城市
            current_path_cost += depot_distances[segment_indices[0]]

            # 成本：在城市之间
            for i in range(len(segment_indices) - 1):
                start_city_idx = segment_indices[i]
                end_city_idx = segment_indices[i + 1]
                current_path_cost += cost_matrix[start_city_idx, end_city_idx]

            # 成本：从最后一个城市返回Depot
            current_path_cost += depot_distances[segment_indices[-1]]

        salesman_path_lengths[s] = current_path_cost

    total_distance = np.sum(salesman_path_lengths)
    max_salesman_path = np.max(salesman_path_lengths) if salesman_path_lengths.size > 0 else 0

    penalty_task_count_std = np.std(task_counts) if ga_params.get('balance_weight_task_count',
                                                                  0) > 0 and n_salesmen > 0 else 0

    fitness = (ga_params['weight_max_path_length'] * max_salesman_path) + \
              (ga_params['weight_total_path_length'] * total_distance) + \
              (ga_params['balance_weight_task_count'] * penalty_task_count_std)

    if math.isinf(fitness) or math.isnan(fitness):
        return inf_return

    return fitness, max_salesman_path, total_distance, salesman_path_lengths
