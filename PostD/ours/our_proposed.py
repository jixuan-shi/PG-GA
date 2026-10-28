import random
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sys
import time
import itertools
from scipy.ndimage import binary_dilation
from multiprocessing import Pool, cpu_count, Array, JoinableQueue, Process, Lock, Manager, current_process
import ctypes
from tqdm import tqdm
import math
import os
from matplotlib.lines import Line2D

# --- 模块导入 ---
from map.map_processor import process_tif_map
from path_planning.rrt_vision import RRT
from path_planning.path_smoother import smooth_path
from path_planning.path_optimizer import optimize_path_elliptical
from mtsp.ga_functions import (
    initialize_population, tournament_selection, crossover_task_counts,
    crossover_permutation_ox1, mutate_task_counts_special,
    mutate_permutation_segmented, get_city_segments
)

# ----------------------------------------------------------------------------------
# 全局变量
# ----------------------------------------------------------------------------------
ga_worker_matrix_ptr, ga_worker_params = None, None
init_worker_points = None


def init_worker_seed(base_seed):
    seed = 41
    random.seed(seed)
    np.random.seed(seed)


# ----------------------------------------------------------------------------------
# 适配RRT成本矩阵的2-Opt局部搜索函数
# ----------------------------------------------------------------------------------
def _two_opt_rrt(route, cost_matrix):
    n_points = len(route)
    if n_points < 4:
        return route
    best_route = list(route)
    improved = True
    while improved:
        improved = False
        for i in range(1, n_points - 2):
            for j in range(i + 1, n_points):
                if j == i + 1: continue
                current_cost = cost_matrix[best_route[i - 1], best_route[i]] + cost_matrix[
                    best_route[j - 1], best_route[j]]
                new_cost = cost_matrix[best_route[i - 1], best_route[j - 1]] + cost_matrix[best_route[i], best_route[j]]
                if new_cost < current_cost:
                    best_route[i:j] = best_route[j - 1:i - 1:-1]
                    improved = True
    return best_route


def apply_2opt_to_individual_rrt(individual, cost_matrix, ga_params):
    n_salesmen = ga_params['n_salesmen']
    task_counts = individual['task_counts']
    permutation = individual['permutation']
    new_permutation = []
    current_pos = 0
    for s in range(n_salesmen):
        count = task_counts[s]
        if count > 1:
            segment_indices = permutation[current_pos: current_pos + count]
            route_indices = [0] + (segment_indices + 1).tolist()
            optimized_route_indices = _two_opt_rrt(route_indices, cost_matrix)
            optimized_segment = [idx - 1 for idx in optimized_route_indices[1:]]
            new_permutation.extend(optimized_segment)
        elif count == 1:
            new_permutation.extend(permutation[current_pos: current_pos + 1])
        current_pos += count
    individual['permutation'] = np.array(new_permutation)
    return individual


# ----------------------------------------------------------------------------------
# 辅助函数
# ----------------------------------------------------------------------------------
def calculate_path_length(path):
    if not path or len(path) < 2: return 0.0
    return sum(np.linalg.norm(np.array(path[i + 1]) - np.array(path[i])) for i in range(len(path) - 1))


def plan_and_optimize_segment(start_coords, goal_coords, obstacle_map, rrt_params):
    rrt = RRT(start=start_coords, goal=goal_coords, obstacle_map=obstacle_map, **rrt_params['rrt_base'])
    path_raw = rrt.plan()
    if not path_raw: return None
    path_smooth = smooth_path(path_raw, obstacle_map)
    path_optimized = optimize_path_elliptical(path_smooth, obstacle_map, **rrt_params['optimizer'])
    return path_optimized


# ----------------------------------------------------------------------------------
# 并行工作函数定义
# ----------------------------------------------------------------------------------
def init_ga_worker(stable_matrix_ptr, ga_params, base_seed):
    global ga_worker_matrix_ptr, ga_worker_params
    ga_worker_matrix_ptr = stable_matrix_ptr
    ga_worker_params = ga_params
    init_worker_seed(base_seed)


def parallel_fitness_calculation(individual, cost_matrix_local=None, ga_params_local=None):
    if cost_matrix_local is not None and ga_params_local is not None:
        cost_matrix = cost_matrix_local
        ga_params = ga_params_local
    else:
        n_total_points = ga_worker_params['n_cities'] + 1
        cost_matrix = np.frombuffer(ga_worker_matrix_ptr.get_obj(), dtype=ctypes.c_double).reshape(
            (n_total_points, n_total_points))
        ga_params = ga_worker_params
    n_salesmen = ga_params['n_salesmen']
    n_cities = ga_params['n_cities']
    task_counts = np.array(individual.get('task_counts', []), dtype=int)
    permutation = individual.get('permutation', np.array([]))
    inf_return = (float('inf'), float('inf'), float('inf'), np.full(n_salesmen, float('inf')))
    if n_salesmen == 0:
        return (0, 0, 0, np.array([])) if n_cities == 0 else inf_return
    if task_counts.size != n_salesmen or (n_cities > 0 and np.sum(task_counts) != n_cities):
        return inf_return
    workloads = np.zeros(n_salesmen)
    current_pos = 0
    for s in range(n_salesmen):
        count = task_counts[s]
        segment_indices = permutation[current_pos: current_pos + count]
        current_pos += count
        path_dist = 0.0
        if count > 0:
            path_indices = [0] + (segment_indices + 1).tolist() + [0]
            for k in range(len(path_indices) - 1):
                path_dist += cost_matrix[path_indices[k], path_indices[k + 1]]
        task_cost = count * 20.0
        workloads[s] = (path_dist / 10.0) + task_cost
    max_workload = np.max(workloads) if workloads.size > 0 else 0
    total_workload = np.sum(workloads)
    balance_penalty = np.std(task_counts) * ga_params.get('balance_weight_task_count', 0.0)
    fitness = (ga_params.get('weight_max_path_length', 1.0) * max_workload +
               ga_params.get('weight_total_path_length', 0.0) * total_workload +
               balance_penalty)
    if math.isinf(fitness) or math.isnan(fitness):
        return inf_return
    return fitness, max_workload, total_workload, workloads


def rrt_cost_updater_seeded(args):
    task_queue, live_matrix_ptr, shared_path_cache, all_points, obstacle_map, rrt_params, base_seed = args
    init_worker_seed(base_seed)
    rrt_cost_updater(task_queue, live_matrix_ptr, shared_path_cache, all_points, obstacle_map, rrt_params)


def rrt_cost_updater(task_queue, live_matrix_ptr, shared_path_cache, all_points, obstacle_map, rrt_params):
    n_total_points = len(all_points)
    live_matrix = np.frombuffer(live_matrix_ptr.get_obj(), dtype=ctypes.c_double).reshape(
        (n_total_points, n_total_points))
    while True:
        task = task_queue.get()
        if task is None:
            task_queue.task_done()
            break
        i, j = task
        start_coords, goal_coords = all_points[i], all_points[j]
        optimized_path = plan_and_optimize_segment(start_coords, goal_coords, obstacle_map, rrt_params)
        if optimized_path:
            path_len = calculate_path_length(optimized_path)
            live_matrix[i, j] = path_len
            live_matrix[j, i] = path_len
            cache_key = tuple(sorted((i, j)))
            shared_path_cache[cache_key] = optimized_path
        else:
            live_matrix[i, j] = float('inf')
            live_matrix[j, i] = float('inf')
        task_queue.task_done()


def init_euclidean_worker(points, base_seed):
    global init_worker_points
    init_worker_points = points
    init_worker_seed(base_seed)


def calculate_euclidean_row_worker(i):
    point_i = init_worker_points[i]
    row = np.linalg.norm(init_worker_points - point_i, axis=1)
    return i, row


# ----------------------------------------------------------------------------------
# 主执行逻辑
# ----------------------------------------------------------------------------------
def main():
    # 主随机种子
    MAIN_SEED = 41
    random.seed(MAIN_SEED)
    np.random.seed(MAIN_SEED)

    # --- 1. 配置参数 ---
    TIF_FILEPATH = r"city2.tif"
    NUM_TASK_POINTS = 100
    DEPOT_LOCATION = (1000, 1000)
    NUM_AGENTS = 3
    INFLATION_RADIUS = 8
    SWAP_INTERVAL = 50

    RRT_PARAMS = {
        'rrt_base': {'iter_max': 8000, 'min_step_size': 10, 'max_step_size': 50, 'influence_radius': 100},
        'optimizer': {'num_samples': 1000, 'max_passes': 1}
    }
    GA_PARAMS = {
        'n_salesmen': NUM_AGENTS, 'population_size': 50, 'n_generations': 10000,
        'tournament_size': 3, 'crossover_rate_tasks': 0.2, 'crossover_rate_perm': 0.2,
        'mutation_rate_tasks': 0.3, 'mutation_rate_perm': 0.35, 'min_cities_per_salesman': 2,

        # [修改] 开启全局开关，但循环中只对精英使用
        'use_2opt': True,

        'weight_max_path_length': 1.0, 'weight_total_path_length': 0.01,
        'balance_weight_task_count': 0.0,
    }

    # --- 2. 地图处理 ---
    map_data_tuple = process_tif_map(TIF_FILEPATH)
    if map_data_tuple is None: sys.exit("地图处理失败。")
    terrain_map, cmap, norm, boundaries, map_width, map_height = map_data_tuple
    base_obstacle_map = terrain_map.copy()
    base_obstacle_map[binary_dilation(terrain_map >= 60, iterations=INFLATION_RADIUS)] = 60
    if base_obstacle_map[int(DEPOT_LOCATION[1]), int(DEPOT_LOCATION[0])] >= 60:
        print(f"错误：仓库位置 {DEPOT_LOCATION} 位于障碍物内，请重新指定！")
        sys.exit("仓库位置无效。")

    print(f"正在随机生成 {NUM_TASK_POINTS} 个任务点...")
    generated_points = [DEPOT_LOCATION]
    while len(generated_points) < NUM_TASK_POINTS + 1:
        x = np.random.randint(0, map_width)
        y = np.random.randint(0, map_height)
        if base_obstacle_map[y, x] < 60:
            generated_points.append((x, y))
    TASK_POINTS = generated_points
    TASK_POINTS_NP = np.array(TASK_POINTS)
    n_total_points = len(TASK_POINTS)
    GA_PARAMS['n_cities'] = n_total_points - 1

    # --- 3. 多进程工具初始化 ---
    manager = Manager()
    lock = Lock()
    task_queue = JoinableQueue()
    shared_path_cache = manager.dict()
    published_rrt_tasks = manager.dict()
    stable_arr_base = Array(ctypes.c_double, n_total_points * n_total_points)
    stable_matrix_view = np.frombuffer(stable_arr_base.get_obj(), dtype=ctypes.c_double).reshape(
        (n_total_points, n_total_points))
    live_arr_base = Array(ctypes.c_double, n_total_points * n_total_points)
    live_matrix_view = np.frombuffer(live_arr_base.get_obj(), dtype=ctypes.c_double).reshape(
        (n_total_points, n_total_points))

    # 启动后台RRT消费者进程
    num_consumers = cpu_count() - 2 if cpu_count() > 2 else 1
    consumer_args = (
        task_queue, live_arr_base, shared_path_cache, TASK_POINTS_NP, base_obstacle_map, RRT_PARAMS, MAIN_SEED)
    consumers = [Process(target=rrt_cost_updater_seeded, args=(consumer_args,)) for _ in range(num_consumers)]
    for p in consumers:
        p.daemon = True
        p.start()

    start_time_total = time.time()

    # 计算初始欧几里得成本矩阵
    print("正在快速计算初始欧几里得距离成本矩阵...")
    with Pool(processes=cpu_count(), initializer=init_euclidean_worker, initargs=(TASK_POINTS_NP, MAIN_SEED)) as pool:
        for i, row in tqdm(pool.imap_unordered(calculate_euclidean_row_worker, range(n_total_points)),
                           total=n_total_points, desc="Initializing Matrix"):
            live_matrix_view[i, :] = row
    with lock:
        np.copyto(stable_matrix_view, live_matrix_view)

    # --- 6. 异步遗传算法主循环 ---
    city_coords = TASK_POINTS_NP[1:]
    depot_coords = TASK_POINTS_NP[0]
    city_angles = np.arctan2(city_coords[:, 1] - depot_coords[1], city_coords[:, 0] - depot_coords[0])
    city_angle_sort_map = np.argsort(city_angles)
    population = initialize_population(GA_PARAMS, city_angle_sort_map)

    path_discovery_history = []

    # 初始 2-Opt
    if GA_PARAMS.get('use_2opt', False):
        population = [apply_2opt_to_individual_rrt(ind, stable_matrix_view, GA_PARAMS) for ind in
                      tqdm(population, desc="Initial 2-Opt")]

    initargs_ga = (stable_arr_base, GA_PARAMS, MAIN_SEED)
    with Pool(processes=cpu_count(), initializer=init_ga_worker, initargs=initargs_ga) as pool:
        gen_iterator = tqdm(range(GA_PARAMS['n_generations']), desc="Generations")
        for gen in gen_iterator:
            if gen > 0 and gen % SWAP_INTERVAL == 0:
                with lock:
                    np.copyto(stable_matrix_view, live_matrix_view)

            fitness_results_list = pool.map(parallel_fitness_calculation, population)
            fitness_values = np.array([res[0] for res in fitness_results_list])

            elite_idx = np.argmin(fitness_values)
            elite_individual = population[elite_idx]

            # --- 精英策略：仅对精英个体执行 2-Opt ---
            if GA_PARAMS.get('use_2opt', False):
                elite_individual = apply_2opt_to_individual_rrt(elite_individual, stable_matrix_view, GA_PARAMS)
                population[elite_idx] = elite_individual

            current_pos = 0
            gen_new_paths = 0
            for count in elite_individual['task_counts']:
                if count > 0:
                    perm_indices = elite_individual['permutation'][current_pos:current_pos + count]
                    route_point_indices = [0] + (perm_indices + 1).tolist() + [0]
                    for i in range(len(route_point_indices) - 1):
                        start_node, end_node = route_point_indices[i], route_point_indices[i + 1]
                        task_key = tuple(sorted((start_node, end_node)))
                        if task_key not in published_rrt_tasks:
                            task_queue.put(task_key)
                            published_rrt_tasks[task_key] = True
                            gen_new_paths += 1
                current_pos += count
            path_discovery_history.append(gen_new_paths)

            new_population = [elite_individual.copy()]

            for _ in range(1, GA_PARAMS['population_size']):
                p1_idx = tournament_selection(fitness_values, GA_PARAMS['tournament_size'])
                p2_idx = tournament_selection(fitness_values, GA_PARAMS['tournament_size'])
                p1, p2 = population[p1_idx], population[p2_idx]
                child = p1.copy()
                if np.random.rand() < GA_PARAMS['crossover_rate_tasks']: child[
                    'task_counts'], _ = crossover_task_counts(p1, p2, GA_PARAMS, city_angle_sort_map)
                if np.random.rand() < GA_PARAMS['crossover_rate_perm']: child[
                    'permutation'] = crossover_permutation_ox1(p1, p2, child['task_counts'], GA_PARAMS)
                if np.random.rand() < GA_PARAMS['mutation_rate_tasks']: child = mutate_task_counts_special(child,
                                                                                                           GA_PARAMS,
                                                                                                           city_angle_sort_map)
                if np.random.rand() < GA_PARAMS['mutation_rate_perm']: child[
                    'permutation'] = mutate_permutation_segmented(child, GA_PARAMS)
                # [注意] 这里移除了普通子代的 2-Opt
                new_population.append(child)
            population = new_population

    print("\n等待RRT任务完成...")
    task_queue.join()
    for _ in range(num_consumers): task_queue.put(None)
    task_queue.join()
    for p in consumers: p.join()

    with lock:
        np.copyto(stable_matrix_view, live_matrix_view)
    final_fitness_values = np.array(
        [parallel_fitness_calculation(ind, stable_matrix_view, GA_PARAMS)[0] for ind in population])
    best_idx = np.argmin(final_fitness_values)
    best_solution = population[best_idx]
    best_solution = apply_2opt_to_individual_rrt(best_solution, stable_matrix_view, GA_PARAMS)

    # 整合最终路径
    final_routes_indices = []
    current_pos = 0
    for count in best_solution['task_counts']:
        perm_indices = best_solution['permutation'][current_pos:current_pos + count]
        route = [0] + (perm_indices + 1).tolist() + [0]
        final_routes_indices.append(route)
        current_pos += count

    planned_final_paths = {}
    for agent_id, route_indices in enumerate(final_routes_indices):
        if len(route_indices) < 2:
            planned_final_paths[agent_id] = []
            continue
        full_agent_path = []
        for i in range(len(route_indices) - 1):
            start_idx, goal_idx = route_indices[i], route_indices[i + 1]
            if start_idx == goal_idx: continue
            cache_key = tuple(sorted((start_idx, goal_idx)))
            segment_path = shared_path_cache.get(cache_key)
            if segment_path:
                if np.linalg.norm(np.array(segment_path[0]) - np.array(TASK_POINTS[start_idx])) > 1:
                    segment_path = segment_path[::-1]
                if full_agent_path: full_agent_path.pop()
                full_agent_path.extend(segment_path)
            else:
                if full_agent_path and np.array_equal(full_agent_path[-1], TASK_POINTS[start_idx]):
                    full_agent_path.append(TASK_POINTS[goal_idx])
                else:
                    if full_agent_path: full_agent_path.pop()
                    full_agent_path.extend([TASK_POINTS[start_idx], TASK_POINTS[goal_idx]])
        planned_final_paths[agent_id] = full_agent_path

    end_time_total = time.time()

    # ==================================================================================
    # [NEW] IEEE Journal Style Plotting: Path Discovery Analysis
    # ==================================================================================
    print("\n正在生成符合 IEEE 格式的路径发现分析图...")
    generations = np.arange(len(path_discovery_history))
    new_paths = np.array(path_discovery_history)
    cumulative_paths = np.cumsum(new_paths)

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 10,
        'axes.grid': True,
        'grid.alpha': 0.3,
    })

    fig_analysis, ax1 = plt.subplots(figsize=(3.5, 2.5), constrained_layout=True)

    color_left = '#1f77b4'
    ax1.set_xlabel('Generation $t$')
    ax1.set_ylabel(r'Per generation $M_{\text{new}}(t)$', color=color_left)
    ax1.vlines(generations, 0, new_paths, color=color_left, linewidth=0.5, alpha=0.8, zorder=2)
    ax1.tick_params(axis='y', labelcolor=color_left)

    ax2 = ax1.twinx()
    color_right = '#d62728'
    ax2.set_ylabel(r'Cumulative $M_{\text{cum}}(t)$', color=color_right)
    ax2.plot(generations, cumulative_paths, color=color_right, linewidth=1.5, linestyle='--', zorder=3)
    ax2.tick_params(axis='y', labelcolor=color_right)

    proxy_line = Line2D([0], [0], color=color_left, linewidth=1, alpha=0.8)
    lines_2 = ax2.get_lines()
    ax1.legend(
        [proxy_line, lines_2[0]],
        ['Per generation', 'Cumulative'],
        loc='right',
        framealpha=0.9
    )

    ax1.set_xlim(-100, len(generations))
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)

    plt.savefig('ieee_path_discovery_analysis.pdf', format='pdf', dpi=600)
    plt.savefig('ieee_path_discovery_analysis.svg', format='svg', dpi=600)
    # ==================================================================================

    fig, ax = plt.subplots(figsize=(18, 14))
    ax.contourf(np.arange(0.5, map_width + 0.5), np.arange(0.5, map_height + 0.5), terrain_map, levels=boundaries,
                cmap=cmap, norm=norm)
    ax.set_aspect('equal')

    colors = list(matplotlib.colormaps['tab10'].colors)

    for i, point in enumerate(TASK_POINTS):
        if i == 0: continue
        ax.plot(point[0], point[1], 'o', markersize=10, markeredgecolor='white', c='red', zorder=11)

    ax.plot(TASK_POINTS[0][0], TASK_POINTS[0][1], 'ks', markersize=14, markeredgecolor='white', zorder=12,
            label='Depot')

    for agent_id, path in planned_final_paths.items():
        if path:
            px, py = zip(*path)
            agent_label = f'UAV {agent_id + 1}'
            ax.plot(px, py, color=colors[agent_id % len(colors)], linewidth=5, zorder=8, label=agent_label)

    #ax.legend(loc='upper left', fontsize='large')
    plt.savefig('final_async_rrt_ga_plan_with_legend.pdf', dpi=300)
    plt.savefig('final_async_rrt_ga_plan_with_legend.svg', dpi=300)
    # ==================================================================================
    # [RESTORED] 详细输出打印块
    # ==================================================================================
    print("\n" + "=" * 70 + "\n--- 最终成本分析 (使用GA适应度函数) ---\n" + "=" * 70)
    print(f"最终任务分配方案 (任务点索引): {final_routes_indices}")

    final_fitness, final_max_workload, final_total_workload, final_agent_workloads = parallel_fitness_calculation(
        best_solution, stable_matrix_view, GA_PARAMS)

    agent_costs = final_agent_workloads
    if agent_costs.size > 0:
        print(f"各代理最终综合成本 (Workload): {[round(c, 2) for c in agent_costs]}")
        path_only_costs = [calculate_path_length(p) for p in planned_final_paths.values()]
        print(f"   (参考) 各代理单纯路径长度: {[round(c, 2) for c in path_only_costs]}")
        print(f"\n最大综合成本 (Makespan / Max Workload): {final_max_workload:.2f}")
        print(f"所有代理总综合成本 (Total Workload): {final_total_workload:.2f}")
        balance_penalty = np.std(best_solution['task_counts']) * GA_PARAMS.get('balance_weight_task_count', 0.0)
        if balance_penalty > 0:
            print(f"任务平衡惩罚项: {balance_penalty:.2f}")
        print(f"\n最终总适应度 (Fitness Value): {final_fitness:.2f}")

    print(f"\n总计算时间: {end_time_total - start_time_total:.2f} 秒")
    plt.show()


if __name__ == '__main__':
    main()