import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from matplotlib.path import Path
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import os

# 您的 ga_functions 导入保持不变
from ga_functions import (
    initialize_population,
    calculate_fitness_minimize_max_workload,
    tournament_selection,
    crossover_task_counts,
    crossover_permutation_ox1,
    mutate_task_counts_special,
    mutate_permutation_segmented,
    apply_2opt_to_individual,
    get_city_segments,
    _two_opt,
    parallel_fitness_calculation
)

def load_tsp_dataset(file_path):
    """
    从 TSPLIB 格式文件中读取城市坐标。
    支持标准格式，如：
    NAME: eil51
    TYPE: TSP
    DIMENSION: 51
    NODE_COORD_SECTION
    1 37 52
    2 49 49
    ...
    EOF
    """
    city_points = []
    reading_coords = False

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("NODE_COORD_SECTION"):
                reading_coords = True
                continue
            if line.startswith("EOF"):
                break
            if reading_coords and line:
                parts = line.split()
                if len(parts) >= 3:
                    x, y = float(parts[1]), float(parts[2])
                    city_points.append((x, y))

    return city_points

def solve_mtsp_ga_memetic(ga_params, task_points, show_plots=True, verbose=True, custom_colors=None, plot_params=None):
    # --- 算法开始到主循环部分，完全保持您原始代码不变 ---
    start_time = time.time()

    depot_coordinates = np.array(task_points[0])
    city_coordinates = np.array(task_points[1:])
    ga_params['n_cities'] = len(city_coordinates)

    if verbose:
        print('Initializing MTSP Memetic Algorithm Solver...')
        print(f"Number of cities: {ga_params['n_cities']}")
        print(f"Number of salesmen: {ga_params['n_salesmen']}")
        print(f"Population size: {ga_params['population_size']}")
        print(f"Number of generations: {ga_params['n_generations']}")

    if verbose:
        print("Warming up JIT-compiled functions (e.g., 2-Opt)...")
    if ga_params.get('use_2opt', False) and ga_params['n_cities'] > 1:
        dummy_route = np.arange(min(10, ga_params['n_cities']), dtype=np.int64)
        _two_opt(dummy_route, city_coordinates, depot_coordinates)

    if ga_params['n_cities'] > 0:
        city_angles = np.arctan2(city_coordinates[:, 1] - depot_coordinates[1],
                                 city_coordinates[:, 0] - depot_coordinates[0])
        city_angle_sort_map = np.argsort(city_angles)
    else:
        city_angle_sort_map = np.array([], dtype=int)

    population = initialize_population(ga_params, city_angle_sort_map)
    if ga_params.get('use_2opt', False):
        if verbose:
            print('Applying 2-Opt to the initial population...')
        population = [apply_2opt_to_individual(ind, city_coordinates, depot_coordinates, ga_params)
                      for ind in tqdm(population, desc="Initial 2-Opt", disable=not verbose)]

    best_fitness_history = np.zeros(ga_params['n_generations'])

    if verbose:
        print('\nStarting the evolution process (using multiprocessing.Pool)...')

    with Pool(processes=cpu_count()) as pool:
        gen_iterator = tqdm(range(ga_params['n_generations']), desc="Generations", disable=not verbose)

        for gen in gen_iterator:
            tasks_args = [(ind, city_coordinates, depot_coordinates, ga_params) for ind in population]
            fitness_values = np.array(pool.map(parallel_fitness_calculation, tasks_args))
            best_fitness_history[gen] = np.min(fitness_values)

            new_population = []
            elite_idx = np.argmin(fitness_values)
            new_population.append(population[elite_idx].copy())

            for _ in range(1, ga_params['population_size']):
                parent1_idx = tournament_selection(fitness_values, ga_params['tournament_size'])
                parent2_idx = tournament_selection(fitness_values, ga_params['tournament_size'])
                parent1 = population[parent1_idx]
                parent2 = population[parent2_idx]
                child = parent1.copy()

                if np.random.rand() < ga_params['crossover_rate_tasks'] and ga_params['n_salesmen'] > 1:
                    child_tasks, _ = crossover_task_counts(parent1, parent2, ga_params, city_angle_sort_map)
                    child['task_counts'] = child_tasks
                if np.random.rand() < ga_params['crossover_rate_perm'] and ga_params['n_cities'] > 0:
                    child['permutation'] = crossover_permutation_ox1(parent1, parent2, child['task_counts'], ga_params)
                if np.random.rand() < ga_params['mutation_rate_tasks'] and ga_params['n_salesmen'] > 1:
                    child = mutate_task_counts_special(child, ga_params, city_angle_sort_map)
                if np.random.rand() < ga_params['mutation_rate_perm'] and ga_params['n_cities'] > 0:
                    child['permutation'] = mutate_permutation_segmented(child, ga_params)
                if ga_params.get('use_2opt', False) and gen > 0 and gen % int(ga_params['n_generations'] * 0.1) == 0:
                    child = apply_2opt_to_individual(child, city_coordinates, depot_coordinates, ga_params)

                new_population.append(child)
            population = new_population

    total_time = time.time() - start_time
    # --- 5. 结果提取 (保持不变) ---
    final_results = [calculate_fitness_minimize_max_workload(ind, city_coordinates, depot_coordinates, ga_params) for
                     ind in population]
    final_fitness = np.array([res[0] for res in final_results])
    best_idx = np.argmin(final_fitness)
    best_fitness, best_max_path, best_total_path, best_paths = final_results[best_idx]
    best_solution = population[best_idx]
    city_segments = get_city_segments(best_solution, ga_params)


    # ========================== MODIFICATION START ==========================
    # --- 优化后的信息打印 ---
    if verbose:
        print("\n" + "="*40)
        print("--- Algorithm Execution Summary ---")
        print(f"Total Calculation Time: {total_time:.2f} seconds")
        print("\n--- Optimal Solution Details ---")
        # "最大成本" 即为算法优化的目标 (Minimized Max Path Length)
        print(f"Max Cost (Minimized Objective): {best_max_path:.2f}")
        # "总成本" 即为所有无人机的路径总长 (Corresponding Total Path Length)
        print(f"Total Cost (Sum of all routes): {best_total_path:.2f}")
        print("\n--- Costs per UAV ---")
        if ga_params['n_salesmen'] > 0:
            for i, cost in enumerate(best_paths):
                # "每个无人机的成本"
                print(f"  UAV {i+1} Cost: {cost:.2f}")
        print("-" * 25)
        print(f"Task Allocation: {best_solution['task_counts']}")
        print("="*40)

    # --- 6. 可视化 (修改为两个独立窗口) ---
    if show_plots and ga_params['n_cities'] > 0:
        plt.style.use('seaborn-v0_8-whitegrid')

        # --- 图 1: 绘制收敛曲线 (独立窗口) ---
        plt.figure("GA Convergence Curve", figsize=(10, 6))
        plt.plot(best_fitness_history, 'b-', linewidth=2, label='Best Fitness (Cost)')
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Fitness (Cost)', fontsize=12)
        plt.title(f'GA Convergence (Final Minimized Cost: {best_fitness:.2f})', fontsize=14)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # --- 图 2: 绘制路径图 (独立窗口) ---
        plt.figure("Optimized UAV Routes", figsize=(12, 12))
        ax2 = plt.gca() # 获取当前Axes

        # A. 绘制背景图、遮罩和边界 (逻辑不变)
        if plot_params and 'background_image_path' in plot_params:
            try:
                img = mpimg.imread(plot_params['background_image_path'])
                img_extent = plot_params.get('image_extent', [-1200, 1200, -1200, 1200])
                ax2.imshow(img, extent=img_extent, zorder=-10)

                if 'task_area_coords' in plot_params:
                    coords = plot_params['task_area_coords']
                    tx, ty, tw, th = coords['x'], coords['y'], coords['width'], coords['height']
                    rect_verts = [(img_extent[0], img_extent[2]), (img_extent[1], img_extent[2]),
                                  (img_extent[1], img_extent[3]), (img_extent[0], img_extent[3]), (0, 0)]
                    rect_codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
                    hole_verts = [(tx, ty), (tx, ty + th), (tx + tw, ty + th), (tx + tw, ty), (tx, ty)]
                    hole_codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
                    path_verts = np.vstack([rect_verts, hole_verts])
                    path_codes = np.concatenate([rect_codes, hole_codes])
                    path = Path(path_verts, path_codes)
                    patch = patches.PathPatch(path, facecolor='white', alpha=plot_params.get('mask_alpha', 0.65),
                                              edgecolor='none', zorder=5)
                    ax2.add_patch(patch)
                    boundary_rect = patches.Rectangle((tx, ty), tw, th,
                                                      linewidth=plot_params.get('boundary_linewidth', 2.5),
                                                      edgecolor=plot_params.get('boundary_color', 'black'),
                                                      facecolor='none', zorder=6)
                    ax2.add_patch(boundary_rect)
            except FileNotFoundError:
                if verbose: print(f"Warning: Background image not found. Skipping.")
            except Exception as e:
                if verbose: print(f"An error occurred with the background image: {e}")

        # B. 颜色自定义逻辑 (逻辑不变)
        n_salesmen = ga_params['n_salesmen']
        plot_colors = custom_colors[:n_salesmen] if custom_colors and len(custom_colors) >= n_salesmen else [
            plt.cm.tab10(i) for i in range(n_salesmen)]

        # C. 绘制路径、箭头和任务点
        for s, seg in enumerate(city_segments):
            if seg.size > 0:
                path_coords = np.vstack([depot_coordinates, city_coordinates[seg, :], depot_coordinates])
                ax2.plot(path_coords[:, 0], path_coords[:, 1], '-', color=plot_colors[s], linewidth=10.0,
                         alpha=0.9, label=f'UAV {s + 1}', zorder=9)
                # 删除了箭头的绘制

        ax2.plot(depot_coordinates[0], depot_coordinates[1], 's', color='#0055A4', markersize=14,
                 markeredgecolor='white', mew=2.0, label='Depot', zorder=10)
        for s, seg in enumerate(city_segments):
            if seg.size > 0:
                ax2.plot(city_coordinates[seg, 0], city_coordinates[seg, 1], 'o', color=plot_colors[s], markersize=9,
                         markeredgecolor='k', mew=1.5, zorder=10)

        # D. 美化图表
        ax2.set_xlabel('X Coordinate', fontsize=18)
        ax2.set_ylabel('Y Coordinate', fontsize=18)
        # 删除了标题的设置
        if plot_params and 'image_extent' in plot_params:
            ax2.set_xlim(plot_params['image_extent'][0], plot_params['image_extent'][1])
            ax2.set_ylim(plot_params['image_extent'][2], plot_params['image_extent'][3])
        ax2.set_aspect('equal')
        ax2.grid(False) # <--- 新增此行以关闭路径图的网格
        legend = ax2.legend(loc='upper left',  # 位置
                            fontsize=18,  # 字体大小
                            frameon=True,  # 强制开启图例框
                            facecolor='white',  # 设置背景为白色
                            edgecolor='black',  # 设置一个清晰的黑色边框
                            shadow=True,  # 添加阴影以增强可读性
                            fancybox=True  # 可选：圆角边框
                            )
        legend.set_zorder(100)
        plt.tight_layout()

        plt.savefig(
            "our_proposed.pdf", # 1. 文件名和格式
            dpi=300,                     # 2. (可选) 设置高分辨率
            bbox_inches='tight'          # 3. (可选) 自动裁剪掉多余的白边
        )

        # 显示所有创建的图形
        plt.show()

    # ========================== MODIFICATION END ============================

    # --- 7. 返回结果 (保持不变) ---
    results_dict = {
        "best_fitness": best_fitness,
        "max_path_length": best_max_path,
        "total_path_length": best_total_path,
        "task_counts": list(best_solution["task_counts"]),
        "final_routes_indices": [arr.tolist() for arr in city_segments],
        "individual_path_lengths": best_paths.tolist(),
        "fitness_history": best_fitness_history.tolist(),
        "total_time_seconds": total_time,
    }
    return results_dict


if __name__ == '__main__':
    # --- 主程序部分，完全保持您原始代码不变 ---
    # 确保在 Windows 上使用 multiprocessing 时代码被 __main__ 保护

    # --- 算法参数 ---
    ga_parameters = {
        'n_salesmen': 4,
        'population_size': 50,
        'n_generations': 10000,
        'tournament_size': 3,
        'crossover_rate_tasks': 0.2, 'crossover_rate_perm': 0.2,
        'mutation_rate_tasks': 0.3, 'mutation_rate_perm': 0.35,
        'min_cities_per_salesman': 2, 'use_2opt': True,
        'weight_max_path_length': 1.0, 'weight_total_path_length':0.01,
        'balance_weight_task_count': 0.0,
    }

    # --- 任务点生成 ---
    num_cities = 51
    np.random.seed(42)
    depot_point = (0, 0)
    # 确保任务点主要分布在任务区域内部
    city_points = list(zip(np.random.uniform(-1000, 1000, num_cities),
                           np.random.uniform(-1000, 1000, num_cities)))


    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'VrpLib', 'eil51.vrp')
    #city_points = load_tsp_dataset(file_path)

    # 将坐标缩放到 2km × 2km 范围
    city_points = np.array(city_points)
    city_points = 2000 * (city_points - city_points.min(axis=0)) / (city_points.max(axis=0) - city_points.min(axis=0))
    city_points = city_points - 1000
    city_points = list(map(tuple, city_points))

    all_task_points = [depot_point] + city_points

    # --- 绘图参数设置 ---
    plot_parameters = {
        'background_image_path': 'Agriculture.png',
        'image_extent': [-1100, 1100, -1100, 1100],
        'task_area_coords': {'x': -1000, 'y': -1000, 'width': 2000, 'height': 2000},
        'mask_alpha': 0.65,
        'boundary_color': 'black',
        'boundary_linewidth': 2.0
    }

    # --- 运行算法 ---
    print("\n--- Running Solver with Custom Plotting ---")
    my_custom_colors = [
        (235 / 255.0, 108 / 255.0, 21 / 255.0), (129 / 255.0, 59 / 255.0, 182 / 255.0),
        (71 / 255.0, 194 / 255.0, 243 / 255.0), (255 / 255.0, 113 / 255.0, 113 / 255.0)
    ]

    results_custom = solve_mtsp_ga_memetic(
        ga_params=ga_parameters,
        task_points=all_task_points,
        show_plots=True,
        verbose=True,
        custom_colors=my_custom_colors,
        plot_params=plot_parameters
    )
