# path_planning/rrt_path_planning.py (VISUALIZATION-ENABLED VERSION)

import math
import random
import numpy as np
import matplotlib.pyplot as plt


class RRT:
    class Node:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.parent = None

    def __init__(self, start, goal, obstacle_map, iter_max,
                 min_step_size=10, max_step_size=50, influence_radius=100,
                 goal_sample_rate=0.3,
                 # --- 新增可视化参数 ---
                 ax=None, visualize=False, vis_interval=200):
        self.start = self.Node(start[0], start[1])
        self.goal = self.Node(goal[0], goal[1])
        self.obstacle_map = obstacle_map
        self.iter_max = iter_max
        self.min_step_size = min_step_size
        self.max_step_size = max_step_size
        self.influence_radius = influence_radius
        self.goal_sample_rate = goal_sample_rate

        self.map_height, self.map_width = self.obstacle_map.shape
        self.node_list = [self.start]

        # --- 新增可视化属性 ---
        self.ax = ax
        self.visualize = visualize
        self.vis_interval = vis_interval
        # 用于存储绘制的树枝，以便后续清除
        self.plotted_tree_lines = []

    def plan(self):
        """RRT路径规划主函数"""
        for i in range(self.iter_max):
            rnd_node = self.get_random_node()
            nearest_node = self.get_nearest_node(rnd_node)
            new_node = self.steer(nearest_node, rnd_node)

            if new_node and self.is_collision_free(nearest_node, new_node):
                near_nodes_indices = self.find_near_nodes(new_node)
                self.choose_parent(new_node, near_nodes_indices)
                self.rewire(new_node, near_nodes_indices)
                self.node_list.append(new_node)

                # --- 核心修改：周期性地绘制扩展树 ---
                if self.visualize and self.ax and i % self.vis_interval == 0:
                    # 只绘制最新的树枝，效率更高
                    if new_node.parent:
                        line, = self.ax.plot([new_node.x, new_node.parent.x],
                                             [new_node.y, new_node.parent.y],
                                             color="gray", linewidth=0.5, alpha=0.7)
                        self.plotted_tree_lines.append(line)
                        plt.pause(0.001)  # 暂停一小下以刷新画布

            # 检查是否可以连接到目标点
            dist_to_goal = self.calc_dist_to_goal(self.node_list[-1].x, self.node_list[-1].y)
            if dist_to_goal <= self.max_step_size:
                if self.is_collision_free(self.node_list[-1], self.goal):
                    final_path = self.generate_final_path(self.node_list[-1])
                    if self.visualize and self.ax:
                        self.draw_final_path(final_path)
                    return final_path

        return None  # 未找到路径

    # --- 新增：绘制最终找到的原始路径 ---
    def draw_final_path(self, path):
        path_coords = np.array(path)
        self.ax.plot(path_coords[:, 0], path_coords[:, 1], 'r-', linewidth=2, zorder=10)
        plt.pause(0.01)

    def steer(self, from_node, to_node):
        dist = np.hypot(to_node.x - from_node.x, to_node.y - from_node.y)
        step_size = np.clip(dist, self.min_step_size, self.max_step_size)

        theta = math.atan2(to_node.y - from_node.y, to_node.x - from_node.x)

        new_node = self.Node(from_node.x + step_size * math.cos(theta),
                             from_node.y + step_size * math.sin(theta))
        new_node.parent = from_node
        return new_node

    def get_random_node(self):
        if random.random() > self.goal_sample_rate:
            return self.Node(random.uniform(0, self.map_width),
                             random.uniform(0, self.map_height))
        return self.goal

    def get_nearest_node(self, rnd_node):
        dists = [(node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y) ** 2 for node in self.node_list]
        return self.node_list[np.argmin(dists)]

    def is_collision_free(self, from_node, to_node, delta=1.0):
        p1 = np.array([from_node.x, from_node.y])
        p2 = np.array([to_node.x, to_node.y])
        v = p2 - p1
        distance = np.linalg.norm(v)
        if distance < 1e-6: return True

        u = v / distance
        num_steps = int(distance / delta)
        for i in range(num_steps):
            current_pos = p1 + i * delta * u
            x_idx, y_idx = int(current_pos[0]), int(current_pos[1])
            if not (0 <= x_idx < self.map_width and 0 <= y_idx < self.map_height):
                return False
            if self.obstacle_map[y_idx, x_idx] >= 60:
                return False
        return True

    def find_near_nodes(self, new_node):
        n_node = len(self.node_list)
        r = min(self.influence_radius * math.sqrt((math.log(n_node) / n_node)), self.max_step_size * 2)

        dists = [(node.x - new_node.x) ** 2 + (node.y - new_node.y) ** 2 for node in self.node_list]
        near_indices = [i for i, dist in enumerate(dists) if dist <= r ** 2]
        return near_indices

    def choose_parent(self, new_node, near_indices):
        if not near_indices: return

        costs = []
        for i in near_indices:
            near_node = self.node_list[i]
            cost = self.calc_new_cost(near_node, new_node)
            if self.is_collision_free(near_node, new_node):
                costs.append(cost)
            else:
                costs.append(float('inf'))

        min_cost = min(costs)
        if min_cost == float('inf'): return

        min_idx = near_indices[costs.index(min_cost)]
        new_node.parent = self.node_list[min_idx]

    def rewire(self, new_node, near_indices):
        for i in near_indices:
            near_node = self.node_list[i]
            edge_cost = self.calc_dist(new_node, near_node)
            new_cost = self.get_node_cost(new_node) + edge_cost

            if self.get_node_cost(near_node) > new_cost:
                if self.is_collision_free(new_node, near_node):
                    near_node.parent = new_node

    def get_node_cost(self, node):
        cost = 0.0
        current = node
        while current.parent:
            cost += self.calc_dist(current, current.parent)
            current = current.parent
        return cost

    def calc_new_cost(self, from_node, to_node):
        return self.get_node_cost(from_node) + self.calc_dist(from_node, to_node)

    @staticmethod
    def calc_dist(from_node, to_node):
        return np.hypot(to_node.x - from_node.x, to_node.y - from_node.y)

    def calc_dist_to_goal(self, x, y):
        return self.calc_dist(self.Node(x, y), self.goal)

    def generate_final_path(self, goal_node):
        path = [[self.goal.x, self.goal.y]]
        node = self.Node(goal_node.x, goal_node.y)
        node.parent = goal_node.parent

        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([self.start.x, self.start.y])
        return path[::-1]