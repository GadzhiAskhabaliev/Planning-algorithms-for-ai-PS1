#!/usr/bin/env python3
"""Planning Algorithms in AI - PS1: Discrete Planning."""

from __future__ import annotations

import heapq
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

import utils

# Avoid blocking on plt.show() inside utils plotting helpers.
plt.show = lambda *args, **kwargs: None

DATA_PATH = Path(__file__).resolve().parent / "data_ps1.npz"
OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def reconstruct_path(came_from: dict, goal):
    path = [goal]
    cur = goal
    while cur in came_from:
        cur = came_from[cur]
        path.append(cur)
    path.reverse()
    return path


def astar(start, goal, neighbors_fn, heuristic_fn, is_valid_fn=None):
    """A* / Dijkstra (heuristic_fn = 0). Returns (path, visited_count, cost)."""
    if is_valid_fn is not None and (not is_valid_fn(start) or not is_valid_fn(goal)):
        raise ValueError(f"Invalid start/goal: {start} -> {goal}")

    open_heap = []  # (f, g, counter, state)
    counter = 0
    g_score = {start: 0}
    came_from = {}
    closed = set()
    heapq.heappush(open_heap, (heuristic_fn(start, goal), 0, counter, start))
    counter += 1
    visited = 0

    while open_heap:
        f, g, _, state = heapq.heappop(open_heap)
        if state in closed:
            continue
        closed.add(state)
        visited += 1

        if state == goal:
            path = reconstruct_path(came_from, goal)
            return path, visited, g_score[goal]

        for nxt, step_cost in neighbors_fn(state):
            if is_valid_fn is not None and not is_valid_fn(nxt):
                continue
            if nxt in closed:
                continue
            tentative = g + step_cost
            if tentative < g_score.get(nxt, float("inf")):
                came_from[nxt] = state
                g_score[nxt] = tentative
                f_new = tentative + heuristic_fn(nxt, goal)
                heapq.heappush(open_heap, (f_new, tentative, counter, nxt))
                counter += 1

    raise RuntimeError(f"No path found from {start} to {goal}")


# Task 1: point robot

MOVES_2D = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def neighbors_2d(state):
    x, y = state
    for dx, dy in MOVES_2D:
        yield (x + dx, y + dy), 1


def make_valid_2d(environment: np.ndarray):
    h, w = environment.shape

    def is_valid(state):
        x, y = state
        return 0 <= x < h and 0 <= y < w and environment[x, y] == 0

    return is_valid


def h_zero(q, qg):
    return 0


def h_l1_2d(q, qg):
    return abs(q[0] - qg[0]) + abs(q[1] - qg[1])


def run_task1(environment: np.ndarray):
    print("\n=== Task 1: Dijkstra / A* in 2D ===")
    start, goal = (6, 6), (85, 85)
    is_valid = make_valid_2d(environment)

    path_d, vis_d, cost_d = astar(start, goal, neighbors_2d, h_zero, is_valid)
    path_a, vis_a, cost_a = astar(start, goal, neighbors_2d, h_l1_2d, is_valid)

    print(f"Dijkstra (h=0): visited={vis_d}, cost={cost_d}, path_len={len(path_d)}")
    print(f"A* (L1):        visited={vis_a}, cost={cost_a}, path_len={len(path_a)}")

    gif_path = OUT_DIR / "point_solve.gif"
    utils.plotting_results_2d(environment, path_a, save_path=str(gif_path))
    # Also copy to submission root name
    utils.plotting_results_2d(environment, path_a, save_path=str(Path(__file__).parent / "point_solve.gif"))
    print(f"Saved {gif_path}")

    # Start/goal visualization
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(environment, cmap="gray_r", origin="upper")
    ax.plot(start[1], start[0], "go", label="start")
    ax.plot(goal[1], goal[0], "ro", label="goal")
    ys = [p[1] for p in path_a]
    xs = [p[0] for p in path_a]
    ax.plot(ys, xs, "b-", linewidth=1, label="A* path")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Task 1: point robot path (A* L1)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "task1_path.png", dpi=150)
    plt.close(fig)

    return {
        "dijkstra": {"visited": vis_d, "cost": cost_d, "path": path_d},
        "astar_l1": {"visited": vis_a, "cost": cost_a, "path": path_a},
    }


# Task 2: configuration space

def build_cspace(environment: np.ndarray, rod: np.ndarray) -> np.ndarray:
    """C-space as (H, W, 4); 0 free, 1 collision/obstacle."""
    n_theta = rod.shape[2]
    cspace = np.zeros(environment.shape + (n_theta,), dtype=np.float64)
    for theta in range(n_theta):
        # Rod footprints are centrosymmetric (k == k[::-1,::-1]), so convolve2d
        # equals correlate2d; keep convolve2d as in the assignment hint.
        # boundary='symm' matches free counts of fill/fillvalue=1 on this map.
        kernel = rod[:, :, theta]
        conv = signal.convolve2d(environment, kernel, boundary="symm", mode="same")
        cspace[:, :, theta] = utils.normalize_image(conv)
    return cspace


def run_task2(environment: np.ndarray, rod: np.ndarray):
    print("\n=== Task 2: State / Configuration Space ===")

    # A: workspace + rod orientations
    fig, axes = plt.subplots(1, 5, figsize=(14, 3))
    axes[0].imshow(environment, cmap="gray_r")
    axes[0].set_title("Workspace")
    for i in range(4):
        axes[i + 1].imshow(rod[:, :, i], cmap="gray_r")
        axes[i + 1].set_title(f"rod θ={i} ({i * 45}°)")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT_DIR / "task2_workspace_rod.png", dpi=150)
    plt.close(fig)

    # B: environment + rod at a valid config
    demo_state = (50, 50, 0)
    merged = utils.plot_environment_3d(environment, rod, demo_state)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(merged, cmap="gray_r")
    ax.set_title(f"Env + rod at {demo_state}")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT_DIR / "task2_env_rod.png", dpi=150)
    plt.close(fig)

    # C: C-space slices
    cspace = build_cspace(environment, rod)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for i in range(4):
        axes[i].imshow(cspace[:, :, i], cmap="gray_r")
        free = int(np.sum(cspace[:, :, i] == 0))
        axes[i].set_title(f"C-space θ={i}\nfree={free}")
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    fig.suptitle(f"C-space shape = {cspace.shape}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "task2_cspace.png", dpi=150)
    plt.close(fig)

    print(f"C-space shape: {cspace.shape}")
    for i in range(4):
        print(f"  θ={i}: free={int(np.sum(cspace[:,:,i]==0))}, "
              f"occ={int(np.sum(cspace[:,:,i]==1))}")

    return cspace


# Task 3: rod planning

def neighbors_3d(state):
    x, y, theta = state
    for dx, dy in MOVES_2D:
        yield (x + dx, y + dy, theta), 1
    yield (x, y, (theta + 1) % 4), 1  # rotate right
    yield (x, y, (theta - 1) % 4), 1  # rotate left


def make_valid_3d(cspace: np.ndarray):
    h, w, n_theta = cspace.shape

    def is_valid(state):
        x, y, theta = state
        return (
            0 <= x < h
            and 0 <= y < w
            and 0 <= theta < n_theta
            and cspace[x, y, theta] == 0
        )

    return is_valid


def h_l1_xy(q, qg):
    return abs(q[0] - qg[0]) + abs(q[1] - qg[1])


def h_l1_xy_theta(q, qg):
    """Admissible: L1 on (x,y) + shortest discrete orientation distance."""
    d_xy = abs(q[0] - qg[0]) + abs(q[1] - qg[1])
    d_th = abs(q[2] - qg[2])
    d_th = min(d_th, 4 - d_th)
    return d_xy + d_th


def run_task3(environment: np.ndarray, rod: np.ndarray, cspace: np.ndarray):
    print("\n=== Task 3: A* in 3D (x, y, θ) ===")
    start, goal = (6, 6, 2), (85, 85, 0)
    is_valid = make_valid_3d(cspace)
    print(f"start valid={is_valid(start)}, goal valid={is_valid(goal)}")

    path_d, vis_d, cost_d = astar(start, goal, neighbors_3d, h_zero, is_valid)
    path_l1, vis_l1, cost_l1 = astar(start, goal, neighbors_3d, h_l1_xy, is_valid)
    path_th, vis_th, cost_th = astar(start, goal, neighbors_3d, h_l1_xy_theta, is_valid)

    print(f"Dijkstra (h=0):     visited={vis_d}, cost={cost_d}, path_len={len(path_d)}")
    print(f"A* (L1 xy):         visited={vis_l1}, cost={cost_l1}, path_len={len(path_l1)}")
    print(f"A* (L1 xy + θ):     visited={vis_th}, cost={cost_th}, path_len={len(path_th)}")

    gif_path = OUT_DIR / "rod_solve.gif"
    # Prefer the orientation-aware path for visualization (same optimal cost).
    utils.plotting_results_3d(environment, rod, path_th, save_path=str(gif_path))
    utils.plotting_results_3d(
        environment, rod, path_th, save_path=str(Path(__file__).parent / "rod_solve.gif")
    )
    print(f"Saved {gif_path}")

    return {
        "dijkstra": {"visited": vis_d, "cost": cost_d, "path": path_d},
        "astar_l1": {"visited": vis_l1, "cost": cost_l1, "path": path_l1},
        "astar_orient": {"visited": vis_th, "cost": cost_th, "path": path_th},
    }


def main():
    data = np.load(DATA_PATH)
    environment = data["environment"]
    rod = data["rod"]

    run_task1(environment)
    cspace = run_task2(environment, rod)
    run_task3(environment, rod, cspace)
    print("\nDone. Figures/GIFs in:", OUT_DIR)


if __name__ == "__main__":
    main()
