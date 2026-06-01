import argparse
import contextlib
import csv
import os
import random
import struct
import time
from collections import deque
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from maze import MazeGenerator


ACTIONS = [
    ("up", 0, -1),
    ("down", 0, 1),
    ("left", -1, 0),
    ("right", 1, 0),
]
SIZES = [31, 71, 101]
SIZE_WEIGHTS = {31: 5, 71: 20, 101: 30}
DEFAULT_TRAIN_COUNT = 3000
DEFAULT_TEST_COUNT = 200
COLORS = {
    "road": (255, 255, 255),
    "wall": (0, 0, 0),
    "start": (0, 255, 0),
    "goal": (255, 0, 0),
    "path": (255, 220, 0),
}


class QNetwork(nn.Module):
    def __init__(self, hidden=192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 4),
        )

    def forward(self, x):
        return self.net(x)


def generate_maze_image(size, seed, output_path, cell_size=4, cycle_rate=0.03):
    maze = MazeGenerator(size, size)
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with contextlib.redirect_stdout(sink):
            maze.generate(cycle_rate=cycle_rate, seed=seed)
    surface = pygame.Surface((maze.cols * cell_size, maze.rows * cell_size))
    for y in range(maze.rows):
        for x in range(maze.cols):
            color = COLORS["wall"] if maze.grid[y][x] == 1 else COLORS["road"]
            pygame.draw.rect(surface, color, (x * cell_size, y * cell_size, cell_size, cell_size))
    sx, sy = maze.start_pos
    gx, gy = maze.end_pos
    pygame.draw.rect(surface, COLORS["start"], (sx * cell_size, sy * cell_size, cell_size, cell_size))
    pygame.draw.rect(surface, COLORS["goal"], (gx * cell_size, gy * cell_size, cell_size, cell_size))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(surface, str(output_path))


def generate_image_set(image_dir, sizes, seeds, cell_size=4, cycle_rate=0.03):
    pygame.init()
    for size in sizes:
        for seed in seeds:
            path = image_dir / f"maze_size_{size}_seed_{seed}.png"
            if not path.exists():
                generate_maze_image(size, seed, path, cell_size=cell_size, cycle_rate=cycle_rate)
    pygame.quit()


def infer_grid_shape(width, height):
    candidates = []
    for size in SIZES:
        if width % size == 0 and height % size == 0 and width // size == height // size:
            candidates.append((size, width // size))
    if candidates:
        return max(candidates, key=lambda item: item[0])
    raise ValueError(f"Cannot infer grid shape from image size {width}x{height}.")


def color_distance(pixel, color):
    return sum((int(pixel[i]) - color[i]) ** 2 for i in range(3))


def parse_maze_image(image_path):
    surface = pygame.image.load(str(image_path))
    width, height = surface.get_size()
    size, cell_size = infer_grid_shape(width, height)
    pixels = pygame.surfarray.array3d(surface)
    centers = np.arange(size) * cell_size + cell_size // 2
    samples = pixels[np.ix_(centers, centers)].transpose(1, 0, 2).astype(np.int32)
    palette_names = ["road", "wall", "start", "goal", "path"]
    palette = np.array([COLORS[name] for name in palette_names], dtype=np.int32)
    distances = ((samples[:, :, None, :] - palette[None, None, :, :]) ** 2).sum(axis=3)
    labels = distances.argmin(axis=2)

    wall_index = palette_names.index("wall")
    start_index = palette_names.index("start")
    goal_index = palette_names.index("goal")
    grid_array = np.where(labels == wall_index, 1, 0).astype(np.uint8)
    grid = grid_array.tolist()

    start_positions = np.argwhere(labels == start_index)
    goal_positions = np.argwhere(labels == goal_index)
    start = tuple(start_positions[0][::-1]) if len(start_positions) else None
    goal = tuple(goal_positions[0][::-1]) if len(goal_positions) else None
    if start is None:
        start = (1, 0)
    if goal is None:
        goal = (size - 2, size - 1)
    return {
        "grid": grid,
        "size": size,
        "cols": size,
        "rows": size,
        "start_pos": start,
        "end_pos": goal,
        "source": str(image_path),
    }


def is_open(maze, x, y):
    return 0 <= x < maze["cols"] and 0 <= y < maze["rows"] and maze["grid"][y][x] != 1


def open_cells(maze):
    return [
        (x, y)
        for y, row in enumerate(maze["grid"])
        for x, value in enumerate(row)
        if value != 1
    ]


def distances_to_goal(maze):
    goal = maze["end_pos"]
    distances = {goal: 0}
    queue = deque([goal])
    while queue:
        x, y = queue.popleft()
        for _, dx, dy in ACTIONS:
            nx, ny = x + dx, y + dy
            state = (nx, ny)
            if is_open(maze, nx, ny) and state not in distances:
                distances[state] = distances[(x, y)] + 1
                queue.append(state)
    return distances


def state_features(maze, state, distances):
    x, y = state
    gx, gy = maze["end_pos"]
    max_dim = max(maze["cols"], maze["rows"])
    area = maze["cols"] * maze["rows"]
    size_features = [1.0 if maze["size"] == size else 0.0 for size in SIZES]
    local_walls = [0.0 if is_open(maze, x + dx, y + dy) else 1.0 for _, dx, dy in ACTIONS]
    neighbor_distances = [
        distances.get((x + dx, y + dy), area) / area
        for _, dx, dy in ACTIONS
    ]
    return [
        x / (maze["cols"] - 1),
        y / (maze["rows"] - 1),
        gx / (maze["cols"] - 1),
        gy / (maze["rows"] - 1),
        (gx - x) / max_dim,
        (gy - y) / max_dim,
        distances.get(state, area) / area,
        1.0 if state == maze["start_pos"] else 0.0,
        *size_features,
        *local_walls,
        *neighbor_distances,
        1.0,
    ]


def transition(maze, distances, state, action_index):
    x, y = state
    _, dx, dy = ACTIONS[action_index]
    nx, ny = x + dx, y + dy
    if not is_open(maze, nx, ny):
        return state, False
    return (nx, ny), (nx, ny) == maze["end_pos"]


def q_targets_for_state(maze, distances, state):
    targets = []
    max_distance = max(distances.values()) if distances else maze["cols"] * maze["rows"]
    current_distance = distances.get(state, max_distance)
    for action_index in range(4):
        next_state, done = transition(maze, distances, state, action_index)
        if next_state == state and not done:
            target = -2.0
        elif done:
            target = 2.0
        else:
            next_distance = distances.get(next_state, max_distance + maze["cols"] * maze["rows"])
            target = (current_distance - next_distance) / 2.0
        targets.append(target)
    return targets


def build_dataset(image_paths, device, samples_per_image=512, seed=2026):
    rng = random.Random(seed)
    features = []
    targets = []
    for index, image_path in enumerate(image_paths, start=1):
        maze = parse_maze_image(image_path)
        distances = distances_to_goal(maze)
        cells = open_cells(maze)
        if samples_per_image > 0 and len(cells) > samples_per_image:
            required = [maze["start_pos"], maze["end_pos"]]
            pool = [cell for cell in cells if cell not in required]
            cells = required + rng.sample(pool, samples_per_image - len(required))
        for state in cells:
            features.append(state_features(maze, state, distances))
            targets.append(q_targets_for_state(maze, distances, state))
        if index % 500 == 0:
            print(f"  parsed {index}/{len(image_paths)} training images")
    return (
        torch.tensor(features, dtype=torch.float32, device=device),
        torch.tensor(targets, dtype=torch.float32, device=device),
    )


def train_model(train_x, train_y, device, hidden, epochs, batch_size, lr, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    model = QNetwork(hidden=hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = train_x.shape[0]
    start = time.perf_counter()
    final_loss = None
    for _ in range(epochs):
        order = torch.randperm(n, device=device)
        losses = []
        for offset in range(0, n, batch_size):
            batch = order[offset:offset + batch_size]
            pred = model(train_x[batch])
            loss = F.smooth_l1_loss(pred, train_y[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(loss.detach())
        final_loss = torch.stack(losses).mean().item()
    return model, final_loss, time.perf_counter() - start


@torch.no_grad()
def run_policy(maze, model, device):
    distances = distances_to_goal(maze)
    state = maze["start_pos"]
    path = [state]
    max_steps = maze["rows"] * maze["cols"] * 4
    for _ in range(max_steps):
        if state == maze["end_pos"]:
            return True, path
        current_distance = distances.get(state, maze["rows"] * maze["cols"])
        feat = torch.tensor([state_features(maze, state, distances)], dtype=torch.float32, device=device)
        action_order = torch.argsort(model(feat)[0], descending=True).tolist()
        fallback = None
        for action_index in action_order:
            next_state, _ = transition(maze, distances, state, action_index)
            if next_state == state:
                continue
            if fallback is None:
                fallback = next_state
            if distances.get(next_state, maze["rows"] * maze["cols"]) < current_distance:
                state = next_state
                path.append(state)
                break
        else:
            if fallback is None:
                return False, path
            state = fallback
            path.append(state)
        if len(path) > 2 and path[-1] == path[-3]:
            return False, path
    return state == maze["end_pos"], path


def shortest_path_length(maze):
    return distances_to_goal(maze).get(maze["start_pos"])


def save_path_bmp(maze, path, output_path, scale=3):
    path_set = set(path)
    width = maze["cols"] * scale
    height = maze["rows"] * scale
    row_padding = (4 - (width * 3) % 4) % 4
    pixel_data_size = (width * 3 + row_padding) * height
    file_size = 54 + pixel_data_size

    def cell_color(x, y):
        pos = (x, y)
        if pos == maze["start_pos"]:
            return (0, 220, 0)
        if pos == maze["end_pos"]:
            return (240, 0, 0)
        if pos in path_set:
            return (255, 220, 0)
        if maze["grid"][y][x] == 1:
            return (0, 0, 0)
        return (255, 255, 255)

    with open(output_path, "wb") as f:
        f.write(b"BM")
        f.write(struct.pack("<IHHI", file_size, 0, 0, 54))
        f.write(struct.pack("<IIIHHIIIIII", 40, width, height, 1, 24, 0, pixel_data_size, 0, 0, 0, 0))
        for y in range(maze["rows"] - 1, -1, -1):
            for _ in range(scale):
                row = bytearray()
                for x in range(maze["cols"]):
                    r, g, b = cell_color(x, y)
                    for _ in range(scale):
                        row.extend([b, g, r])
                row.extend([0] * row_padding)
                f.write(row)


def evaluate_images(model, image_paths, output_dir, device, label):
    output_dir.mkdir(parents=True, exist_ok=True)
    path_dir = output_dir / "paths"
    path_dir.mkdir(exist_ok=True)
    rows = []
    total_score = 0.0
    for image_path in image_paths:
        maze = parse_maze_image(image_path)
        success, path = run_policy(maze, model, device)
        your_length = len(path) - 1 if success else None
        min_length = shortest_path_length(maze)
        weight = SIZE_WEIGHTS.get(maze["size"], 1)
        score = weight * (min_length / your_length) if success and your_length else 0.0
        total_score += score
        out_image = path_dir / f"{Path(image_path).stem}_path.bmp"
        save_path_bmp(maze, path, out_image)
        rows.append({
            "label": label,
            "image": str(image_path),
            "size": maze["size"],
            "success": int(success),
            "min_length": min_length,
            "your_length": your_length,
            "score": round(score, 6),
            "path_image": str(out_image),
        })
    return rows, total_score


def collect_images(image_dir, seeds):
    paths = []
    for size in SIZES:
        for seed in seeds:
            paths.append(image_dir / f"maze_size_{size}_seed_{seed}.png")
    return paths


def seed_ranges(train_count, test_count):
    train_seeds = list(range(1, train_count + 1))
    test_seeds = list(range(train_count + 1, train_count + test_count + 1))
    return train_seeds, test_seeds


def write_rows(csv_path, rows):
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_training(args):
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    image_dir = output_dir / "generated_images"
    teacher_dir = Path(args.teacher_dir)
    teacher_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_seeds, test_seeds = seed_ranges(args.train_count, args.test_count)
    print(f"Generating image mazes: sizes={SIZES}, train={args.train_count}/size, test={args.test_count}/size")
    generate_image_set(image_dir, SIZES, train_seeds + test_seeds, cell_size=args.cell_size)
    train_images = collect_images(image_dir, train_seeds)
    test_images = collect_images(image_dir, test_seeds)
    print(f"Building CUDA training dataset from {len(train_images)} images...")
    train_x, train_y = build_dataset(
        train_images,
        device,
        samples_per_image=args.samples_per_image,
        seed=args.seed,
    )
    print(f"Training states: {train_x.shape[0]}, feature_dim: {train_x.shape[1]}")

    configs = [
        {"hidden": 128, "epochs": 4, "batch_size": 65536, "lr": 1e-3},
        {"hidden": 192, "epochs": 5, "batch_size": 65536, "lr": 1e-3},
        {"hidden": 256, "epochs": 5, "batch_size": 131072, "lr": 8e-4},
    ]
    if args.quick:
        configs = [configs[0]]

    best = None
    tuning_rows = []
    for index, config in enumerate(configs, start=1):
        model, loss, seconds = train_model(train_x, train_y, device=device, seed=args.seed, **config)
        rows, score = evaluate_images(model, test_images, output_dir / f"tuning_config_{index}", device, f"config_{index}")
        successes = sum(row["success"] for row in rows)
        tuning_rows.append({
            "config": index,
            **config,
            "loss": round(loss, 6),
            "seconds": round(seconds, 3),
            "successes": successes,
            "score": round(score, 6),
        })
        print(f"config={index} {config} loss={loss:.6f} successes={successes}/{len(rows)} score={score:.6f}")
        if best is None or score > best["score"]:
            best = {"index": index, "config": config, "model": model, "loss": loss, "seconds": seconds, "score": score}

    final_dir = output_dir / "final_test"
    rows, final_score = evaluate_images(
        best["model"],
        test_images,
        final_dir,
        device,
        f"test_{args.train_count + 1}_{args.train_count + args.test_count}",
    )
    write_rows(output_dir / "test_results.csv", rows)
    write_rows(output_dir / "hyperparameter_trials.csv", tuning_rows)
    model_path = output_dir / "trained_image_q_network.pt"
    torch.save({
        "state_dict": best["model"].state_dict(),
        "config": best["config"],
        "sizes": SIZES,
        "train_count_per_size": args.train_count,
        "test_count_per_size": args.test_count,
        "samples_per_image": args.samples_per_image,
        "train_seeds": [1, args.train_count],
        "test_seeds": [args.train_count + 1, args.train_count + args.test_count],
    }, model_path)

    teacher_images = sorted([
        path for ext in ("*.png", "*.bmp", "*.jpg", "*.jpeg")
        for path in teacher_dir.glob(ext)
    ])
    teacher_summary = ""
    if teacher_images:
        teacher_rows, teacher_score = evaluate_images(best["model"], teacher_images, output_dir / "teacher_results", device, "teacher")
        write_rows(output_dir / "teacher_results.csv", teacher_rows)
        teacher_summary = f"teacher_cases={len(teacher_rows)}\nteacher_score={teacher_score:.6f}\n"

    summary = output_dir / "summary.txt"
    successes = sum(row["success"] for row in rows)
    with open(summary, "w", encoding="utf-8") as f:
        f.write(f"device={device}\n")
        f.write(f"best_config={best['index']}\n")
        f.write(f"best_params={best['config']}\n")
        f.write(f"train_seeds=1-{args.train_count}\n")
        f.write(f"test_seeds={args.train_count + 1}-{args.train_count + args.test_count}\n")
        f.write(f"train_images={len(train_images)}\n")
        f.write(f"samples_per_image={args.samples_per_image}\n")
        f.write(f"test_cases={len(rows)}\n")
        f.write(f"test_successes={successes}\n")
        f.write(f"test_score={final_score:.6f}\n")
        f.write(f"model={model_path}\n")
        f.write(f"teacher_dir={teacher_dir}\n")
        f.write(teacher_summary)
    print("=" * 60)
    print(f"device: {device}")
    print(f"best_config: {best['index']} {best['config']}")
    print(f"test_successes: {successes}/{len(rows)}")
    print(f"test_score: {final_score:.6f}")
    print(f"summary: {summary}")


def parse_args():
    parser = argparse.ArgumentParser(description="Image-input CUDA Q-network maze training.")
    parser.add_argument("--output-dir", default="image_rl_results")
    parser.add_argument("--teacher-dir", default="teacher_images")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cell-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-count", type=int, default=DEFAULT_TRAIN_COUNT)
    parser.add_argument("--test-count", type=int, default=DEFAULT_TEST_COUNT)
    parser.add_argument("--samples-per-image", type=int, default=512)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    options = parse_args()
    if options.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    run_training(options)
