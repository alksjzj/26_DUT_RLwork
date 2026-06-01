import argparse
import random
from pathlib import Path

from image_rl_maze_solver import SIZES, generate_maze_image


def parse_args():
    parser = argparse.ArgumentParser(description="Generate random maze images with maze.py logic.")
    parser.add_argument("--output-dir", default="random_mazes")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--cell-size", type=int, default=4)
    parser.add_argument("--cycle-rate", type=float, default=0.03)
    parser.add_argument("--sizes", nargs="*", type=int, default=SIZES)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    used = set()
    rows = []
    while len(rows) < args.count:
        size = random.choice(args.sizes)
        seed = random.randint(1, 2_147_483_647)
        key = (size, seed)
        if key in used:
            continue
        used.add(key)
        path = output_dir / f"{size}-{seed}.png"
        generate_maze_image(
            size=size,
            seed=seed,
            output_path=path,
            cell_size=args.cell_size,
            cycle_rate=args.cycle_rate,
        )
        rows.append((size, seed, path))
        print(f"generated: size={size} seed={seed} file={path}")

    print("=" * 60)
    print(f"generated_count={len(rows)}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
