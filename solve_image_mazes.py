import argparse
import csv
from pathlib import Path

import torch

from image_rl_maze_solver import (
    QNetwork,
    SIZE_WEIGHTS,
    evaluate_images,
)


IMAGE_PATTERNS = ("*.png", "*.bmp", "*.jpg", "*.jpeg")


def collect_images(input_dir):
    paths = []
    for pattern in IMAGE_PATTERNS:
        paths.extend(Path(input_dir).glob(pattern))
    return sorted(paths)


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    config = checkpoint.get("config", {})
    hidden = config.get("hidden", 128)
    model = QNetwork(hidden=hidden).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, config


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Solve all maze images in a folder using the trained CUDA Q network.")
    parser.add_argument("--input-dir", default="random_mazes")
    parser.add_argument("--output-dir", default="random_maze_solutions")
    parser.add_argument("--model", default="image_rl_results/trained_image_q_network.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    model_path = Path(args.model)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")
    images = collect_images(input_dir)
    if not images:
        raise FileNotFoundError(f"No images found in: {input_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    device = torch.device(args.device)
    model, config = load_model(model_path, device)
    rows, total_score = evaluate_images(model, images, output_dir, device, "random_images")
    write_csv(output_dir / "results.csv", rows)

    successes = sum(row["success"] for row in rows)
    max_score = sum(SIZE_WEIGHTS.get(row["size"], 1) for row in rows)
    with open(output_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"device={device}\n")
        f.write(f"model={model_path}\n")
        f.write(f"model_config={config}\n")
        f.write(f"input_dir={input_dir}\n")
        f.write(f"cases={len(rows)}\n")
        f.write(f"successes={successes}\n")
        f.write(f"score={total_score:.6f}\n")
        f.write(f"max_score={max_score:.6f}\n")
        f.write(f"results_csv={output_dir / 'results.csv'}\n")
        f.write(f"path_images={output_dir / 'paths'}\n")

    print("=" * 60)
    print(f"device={device}")
    print(f"input_dir={input_dir}")
    print(f"cases={len(rows)}")
    print(f"successes={successes}/{len(rows)}")
    print(f"score={total_score:.6f}/{max_score:.6f}")
    print(f"results_csv={output_dir / 'results.csv'}")
    print(f"path_images={output_dir / 'paths'}")


if __name__ == "__main__":
    main()
