# 26强化学习大作业

本项目实现了一个基于图片输入的迷宫强化学习求解系统。程序使用 `maze.py` 生成迷宫图片，使用 PyTorch 构建 Q 网络进行训练，并对测试集或指定文件夹中的迷宫图片输出路径结果和评分。

## 项目内容

- `image_rl_maze_solver.py`：主训练程序，负责生成训练图片、解析图片迷宫、训练 Q 网络、测试模型并保存结果。
- `generate_random_mazes.py`：随机生成迷宫图片，默认输出到 `random_mazes` 文件夹。
- `solve_image_mazes.py`：读取文件夹中的迷宫图片，加载训练好的网络，输出路径图片、结果表和总分。
- `maze.py`：迷宫生成逻辑。
- `maze_tkinter.py`：基于 Tkinter 的迷宫可视化程序。
- `teacher_images`：预留给老师提供的迷宫图片。把图片放入该文件夹后，训练程序会自动对这些图片进行求解测试。
- `image_rl_results`：训练输出目录，包含模型、测试结果、调参记录和路径图片。
- `random_mazes`：随机生成的迷宫图片目录。
- `random_maze_solutions`：随机迷宫求解结果目录。
- `report`：大作业报告和报告相关图片。

## 环境要求

建议使用支持 CUDA 的 Python 环境运行训练程序。项目中用到的主要 Python 包如下：

```bash
pip install pygame numpy torch
```

如果使用 Anaconda 环境，先进入对应环境后再安装依赖和运行程序。

## 训练方法

默认训练会对 31、71、101 三种尺寸的迷宫分别生成 3000 张训练图片和 200 张测试图片：

```bash
python image_rl_maze_solver.py --device cuda
```

如果当前机器没有可用 CUDA，可以改用 CPU：

```bash
python image_rl_maze_solver.py --device cpu
```

训练程序会自动尝试以下三组超参数，并选择测试得分最高的模型：

| config | hidden | epochs | batch_size | lr |
| --- | ---: | ---: | ---: | ---: |
| 1 | 128 | 4 | 65536 | 0.001 |
| 2 | 192 | 5 | 65536 | 0.001 |
| 3 | 256 | 5 | 131072 | 0.0008 |

当前训练结果记录在 `image_rl_results/summary.txt`：

```text
device=cuda
best_config=1
best_params={'hidden': 128, 'epochs': 4, 'batch_size': 65536, 'lr': 0.001}
train_seeds=1-3000
test_seeds=3001-3200
train_images=9000
samples_per_image=512
test_cases=600
test_successes=600
test_score=11000.000000
model=image_rl_results\trained_image_q_network.pt
teacher_dir=teacher_images
```

## 随机生成迷宫图片

运行以下命令会随机生成 10 张任意尺寸迷宫图片，并保存到 `random_mazes`：

```bash
python generate_random_mazes.py
```

也可以指定数量和输出目录：

```bash
python generate_random_mazes.py --count 30 --output-dir random_mazes
```

生成的图片文件名格式为：

```text
分辨率-种子.png
```

例如：

```text
31-1553060734.png
71-894864710.png
101-325302735.png
```

## 求解图片迷宫

先确保 `image_rl_results/trained_image_q_network.pt` 已存在，然后运行：

```bash
python solve_image_mazes.py --input-dir random_mazes --output-dir random_maze_solutions --model image_rl_results/trained_image_q_network.pt --device cuda
```

程序会输出：

- `random_maze_solutions/results.csv`：每张图片的求解结果、最短路径长度、模型路径长度和单图得分。
- `random_maze_solutions/summary.txt`：总测试数量、成功数量和总分。
- `random_maze_solutions/paths`：带路径标注的结果图片。

当前随机迷宫测试结果记录在 `random_maze_solutions/summary.txt`：

```text
device=cuda
model=image_rl_results\trained_image_q_network.pt
model_config={'hidden': 128, 'epochs': 4, 'batch_size': 65536, 'lr': 0.001}
input_dir=random_mazes
cases=30
successes=30
score=520.000000
max_score=520.000000
results_csv=random_maze_solutions\results.csv
path_images=random_maze_solutions\paths
```

## 评分方式

程序会先从图片中识别迷宫结构，再由训练后的 Q 网络给出动作选择。测试时会计算：

- 是否成功到达终点；
- 模型路径长度；
- 迷宫最短路径长度；
- 单图得分；
- 总得分。

评分公式写在 `image_rl_maze_solver.py` 的 `evaluate_images` 函数中：

```text
score = size_weight * (shortest_path_length / model_path_length)
```

当模型走出的路径等于最短路径时，该图片得到对应尺寸的满分权重。

## 老师图片的使用方式

把老师给出的迷宫图片放入：

```text
teacher_images
```

然后重新运行训练程序：

```bash
python image_rl_maze_solver.py --device cuda
```

如果 `teacher_images` 中存在 `png`、`bmp`、`jpg` 或 `jpeg` 图片，程序会自动生成：

- `image_rl_results/teacher_results.csv`
- `image_rl_results/teacher_results`

## 注意事项

- 图片迷宫使用颜色识别：白色表示道路，黑色表示墙，绿色表示起点，红色表示终点。
- 默认支持的迷宫尺寸为 `31`、`71`、`101`。
- 如果在没有 CUDA 的环境中运行 `--device cuda`，程序会报错；此时需要改用 `--device cpu`。
