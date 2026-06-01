import random
import sys

try:
    import pygame
except ImportError:
    pygame = None

# 全局配置参数
CELL_SIZE = 12          # 每个格子的像素大小
WHITE = (255, 255, 255) # 路
BLACK = (0, 0, 0)       # 墙
GREEN = (0, 255, 0)     # 起点
RED = (255, 0, 0)       # 终点
BLUE = (0, 120, 255)    # 背景
YELLOW = (255, 255, 0)  # 路径

# 迷宫生成器 (逻辑层)
class MazeGenerator:
    def __init__(self, width_cells, height_cells):
        # 算法强制要求奇数尺寸。如果输入30，这里会自动修正为31。如果输入70，这里会自动修正为71。如果输入100，这里会自动修正为101。
        self.cols = width_cells if width_cells % 2 != 0 else width_cells + 1
        self.rows = height_cells if height_cells % 2 != 0 else height_cells + 1
        self.grid = [[1 for _ in range(self.cols)] for _ in range(self.rows)]
        self.start_pos = (1, 1)
        self.end_pos = (self.cols - 2, self.rows - 2)
        self.agent_pos = self.start_pos
        self.path = [self.start_pos]

    def generate(self, cycle_rate=0.1, seed=None):
        if seed is not None:
            print(f"使用固定种子: {seed}")
            random.seed(seed)
        else:
            print("使用完全随机模式")
            random.seed() 

        print(f"正在生成 {self.cols}x{self.rows} 的迷宫")
        self._dfs_perfect_maze()
        self._add_cycles(cycle_rate)
        self._set_entrance_exit()

    def _dfs_perfect_maze(self):
        sx, sy = 1, 1
        self.grid[sy][sx] = 0
        stack = [(sx, sy)]
        while stack:
            cx, cy = stack[-1]
            neighbors = []
            directions = [(0, -2), (0, 2), (-2, 0), (2, 0)]
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if 1 <= nx < self.cols - 1 and 1 <= ny < self.rows - 1 and self.grid[ny][nx] == 1:
                    neighbors.append((nx, ny, dx, dy))
            if neighbors:
                nx, ny, dx, dy = random.choice(neighbors)
                self.grid[ny][nx] = 0
                self.grid[cy + dy // 2][cx + dx // 2] = 0
                stack.append((nx, ny))
            else:
                stack.pop()

    def _add_cycles(self, rate):
        candidates = []
        for r in range(1, self.rows - 1):
            for c in range(1, self.cols - 1):
                if self.grid[r][c] == 1:
                    if (self.grid[r-1][c]==0 and self.grid[r+1][c]==0) or \
                       (self.grid[r][c-1]==0 and self.grid[r][c+1]==0):
                        candidates.append((r, c))
        if candidates:
            k = int(len(candidates) * rate)
            for r, c in random.sample(candidates, k):
                self.grid[r][c] = 0

    def _set_entrance_exit(self):
        self.start_pos = (1, 0)
        self.end_pos = (self.cols - 2, self.rows - 1)
        self.grid[self.start_pos[1]][self.start_pos[0]] = 0
        self.grid[self.end_pos[1]][self.end_pos[0]] = 0
        self.reset()

    def reset(self):
        self.agent_pos = self.start_pos
        self.path = [self.start_pos]
        return self.agent_pos

    def _is_valid_position(self, x, y):
        return 0 <= x < self.cols and 0 <= y < self.rows and self.grid[y][x] != 1
    
    def move(self, x, y=None):
        if y is None:
            action_map = {
                0: (0, -1), "up": (0, -1), "UP": (0, -1),
                1: (0, 1), "down": (0, 1), "DOWN": (0, 1),
                2: (-1, 0), "left": (-1, 0), "LEFT": (-1, 0),
                3: (1, 0), "right": (1, 0), "RIGHT": (1, 0),
            }
            if x not in action_map:
                raise ValueError(f"未知动作: {x}")
            dx, dy = action_map[x]
        else:
            dx, dy = x, y

        cx, cy = self.agent_pos
        nx, ny = cx + dx, cy + dy
        if not self._is_valid_position(nx, ny):
            return self.agent_pos, -5, False, {"valid": False}

        self.agent_pos = (nx, ny)
        self.path.append(self.agent_pos)
        done = self.is_end()
        reward = 100 if done else -1
        if not done and self.grid[ny][nx] == 0:
            self.grid[ny][nx] = 2
        return self.agent_pos, reward, done, {"valid": True}

    def is_end(self):
        return self.agent_pos == self.end_pos


# 迷宫渲染器 (显示层)
class MazeRenderer:
    def __init__(self, maze_generator):
        if pygame is None:
            raise RuntimeError("当前 Python 环境未安装 pygame，请使用训练脚本或改用 maze_tkinter.py。")
        self.maze = maze_generator
        self.width = self.maze.cols * CELL_SIZE
        self.height = self.maze.rows * CELL_SIZE
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(f"迷宫 (Seed: {MY_SEED})")
        self.clock = pygame.time.Clock()

    def draw(self):
        self.screen.fill(BLUE)
        for r in range(self.maze.rows):
            for c in range(self.maze.cols):
                # 0是路(白)，1是墙(黑)
                if self.maze.grid[r][c] == 1:
                    color = BLACK 
                elif self.maze.grid[r][c] == 0:
                    color = WHITE
                elif self.maze.grid[r][c] == 2: # 2是路径(黄),可以自己定义修改
                    color = YELLOW
                pygame.draw.rect(self.screen, color, (c*CELL_SIZE, r*CELL_SIZE, CELL_SIZE, CELL_SIZE))
        
        # 绘制起点终点
        sx, sy = self.maze.start_pos
        ex, ey = self.maze.end_pos
        pygame.draw.rect(self.screen, GREEN, (sx*CELL_SIZE, sy*CELL_SIZE, CELL_SIZE, CELL_SIZE))
        pygame.draw.rect(self.screen, RED, (ex*CELL_SIZE, ey*CELL_SIZE, CELL_SIZE, CELL_SIZE))
        pygame.display.flip()

    def run(self):
        print("窗口已启动")
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            self.draw()
            self.clock.tick(30)
    


# 修改区域
if __name__ == "__main__":

    # 修改这里的 MY_SEED 控制迷宫形状
    # 1. 如果填入整数 (例如 9527或 1等整数)，每次运行迷宫都一样。
    # 2. 如果填入 None (注意 N 大写)，每次运行迷宫都不同
    MY_SEED = 1
    
    # 【地图尺寸设置】
    # 虽然题目要求 30，但为了算法逻辑闭合（四周有墙），我们通常传入奇数。如果传入 30，代码也会自动变成 31。
    # 其他尺寸同理，可以修改设置width_cells和height_cells为71或101
    maze = MazeGenerator(width_cells=31, height_cells=31)
    
    # 生成迷宫
    maze.generate(cycle_rate=0.03, seed=MY_SEED)
    
    # 迷宫数据存储在maze.grid中，maze.gird是一个二维列表，0代表可通行，1代表不可通行
    # 使用数字2代表你的智能体所走过的路。
    # 你需要补全迷宫的方法，move()和is_end()需要自己实现，move()智能体移动并判断移动合法性，is_end()判断是否到达终点游戏是否结束。此外，可以根据自己的需求添加其他的方法

    renderer = MazeRenderer(maze)
    renderer.run()
