"""
游戏核心逻辑 - 支持C++加速
"""

import numpy as np
from copy import deepcopy

# 尝试导入C++版本
try:
    from hex_cpp import GameState as GameStateCPP
    from hex_cpp import fast_rollout
    CPP_GAME_AVAILABLE = True
    print("✓ C++ GameState模块已加载")
except ImportError as e:
    CPP_GAME_AVAILABLE = False
    print(f"✗ C++ GameState模块不可用，使用Python版本")


class UnionFind:
    """并查集数据结构"""

    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            return x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            self.parent[root_x] = root_y
            return True
        return False


class GameState:
    """
    海克斯棋游戏状态
    可以使用C++后端加速或纯Python实现
    """

    EMPTY = 0
    RED = 1
    BLUE = 2
    EDGE1 = -1  # 边界1
    EDGE2 = -2  # 边界2

    def __init__(self, size=11, use_cpp=True):
        """
        初始化游戏状态

        Args:
            size: 棋盘大小
            use_cpp: 是否使用C++后端
        """
        self.size = size
        self.use_cpp = use_cpp and CPP_GAME_AVAILABLE

        if self.use_cpp:
            # 使用C++后端
            self._cpp_state = GameStateCPP(size)
            self.toplay = self._cpp_state.turn()
        else:
            # 使用Python实现
            self.board = np.zeros((size, size), dtype=int)
            self.toplay = self.RED
            self.red_groups = UnionFind()
            self.blue_groups = UnionFind()

    @property
    def board(self):
        """获取棋盘状态"""
        if self.use_cpp:
            flat_board = self._cpp_state.get_board()
            return np.array(flat_board).reshape(self.size, self.size)
        else:
            return self._board

    @board.setter
    def board(self, value):
        """设置棋盘状态（仅Python模式）"""
        if not self.use_cpp:
            self._board = value

    def copy(self):
        """复制游戏状态"""
        if self.use_cpp:
            new_state = GameState(self.size, use_cpp=True)
            new_state._cpp_state = GameStateCPP(self._cpp_state)
            new_state.toplay = self._cpp_state.turn()
            return new_state
        else:
            new_state = GameState(self.size, use_cpp=False)
            new_state.board = deepcopy(self.board)
            new_state.toplay = self.toplay
            new_state.red_groups = deepcopy(self.red_groups)
            new_state.blue_groups = deepcopy(self.blue_groups)
            return new_state

    def make_move(self, move):
        """
        执行一步棋

        Args:
            move: (row, col) 坐标
        """
        if self.use_cpp:
            self._cpp_state.play(move[0], move[1])
            self.toplay = self._cpp_state.turn()
        else:
            self._make_move_python(move)

    def _make_move_python(self, move):
        """Python版本的落子"""
        x, y = move

        if self.board[x][y] != self.EMPTY:
            raise ValueError(f"位置 ({x}, {y}) 已被占用")

        self.board[x][y] = self.toplay

        if self.toplay == self.RED:
            self._place_red(x, y)
        else:
            self._place_blue(x, y)

        self.toplay = 3 - self.toplay

    def _place_red(self, x, y):
        """放置红棋并更新连通性"""
        # 连接到边界
        if y == 0:
            self.red_groups.union(self.EDGE1, (x, y))
        if y == self.size - 1:
            self.red_groups.union(self.EDGE2, (x, y))

        # 连接到相邻的红棋
        for nx, ny in self.neighbors(x, y):
            if self.board[nx][ny] == self.RED:
                self.red_groups.union((x, y), (nx, ny))

    def _place_blue(self, x, y):
        """放置蓝棋并更新连通性"""
        # 连接到边界
        if x == 0:
            self.blue_groups.union(self.EDGE1, (x, y))
        if x == self.size - 1:
            self.blue_groups.union(self.EDGE2, (x, y))

        # 连接到相邻的蓝棋
        for nx, ny in self.neighbors(x, y):
            if self.board[nx][ny] == self.BLUE:
                self.blue_groups.union((x, y), (nx, ny))

    def neighbors(self, x, y):
        """获取六边形格子的邻居"""
        directions = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]
        result = []

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                result.append((nx, ny))

        return result

    def winner(self):
        """
        检查游戏胜者

        Returns:
            0: 未结束
            1: 红方胜
            2: 蓝方胜
        """
        if self.use_cpp:
            return self._cpp_state.winner()
        else:
            return self._winner_python()

    def _winner_python(self):
        """Python版本的胜负判断"""
        # 检查红方是否连接左右边界
        if self.red_groups.find(self.EDGE1) == self.red_groups.find(self.EDGE2):
            return self.RED

        # 检查蓝方是否连接上下边界
        if self.blue_groups.find(self.EDGE1) == self.blue_groups.find(self.EDGE2):
            return self.BLUE

        return 0

    def get_legal_moves(self):
        """获取所有合法走法"""
        if self.use_cpp:
            return self._cpp_state.moves()
        else:
            moves = []
            for i in range(self.size):
                for j in range(self.size):
                    if self.board[i][j] == self.EMPTY:
                        moves.append((i, j))
            return moves

    def random_playout(self):
        """
        随机模拟到游戏结束

        Returns:
            胜者 (1=红方, 2=蓝方)
        """
        if self.use_cpp:
            # 使用C++的快速rollout
            return fast_rollout(self._cpp_state)
        else:
            return self._random_playout_python()

    def _random_playout_python(self):
        """Python版本的随机模拟"""
        import random

        temp_state = self.copy()

        while temp_state.winner() == 0:
            moves = temp_state.get_legal_moves()
            if not moves:
                break

            move = random.choice(moves)
            temp_state.make_move(move)

        return temp_state.winner()
