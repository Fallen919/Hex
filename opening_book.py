"""
海克斯棋开局库
提供开局走法推荐和开局理论
"""

import random
import json
import os


class OpeningBook:
    """海克斯棋开局库"""

    def __init__(self, board_size=11):
        self.board_size = board_size
        self.opening_moves = {}
        self.max_opening_moves = 3  # 开局库覆盖的最大手数
        self._initialize_openings()

    def _initialize_openings(self):
        """初始化开局库"""
        if self.board_size == 11:
            self._init_11x11_openings()
        elif self.board_size == 13:
            self._init_13x13_openings()
        elif self.board_size == 9:
            self._init_9x9_openings()
        else:
            # 其他尺寸使用通用开局
            self._init_general_openings()

    def _init_11x11_openings(self):
        """初始化11x11棋盘的开局库"""
        center = 5  # 中心点

        self.opening_moves = {
            # 第1手 (红方第1手)
            0: {
                'best_moves': [
                    (center, center),  # 天元，最强开局
                    (center, center + 1),  # 偏右
                    (center + 1, center),  # 偏下
                    (center - 1, center),  # 偏上
                    (center, center - 1),  # 偏左
                ],
                'weights': [0.5, 0.15, 0.15, 0.1, 0.1],
                'description': '红方第1手，天元是最强开局'
            },

            # 第2手 (蓝方第1手，对应红方第1手)
            1: {
                'strategies': {
                    # 应对天元开局
                    'center_response': {
                        'condition': lambda moves: self._is_center_area(moves[-1]),
                        'moves': [
                            (center - 1, center - 1),  # 对角
                            (center + 1, center + 1),  # 对角
                            (center - 2, center + 1),  # 桥位
                            (center + 1, center - 2),  # 桥位
                            (center, center - 2),  # 边缘呼应
                            (center - 2, center),  # 边缘呼应
                        ],
                        'weights': [0.25, 0.25, 0.15, 0.15, 0.1, 0.1],
                        'description': '应对中心开局，占据战略要点'
                    },

                    # 应对边角开局
                    'edge_response': {
                        'condition': lambda moves: self._is_edge_area(moves[-1]),
                        'moves': [
                            (center, center),  # 抢占中心
                            (center - 1, center),
                            (center + 1, center),
                            (center, center - 1),
                            (center, center + 1),
                        ],
                        'weights': [0.6, 0.15, 0.15, 0.05, 0.05],
                        'description': '应对边角开局，争夺中心'
                    },

                    # 默认策略
                    'default': {
                        'moves': [
                            (center, center),
                            (center - 1, center + 1),
                            (center + 1, center - 1),
                        ],
                        'weights': [0.5, 0.25, 0.25],
                        'description': '默认应对策略'
                    }
                }
            },

            # 第3手 (红方第2手)
            2: {
                'strategies': {
                    # 构建连接结构
                    'build_connection': {
                        'condition': lambda moves: True,
                        'moves': [
                            (center - 1, center + 1),
                            (center + 1, center - 1),
                            (center - 2, center),
                            (center, center - 2),
                            (center + 2, center),
                            (center, center + 2),
                        ],
                        'weights': [0.2, 0.2, 0.15, 0.15, 0.15, 0.15],
                        'description': '构建连接结构，扩大势力范围'
                    }
                }
            }
        }

    def _init_13x13_openings(self):
        """初始化13x13棋盘的开局库"""
        center = 6

        self.opening_moves = {
            0: {
                'best_moves': [(center, center)],
                'weights': [1.0],
                'description': '13x13棋盘，天元开局'
            }
        }

    def _init_9x9_openings(self):
        """初始化9x9棋盘的开局库"""
        center = 4

        self.opening_moves = {
            0: {
                'best_moves': [(center, center)],
                'weights': [1.0],
                'description': '9x9棋盘，天元开局'
            }
        }

    def _init_general_openings(self):
        """初始化通用开局库"""
        center = self.board_size // 2

        self.opening_moves = {
            0: {
                'best_moves': [(center, center)],
                'weights': [1.0],
                'description': '通用开局，占据中心'
            }
        }

    def _is_center_area(self, move):
        """判断是否在中心区域"""
        center = self.board_size // 2
        x, y = move
        # 中心3x3区域
        return abs(x - center) <= 1 and abs(y - center) <= 1

    def _is_edge_area(self, move):
        """判断是否在边缘区域"""
        x, y = move
        edge_size = 2
        return (x < edge_size or x >= self.board_size - edge_size or
                y < edge_size or y >= self.board_size - edge_size)

    def _is_corner_area(self, move):
        """判断是否在角落区域"""
        x, y = move
        corner_size = 3
        return ((x < corner_size and y < corner_size) or
                (x < corner_size and y >= self.board_size - corner_size) or
                (x >= self.board_size - corner_size and y < corner_size) or
                (x >= self.board_size - corner_size and y >= self.board_size - corner_size))

    def get_opening_move(self, game_state, red_move_count):
        """
        获取开局走法推荐

        Args:
            game_state: 当前游戏状态
            red_move_count: 红方已走步数

        Returns:
            推荐的走法 (x, y) 或 None
        """
        # 检查是否是红方回合
        if game_state.toplay != 1:
            return None

        # 检查是否超出开局库范围
        if red_move_count >= self.max_opening_moves:
            return None

        # 总走法数 = 红方走法数 * 2 (如果蓝方也走了的话)
        total_moves = red_move_count * 2
        if red_move_count > 0:
            total_moves -= 1

        # 获取已走的所有棋步
        all_moves = []
        for x in range(self.board_size):
            for y in range(self.board_size):
                if game_state.board[x, y] != 0:
                    all_moves.append((x, y))

        # 根据走法数选择策略
        if red_move_count not in self.opening_moves:
            return None

        opening_data = self.opening_moves[red_move_count]

        # 简单开局（直接给出最佳走法列表）
        if 'best_moves' in opening_data:
            moves = opening_data['best_moves']
            weights = opening_data['weights']

            # 过滤已占用的位置
            available_moves = []
            available_weights = []
            for move, weight in zip(moves, weights):
                if game_state.board[move] == 0:
                    available_moves.append(move)
                    available_weights.append(weight)

            if available_moves:
                # 归一化权重
                total_weight = sum(available_weights)
                normalized_weights = [w / total_weight for w in available_weights]

                # 根据权重随机选择
                return random.choices(available_moves, weights=normalized_weights)[0]

        # 策略性开局（根据对手走法选择）
        elif 'strategies' in opening_data:
            strategies = opening_data['strategies']

            # 尝试匹配策略
            for strategy_name, strategy in strategies.items():
                if strategy_name == 'default':
                    continue

                condition = strategy.get('condition')
                if condition and condition(all_moves):
                    moves = strategy['moves']
                    weights = strategy['weights']

                    # 过滤已占用的位置
                    available_moves = []
                    available_weights = []
                    for move, weight in zip(moves, weights):
                        if game_state.board[move] == 0:
                            available_moves.append(move)
                            available_weights.append(weight)

                    if available_moves:
                        total_weight = sum(available_weights)
                        normalized_weights = [w / total_weight for w in available_weights]
                        return random.choices(available_moves, weights=normalized_weights)[0]

            # 如果没有匹配的策略，使用默认策略
            if 'default' in strategies:
                default_strategy = strategies['default']
                moves = default_strategy['moves']
                weights = default_strategy['weights']

                available_moves = []
                available_weights = []
                for move, weight in zip(moves, weights):
                    if game_state.board[move] == 0:
                        available_moves.append(move)
                        available_weights.append(weight)

                if available_moves:
                    total_weight = sum(available_weights)
                    normalized_weights = [w / total_weight for w in available_weights]
                    return random.choices(available_moves, weights=normalized_weights)[0]

        return None

    def is_opening_phase(self, game_state, red_move_count):
        """判断是否在开局阶段"""
        return (game_state.toplay == 1 and
                red_move_count < self.max_opening_moves)

    def get_opening_description(self, red_move_count):
        """获取开局描述"""
        if red_move_count in self.opening_moves:
            opening_data = self.opening_moves[red_move_count]
            return opening_data.get('description', '无描述')
        return None

    def save_opening_book(self, filename):
        """保存开局库到文件"""
        # 将开局库转换为可序列化的格式
        serializable_data = {
            'board_size': self.board_size,
            'max_opening_moves': self.max_opening_moves,
            'openings': {}
        }

        for move_num, data in self.opening_moves.items():
            serializable_data['openings'][str(move_num)] = {
                'best_moves': data.get('best_moves', []),
                'weights': data.get('weights', []),
                'description': data.get('description', '')
            }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, ensure_ascii=False, indent=2)

    def load_opening_book(self, filename):
        """从文件加载开局库"""
        if not os.path.exists(filename):
            print(f"开局库文件不存在: {filename}")
            return False

        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.board_size = data.get('board_size', self.board_size)
            self.max_opening_moves = data.get('max_opening_moves', 3)

            # 重建开局库
            self.opening_moves = {}
            openings = data.get('openings', {})

            for move_num_str, opening_data in openings.items():
                move_num = int(move_num_str)
                self.opening_moves[move_num] = {
                    'best_moves': [tuple(m) for m in opening_data.get('best_moves', [])],
                    'weights': opening_data.get('weights', []),
                    'description': opening_data.get('description', '')
                }

            return True

        except Exception as e:
            print(f"加载开局库失败: {e}")
            return False

    def add_opening_sequence(self, sequence, weight=1.0, description=""):
        """
        添加开局序列

        Args:
            sequence: 走法序列 [(x1, y1), (x2, y2), ...]
            weight: 权重
            description: 描述
        """
        # 简单实现：只添加第一手
        if len(sequence) > 0 and 0 in self.opening_moves:
            move = sequence[0]
            if 'best_moves' in self.opening_moves[0]:
                if move not in self.opening_moves[0]['best_moves']:
                    self.opening_moves[0]['best_moves'].append(move)
                    self.opening_moves[0]['weights'].append(weight)


# 测试代码
if __name__ == "__main__":
    from game_core import GameState

    # 创建开局库
    book = OpeningBook(11)

    # 创建游戏状态
    game = GameState(11)

    # 测试获取开局走法
    print("第1手推荐:")
    move1 = book.get_opening_move(game, 0)
    print(f"  推荐走法: {move1}")
    print(f"  描述: {book.get_opening_description(0)}")

    # 模拟走棋
    if move1:
        game.play(move1)
        print(f"\n红方走 {move1}")

    # 测试第2手
    game.set_turn(2)  # 切换到蓝方
    print("\n第2手推荐:")
    # 注意：开局库主要为红方设计

    # 保存开局库
    book.save_opening_book('opening_book_11x11.json')
    print("\n开局库已保存")

    # 加载开局库
    new_book = OpeningBook(11)
    if new_book.load_opening_book('opening_book_11x11.json'):
        print("开局库加载成功")
