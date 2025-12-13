"""
MCTS代理 - 整合核心算法和C++加速
"""

import math
import random
import time
from copy import deepcopy
import numpy as np

# 尝试导入C++快速rollout
try:
    from hex_cpp import fast_rollout
    CPP_ROLLOUT_AVAILABLE = True
    print("✓ C++ fast_rollout已加载")
except ImportError:
    CPP_ROLLOUT_AVAILABLE = False
    print("✗ C++ fast_rollout不可用，使用Python版本")


class Node:
    """MCTS节点"""

    def __init__(self, move, parent, state):
        self.move = move
        self.parent = parent
        self.state = state  # ✓ 保存状态对象
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = state.get_legal_moves()
        self.player = state.toplay

    def select_child(self, exploration=1.41):
        """使用UCB1选择子节点"""
        best_score = -float('inf')
        best_child = None

        for child in self.children:
            if child.visits == 0:
                return child

            exploit = child.wins / child.visits
            explore = exploration * math.sqrt(math.log(self.visits) / child.visits)
            score = exploit + explore

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def add_child(self, move, state):
        """添加子节点"""
        child = Node(move, self, state)
        self.untried_moves.remove(move)
        self.children.append(child)
        return child

    def update(self, result):
        """反向传播更新"""
        self.visits += 1
        if result == self.player:
            self.wins += 1

    def best_child(self):
        """返回访问次数最多的子节点"""
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.visits)

    def explore(self):
        """执行一次MCTS迭代"""
        state = self._get_state()
        node = self

        # Selection - 选择
        while node.untried_moves == [] and node.children != []:
            node = node.select_child()
            state.make_move(node.move)

        # Expansion - 扩展
        if node.untried_moves:
            move = random.choice(node.untried_moves)
            state.make_move(move)
            node = node.add_child(move, state)

        # Simulation - 模拟（使用C++加速）
        if CPP_ROLLOUT_AVAILABLE and hasattr(state, '_cpp_state'):
            # 使用C++快速rollout
            winner = fast_rollout(state._cpp_state)
        else:
            # 使用Python版本
            winner = state.random_playout()

        # Backpropagation - 反向传播
        while node is not None:
            node.update(winner)
            node = node.parent

    def update_rave(self, move, winner):
        """
        更新RAVE (Rapid Action Value Estimation) 统计

        Args:
            move: 走法 (x, y)
            winner: 获胜方 (1 或 2)
        """
        # 确保RAVE字典已初始化
        if not hasattr(self, 'rave_wins'):
            self.rave_wins = {}
        if not hasattr(self, 'rave_visits'):
            self.rave_visits = {}

        # 为这个走法初始化统计
        if move not in self.rave_visits:
            self.rave_visits[move] = 0
            self.rave_wins[move] = 0

        # 更新访问次数
        self.rave_visits[move] += 1

        # 如果当前玩家获胜，更新胜利次数
        if winner == self.player:
            self.rave_wins[move] += 1

    def _get_state(self):
        """重建当前节点的游戏状态"""
        moves = []
        node = self

        while node.parent is not None:
            moves.append(node.move)
            node = node.parent

        moves.reverse()

        # 从根节点重建状态
        from game_core import GameState
        state = GameState(size=11, use_cpp=CPP_ROLLOUT_AVAILABLE)

        for move in moves:
            state.make_move(move)

        return state


class RAVENode(Node):
    """RAVE增强的MCTS节点"""

    def __init__(self, move, parent, state):
        super().__init__(move, parent, state)
        self.rave_wins = {}  # ✓ 初始化为字典
        self.rave_visits = {}  # ✓ 初始化为字典
        self.amaf_wins = {}  # All-Moves-As-First统计
        self.amaf_visits = {}

    def select_child(self, exploration=1.41, rave_constant=300):
        """使用RAVE增强的UCB选择"""
        best_score = -float('inf')
        best_child = None

        for child in self.children:
            if child.visits == 0:
                return child

            # 计算beta值 - 控制RAVE和UCT的权重
            beta = math.sqrt(rave_constant / (3 * self.visits + rave_constant))

            # UCB值
            uct = child.wins / child.visits
            uct += exploration * math.sqrt(math.log(self.visits) / child.visits)

            # RAVE值
            rave = 0
            if child.move in child.rave_visits and child.rave_visits[child.move] > 0:
                rave = child.rave_wins[child.move] / child.rave_visits[child.move]

            # 组合UCB和RAVE
            score = (1 - beta) * uct + beta * rave

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def update(self, result):
        """更新节点统计"""
        super().update(result)

    def update_rave(self, move, winner):
        """
        更新RAVE统计

        Args:
            move: 走法 (x, y)
            winner: 获胜方 (1 或 2)
        """
        # 为所有子节点更新RAVE统计
        for child in self.children:
            if child.move == move:
                # 确保字典已初始化
                if not hasattr(child, 'rave_wins'):
                    child.rave_wins = {}
                if not hasattr(child, 'rave_visits'):
                    child.rave_visits = {}

                # 初始化该走法的统计
                if move not in child.rave_visits:
                    child.rave_visits[move] = 0
                    child.rave_wins[move] = 0

                # 更新统计
                child.rave_visits[move] += 1
                if winner == child.player:
                    child.rave_wins[move] += 1

    def explore(self):
        """RAVE版本的MCTS迭代"""
        state = self._get_state()
        node = self
        moves_played = []

        # Selection
        while node.untried_moves == [] and node.children != []:
            node = node.select_child()
            state.make_move(node.move)
            moves_played.append(node.move)

        # Expansion
        if node.untried_moves:
            move = random.choice(node.untried_moves)
            state.make_move(move)
            moves_played.append(move)
            node = node.add_child(move, state)

        # Simulation - 使用C++加速
        playout_moves = []
        if CPP_ROLLOUT_AVAILABLE and hasattr(state, '_cpp_state'):
            winner = fast_rollout(state._cpp_state)
            # C++版本不返回具体走法，只返回结果
        else:
            # Python版本可以记录走法
            temp_state = state.copy()
            while temp_state.winner() == 0:
                legal_moves = temp_state.get_legal_moves()
                if not legal_moves:
                    break
                move = random.choice(legal_moves)
                playout_moves.append(move)
                temp_state.make_move(move)
            winner = temp_state.winner()

        all_moves = moves_played + playout_moves

        # Backpropagation with RAVE
        while node is not None:
            node.update(winner)

            # 更新RAVE统计
            for move in all_moves:
                node.update_rave(move, winner)

            node = node.parent


class MCTSAgent:
    """UCT MCTS代理"""

    def __init__(self):
        self.gamestate = None
        self.name = "UCT"

    def set_gamestate(self, gamestate):
        """设置游戏状态"""
        self.gamestate = gamestate

    def best_move(self, think_time=1):
        """
        获取最佳走法

        Args:
            think_time: 思考时间(秒)

        Returns:
            最佳走法坐标 (row, col)
        """
        if self.gamestate is None:
            raise ValueError("游戏状态未设置")

        # 检查开局库
        if hasattr(self, 'opening_book') and hasattr(self, 'red_move_count'):
            if self.gamestate.toplay == 1 and self.red_move_count < 3:
                move = self.opening_book.get_opening_move(
                    self.gamestate,
                    self.red_move_count
                )
                if move:
                    print(f"使用开局库: 第{self.red_move_count + 1}步")
                    self.red_move_count += 1
                    return move

            if self.gamestate.toplay == 1:
                self.red_move_count += 1

        # 如果只剩一步，直接返回
        legal_moves = self.gamestate.get_legal_moves()
        if len(legal_moves) == 1:
            return legal_moves[0]

        # 使用MCTS搜索
        return self._mcts_search(think_time)

    def _mcts_search(self, think_time):
        """执行MCTS搜索"""
        start_time = time.time()

        root = Node(None, None, self.gamestate)

        iterations = 0
        while time.time() - start_time < think_time:
            root.explore()
            iterations += 1

            # 每100次迭代检查一次时间
            if iterations % 100 == 0 and time.time() - start_time >= think_time:
                break

        best_child = root.best_child()

        elapsed = time.time() - start_time

        # 显示详细信息
        if best_child:
            win_rate = best_child.wins / best_child.visits if best_child.visits > 0 else 0
            print(f"UCT MCTS搜索: {iterations}次迭代, {elapsed:.2f}秒")
            print(f"最佳走法: {best_child.move}, 胜率: {win_rate:.2%}, 访问: {best_child.visits}次")

            # 显示前3个候选走法
            top_children = sorted(root.children, key=lambda c: c.visits, reverse=True)[:3]
            print("候选走法:")
            for i, child in enumerate(top_children):
                wr = child.wins / child.visits if child.visits > 0 else 0
                print(f"  {i+1}. {child.move} - 胜率:{wr:.2%} 访问:{child.visits}")

            return best_child.move

        # 如果没有找到最佳走法，随机选择
        print("警告: MCTS未找到最佳走法，随机选择")
        return random.choice(self.gamestate.get_legal_moves())


class RAVEAgent(MCTSAgent):
    """RAVE增强的MCTS代理"""

    def __init__(self):
        super().__init__()
        self.name = "RAVE"

    def _mcts_search(self, think_time):
        """使用RAVE增强的MCTS搜索"""
        start_time = time.time()

        root = RAVENode(None, None, self.gamestate)

        iterations = 0
        while time.time() - start_time < think_time:
            root.explore()
            iterations += 1

            if iterations % 100 == 0 and time.time() - start_time >= think_time:
                break

        best_child = root.best_child()

        elapsed = time.time() - start_time

        # 显示详细信息
        if best_child:
            win_rate = best_child.wins / best_child.visits if best_child.visits > 0 else 0

            # 获取RAVE统计
            rave_rate = 0
            rave_visits_count = 0
            if hasattr(best_child, 'rave_visits') and best_child.move in best_child.rave_visits:
                rave_visits_count = best_child.rave_visits[best_child.move]
                if rave_visits_count > 0:
                    rave_rate = best_child.rave_wins[best_child.move] / rave_visits_count

            print(f"RAVE MCTS搜索: {iterations}次迭代, {elapsed:.2f}秒")
            print(f"最佳走法: {best_child.move}")
            print(f"  UCT胜率: {win_rate:.2%} (访问:{best_child.visits})")
            print(f"  RAVE胜率: {rave_rate:.2%} (访问:{rave_visits_count})")

            # 显示前3个候选走法
            top_children = sorted(root.children, key=lambda c: c.visits, reverse=True)[:3]
            print("候选走法:")
            for i, child in enumerate(top_children):
                wr = child.wins / child.visits if child.visits > 0 else 0
                rr = 0
                rv = 0
                if hasattr(child, 'rave_visits') and child.move in child.rave_visits:
                    rv = child.rave_visits[child.move]
                    if rv > 0:
                        rr = child.rave_wins[child.move] / rv
                print(f"  {i+1}. {child.move} - UCT:{wr:.2%} RAVE:{rr:.2%} 访问:{child.visits}")

            return best_child.move

        print("警告: RAVE MCTS未找到最佳走法，随机选择")
        return random.choice(self.gamestate.get_legal_moves())


# 为了向后兼容，保留旧的类名
class mctsagent(MCTSAgent):
    """向后兼容的别名"""
    pass


class rave_mctsagent(RAVEAgent):
    """向后兼容的别名"""
    pass
