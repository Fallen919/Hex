"""
TCP自我对弈客户端
连接到服务器，执行自我对弈并发送训练数据
"""

import socket
import pickle
import numpy as np
import time
import argparse
import sys
from game_core import GameState
from mcts_agent import RAVEAgent


class SelfPlayClient:
    """自我对弈客户端"""
    
    def __init__(self, server_host='localhost', server_port=9999, 
                 agent_type='RAVE', think_time=2.0, board_size=11):
        self.server_host = server_host
        self.server_port = server_port
        self.agent_type = agent_type
        self.think_time = think_time
        self.board_size = board_size
        
        self.socket = None
        self.connected = False
        self.games_played = 0
        self.enable_visualization = True  # 是否发送可视化数据
        
    def connect(self):
        """连接到服务器"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.server_host, self.server_port))
            self.connected = True
            print(f"[连接] 已连接到服务器 {self.server_host}:{self.server_port}")
            return True
        except Exception as e:
            print(f"[错误] 连接失败: {e}")
            return False
    
    def _send_message(self, message):
        """发送消息到服务器"""
        try:
            data = pickle.dumps(message)
            msg_len = len(data).to_bytes(4, 'big')
            self.socket.sendall(msg_len + data)
            return True
        except Exception as e:
            print(f"[错误] 发送消息失败: {e}")
            self.connected = False
            return False
    
    def play_game(self):
        """执行一局自我对弈"""
        # 初始化游戏
        game = GameState(self.board_size)
        agent = RAVEAgent()
        agent.set_gamestate(game)
        
        game_data = []  # 临时存储本局数据
        
        # 通知服务器游戏开始
        self._send_message({
            'type': 'game_start',
            'board_size': self.board_size
        })
        
        move_count = 0
        
        # 游戏主循环
        while game.winner() == 0:
            # 记录执行走法前的状态和玩家
            current_player = game.toplay
            state = self._encode_state(game)
            
            # 执行MCTS搜索
            agent.set_gamestate(game)
            move = agent.best_move(think_time=self.think_time)
            
            # 立即从MCTS root获取策略分布（趁root还在）
            policy = self._get_mcts_policy(agent, game)
            
            # 记录训练数据
            game_data.append({
                'state': state,
                'policy': policy,
                'player': current_player,
                'move_number': move_count
            })
            
            # 执行走法
            game.make_move(move)
            move_count += 1
            
            # 发送走法更新（用于可视化）
            if self.enable_visualization:
                self._send_message({
                    'type': 'move',
                    'move': move,
                    'board': game.board.tolist(),
                    'player': current_player  # 刚走的玩家
                })
        
        # 游戏结束，分配价值标签
        winner = game.winner()
        for data in game_data:
            # 从当前玩家角度的价值
            if data['player'] == winner:
                value = 1.0
            else:
                value = -1.0
            
            # 发送训练数据到服务器
            self._send_message({
                'type': 'position',
                'data': {
                    'state': data['state'],
                    'policy': data['policy'],
                    'value': value
                }
            })
        
        # 通知服务器游戏结束
        self._send_message({
            'type': 'game_end',
            'winner': winner,
            'moves': move_count
        })
        
        self.games_played += 1
        return winner, move_count
    
    def _encode_state(self, game):
        """编码游戏状态为神经网络输入格式"""
        board = game.board
        size = game.size
        
        # 3通道: 红棋、蓝棋、当前玩家
        state = np.zeros((3, size, size), dtype=np.float32)
        
        # 通道0: 红棋位置
        state[0] = (board == 1).astype(np.float32)
        
        # 通道1: 蓝棋位置
        state[1] = (board == 2).astype(np.float32)
        
        # 通道2: 当前玩家（红方全1，蓝方全0）
        if game.toplay == 1:
            state[2] = np.ones((size, size), dtype=np.float32)
        
        return state
    
    def _get_mcts_policy(self, agent, game):
        """从MCTS获取策略分布"""
        size = game.size
        policy = np.zeros(size * size, dtype=np.float32)
        
        # 获取MCTS根节点（如果可用）
        if hasattr(agent, 'root') and agent.root is not None and hasattr(agent.root, 'children'):
            root = agent.root
            
            if len(root.children) > 0:
                total_visits = sum(child.visits for child in root.children)
                
                if total_visits > 0:
                    # 从MCTS访问次数构建策略分布
                    for child in root.children:
                        if child.move:
                            x, y = child.move
                            idx = x * size + y
                            policy[idx] = child.visits / total_visits
                    
                    # 验证策略分布
                    policy_sum = policy.sum()
                    if policy_sum > 0.99:  # 允许小的数值误差
                        return policy
                    else:
                        print(f"[警告] 策略分布异常 (总和={policy_sum:.3f})，使用均匀分布")
            else:
                print("[警告] MCTS root没有子节点，使用均匀分布")
        else:
            print("[警告] MCTS root不可用，使用均匀分布")
        
        # 后备方案：使用均匀分布
        legal_moves = game.get_legal_moves()
        if legal_moves:
            for x, y in legal_moves:
                idx = x * size + y
                policy[idx] = 1.0 / len(legal_moves)
        
        return policy
    
    def run(self, num_games=None):
        """运行自我对弈"""
        if not self.connected:
            if not self.connect():
                return
        
        print("[开始] 开始自我对弈")
        print(f"   算法: {self.agent_type}")
        print(f"   思考时间: {self.think_time}秒")
        print(f"   棋盘大小: {self.board_size}x{self.board_size}")
        if num_games:
            print(f"   目标游戏数: {num_games}")
        print()
        
        try:
            game_num = 0
            while True:
                if num_games and game_num >= num_games:
                    break
                
                game_num += 1
                start_time = time.time()
                
                winner, moves = self.play_game()
                
                elapsed = time.time() - start_time
                winner_str = '红方' if winner == 1 else '蓝方'
                
                print(f"游戏 #{game_num}: {winner_str}胜 | {moves}手 | {elapsed:.1f}秒")
                
        except KeyboardInterrupt:
            print("\n[中断] 用户中断")
        except Exception as e:
            print(f"\n[错误] 错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.disconnect()
            if sys.stdin and sys.stdin.isatty():
                print("\n按任意键退出...")
                input()
    
    def disconnect(self):
        """断开连接"""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        self.connected = False
        print(f"\n[断开] 已断开连接 (共完成 {self.games_played} 局游戏)")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='海克斯棋自我对弈客户端')
    parser.add_argument('--host', default='localhost', help='服务器地址')
    parser.add_argument('--port', type=int, default=9999, help='服务器端口')
    parser.add_argument('--games', type=int, default=None, help='游戏数量（不指定则无限运行）')
    parser.add_argument('--time', type=float, default=2.0, help='MCTS思考时间（秒）')
    parser.add_argument('--size', type=int, default=11, help='棋盘大小')
    
    args = parser.parse_args()
    
    client = SelfPlayClient(
        server_host=args.host,
        server_port=args.port,
        think_time=args.time,
        board_size=args.size
    )
    
    client.run(num_games=args.games)


if __name__ == '__main__':
    main()
