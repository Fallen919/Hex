"""
TCP自我对弈服务器
收集来自多个客户端的训练数据，支持分布式训练
"""

import socket
import threading
import pickle
import numpy as np
import json
import time
import sys
from datetime import datetime
from pathlib import Path


class TrainingDataServer:
    """训练数据收集服务器"""
    
    def __init__(self, host='0.0.0.0', port=9999, save_dir='training_data'):
        self.host = host
        self.port = port
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        self.socket = None
        self.running = False
        self.clients = []
        self.observers = []  # 观战客户端列表 (address)
        self.observer_sockets = {}  # 观战者socket映射 {address: socket}
        self.primary_game_client = None  # 主要观战的客户端（第一个连接的）
        
        # 统计信息
        self.total_games = 0
        self.total_positions = 0
        self.data_batches = []
        self.current_batch = []
        self.batch_size = 1000  # 每1000个位置保存一次
        
        # 数据缓冲（用于训练时采样）
        self.memory_buffer = []
        self.memory_limit = 5000  # 内存中保留最近5000个位置
        
        # 当前对局状态（用于观战）
        self.active_games = {}  # {client_id: game_state}
        
        self.lock = threading.Lock()
        
    def start(self):
        """启动服务器"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.listen(5)
        self.running = True
        
        print(f"[服务器] 训练数据服务器启动")
        print(f"[服务器] 监听地址: {self.host}:{self.port}")
        print(f"[服务器] 数据目录: {self.save_dir}")
        
        # 启动接受连接的线程
        accept_thread = threading.Thread(target=self._accept_connections)
        accept_thread.daemon = True
        accept_thread.start()
        
        # 等待accept线程真正开始运行
        time.sleep(1)  # 增加到1秒
        
        print(f"[服务器] 服务器就绪，等待客户端连接...\n")
        sys.stdout.flush()  # 强制刷新输出
        
    def _accept_connections(self):
        """接受客户端连接"""
        print("[服务器] Accept线程已启动，开始监听连接...")
        sys.stdout.flush()
        
        while self.running:
            try:
                client_socket, address = self.socket.accept()
                print(f"[连接] 客户端连接: {address}")
                
                with self.lock:
                    self.clients.append(address)
                
                # 为每个客户端创建处理线程
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, address)
                )
                client_thread.daemon = True
                client_thread.start()
                
            except Exception as e:
                if self.running:
                    print(f"[错误] 接受连接错误: {e}")
    
    def _handle_client(self, client_socket, address):
        """处理客户端数据"""
        buffer = b''
        is_observer = False
        
        try:
            while self.running:
                # 接收数据
                chunk = client_socket.recv(4096)
                if not chunk:
                    break
                
                buffer += chunk
                
                # 尝试解析消息（使用长度前缀）
                while len(buffer) >= 4:
                    msg_len = int.from_bytes(buffer[:4], 'big')
                    
                    if len(buffer) < 4 + msg_len:
                        break  # 数据不完整，继续接收
                    
                    msg_data = buffer[4:4+msg_len]
                    buffer = buffer[4+msg_len:]
                    
                    # 处理消息
                    try:
                        message = pickle.loads(msg_data)
                        # 检查是否是观战者注册
                        if message.get('type') == 'register_observer':
                            is_observer = True
                            with self.lock:
                                self.observer_sockets[address] = client_socket
                        self._process_message(message, address)
                    except Exception as e:
                        print(f"[错误] 解析消息错误: {e}")
                        
        except Exception as e:
            print(f"[错误] 客户端 {address} 错误: {e}")
        finally:
            # 清理连接
            if is_observer:
                with self.lock:
                    if address in self.observer_sockets:
                        del self.observer_sockets[address]
            client_socket.close()
            with self.lock:
                if address in self.clients:
                    self.clients.remove(address)
            print(f"[断开] 客户端断开: {address}")
    
    def _process_message(self, message, address):
        """处理接收到的消息"""
        msg_type = message.get('type')
        
        if msg_type == 'game_start':
            print(f"[游戏] 客户端 {address} 开始新游戏")
            # 初始化游戏状态
            with self.lock:
                self.active_games[address] = {
                    'board_size': message.get('board_size', 11),
                    'board': None,
                    'moves': [],
                    'current_player': 1
                }
                
                # 如果这是第一个游戏客户端，设为主观战对象
                if self.primary_game_client is None:
                    self.primary_game_client = address
                    print(f"[观战] 客户端 {address} 被设为主观战对象")
            
        elif msg_type == 'move':
            # 接收走法更新（用于观战）
            move = message.get('move')
            board = message.get('board')
            player = message.get('player')
            
            with self.lock:
                if address in self.active_games:
                    self.active_games[address]['board'] = board
                    self.active_games[address]['moves'].append(move)
                    self.active_games[address]['current_player'] = player
                    move_num = len(self.active_games[address]['moves'])
                else:
                    move_num = 0
                
                num_observers = len(self.observer_sockets)
                
                # 只广播主观战客户端的更新
                should_broadcast = (address == self.primary_game_client)
            
            # 只广播主要客户端的对局
            if should_broadcast:
                update_msg = {
                    'type': 'game_update',
                    'client': str(address),
                    'move': move,
                    'board': board,
                    'player': player,
                    'move_number': move_num
                }
                
                self._broadcast_to_observers(update_msg)
            
        elif msg_type == 'position':
            # 接收训练位置
            data = message.get('data')
            self._add_position(data)
            
        elif msg_type == 'game_end':
            # 游戏结束
            winner = message.get('winner')
            moves = message.get('moves')
            with self.lock:
                self.total_games += 1
                
                # 如果结束的是主观战游戏，重置主观战客户端
                if address == self.primary_game_client:
                    self.primary_game_client = None
                    # 如果还有其他活跃游戏，选择下一个
                    for client_addr in self.active_games:
                        if client_addr != address:
                            self.primary_game_client = client_addr
                            print(f"[观战] 切换主观战对象到 {client_addr}")
                            break
                
                # 清除游戏状态
                if address in self.active_games:
                    del self.active_games[address]
            
            print(f"[完成] 游戏 #{self.total_games} 完成 | 胜者: {'红方' if winner == 1 else '蓝方'} | 步数: {moves}")
            
            # 通知观战者游戏结束（只通知主观战游戏）
            if address == self.primary_game_client or self.primary_game_client is None:
                self._broadcast_to_observers({
                    'type': 'game_end',
                    'client': str(address),
                    'winner': winner,
                    'moves': moves
                })
            
        elif msg_type == 'register_observer':
            # 注册观战者
            with self.lock:
                if address not in self.observers:
                    self.observers.append(address)
            print(f"[观战] 观战者加入: {address}")
            
        elif msg_type == 'stats':
            # 客户端请求统计信息
            stats = self.get_stats()
            # 发送响应（这里简化处理）
    
    def _broadcast_to_observers(self, message):
        """广播消息给所有观战者"""
        dead_observers = []
        
        with self.lock:
            observer_list = list(self.observer_sockets.items())
        
        for address, sock in observer_list:
            try:
                # 发送消息
                data = pickle.dumps(message)
                msg_len = len(data).to_bytes(4, 'big')
                sock.sendall(msg_len + data)
            except Exception as e:
                print(f"[错误] 向观战者 {address} 广播失败: {e}")
                dead_observers.append(address)
        
        # 清理断开的观战者
        if dead_observers:
            with self.lock:
                for address in dead_observers:
                    if address in self.observers:
                        self.observers.remove(address)
                    if address in self.observer_sockets:
                        del self.observer_sockets[address]
                        print(f"[断开] 观战者 {address} 已断开")
    
    def get_active_games(self):
        """获取当前活跃的游戏"""
        with self.lock:
            return dict(self.active_games)
    
    def get_stats(self):
        """获取统计信息"""
        with self.lock:
            return {
                'total_games': self.total_games,
                'total_positions': self.total_positions,
                'connected_clients': len(self.clients),
                'active_games': len(self.active_games),
                'observers': len(self.observers),
                'batches_saved': len(self.data_batches),
                'memory_buffer_size': len(self.memory_buffer)
            }
    
    def _add_position(self, data):
        """添加训练位置"""
        with self.lock:
            self.current_batch.append(data)
            self.total_positions += 1
            
            # 添加到内存缓冲
            self.memory_buffer.append(data)
            if len(self.memory_buffer) > self.memory_limit:
                self.memory_buffer.pop(0)
            
            # 达到批次大小，保存到磁盘
            if len(self.current_batch) >= self.batch_size:
                self._save_batch()
    
    def _save_batch(self):
        """保存批次到磁盘"""
        if not self.current_batch:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.save_dir / f"batch_{timestamp}_{len(self.data_batches)}.npz"
        
        # 解析数据
        states = []
        policies = []
        values = []
        
        for item in self.current_batch:
            states.append(item['state'])
            policies.append(item['policy'])
            values.append(item['value'])
        
        # 保存为numpy压缩格式
        np.savez_compressed(
            filename,
            states=np.array(states),
            policies=np.array(policies),
            values=np.array(values)
        )
        
        self.data_batches.append(str(filename))
        print(f"[保存] 批次: {filename.name} ({len(self.current_batch)} 个位置)")
        
        self.current_batch = []
    
    def get_stats(self):
        """获取统计信息"""
        with self.lock:
            return {
                'total_games': self.total_games,
                'total_positions': self.total_positions,
                'connected_clients': len(self.clients),
                'batches_saved': len(self.data_batches),
                'memory_buffer_size': len(self.memory_buffer)
            }
    
    def sample_batch(self, batch_size=32):
        """从内存缓冲区采样批次（用于实时训练）"""
        with self.lock:
            if len(self.memory_buffer) < batch_size:
                return None
            
            indices = np.random.choice(len(self.memory_buffer), batch_size, replace=False)
            batch = [self.memory_buffer[i] for i in indices]
            
            states = np.array([item['state'] for item in batch])
            policies = np.array([item['policy'] for item in batch])
            values = np.array([item['value'] for item in batch])
            
            return states, policies, values
    
    def load_all_data(self):
        """加载所有磁盘数据（用于训练）"""
        all_states = []
        all_policies = []
        all_values = []
        
        for batch_file in self.data_batches:
            data = np.load(batch_file)
            all_states.append(data['states'])
            all_policies.append(data['policies'])
            all_values.append(data['values'])
        
        if all_states:
            return (
                np.concatenate(all_states),
                np.concatenate(all_policies),
                np.concatenate(all_values)
            )
        return None, None, None
    
    def stop(self):
        """停止服务器"""
        print("\n[关闭] 正在关闭服务器...")
        self.running = False
        
        # 保存剩余数据
        if self.current_batch:
            self._save_batch()
        
        # 关闭socket
        if self.socket:
            self.socket.close()
        
        # 保存统计信息
        stats = self.get_stats()
        stats_file = self.save_dir / 'training_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print("[关闭] 服务器已关闭")
        print(f"  总游戏数: {stats['total_games']}")
        print(f"  总位置数: {stats['total_positions']}")
        print(f"  保存批次: {stats['batches_saved']}")


def main():
    """主函数"""
    server = TrainingDataServer(host='0.0.0.0', port=9999)
    server.start()
    
    try:
        while True:
            time.sleep(1)
            # 每10秒显示统计
            if int(time.time()) % 10 == 0:
                stats = server.get_stats()
                print(f 状态: {stats['connected_clients']} 客户端 | "
                      f"{stats['total_games']} 游戏 | "
                      f"{stats['total_positions']} 位置", end='')
    except KeyboardInterrupt:
        server.stop()


if __name__ == '__main__':
    main()
