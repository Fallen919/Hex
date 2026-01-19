"""
TCP训练管理器
整合服务器和训练，支持实时训练
"""

import threading
import time
import numpy as np
from pathlib import Path
from tcp_selfplay_server import TrainingDataServer

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from cnn.Hex_cnn_model import HexCNN
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("[警告] PyTorch不可用，无法进行训练")


class TCPTrainingManager:
    """TCP训练管理器 - 边收集边训练"""
    
    def __init__(self, server_host='0.0.0.0', server_port=9999, 
                 save_dir='training_data', model_dir='models'):
        self.server = TrainingDataServer(host=server_host, port=server_port, save_dir=save_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.model = None
        self.optimizer = None
        self.device = None
        
        self.training = False
        self.training_thread = None
        
        if PYTORCH_AVAILABLE:
            self._init_model()
    
    def _init_model(self):
        """初始化模型"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = HexCNN(board_size=11).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        print(f"[初始化] 模型初始化完成 (设备: {self.device})")
    
    def start_server(self):
        """启动服务器"""
        self.server.start()
    
    def start_training(self, batch_size=32, train_interval=10):
        """启动实时训练"""
        if not PYTORCH_AVAILABLE:
            print("[错误] PyTorch不可用，无法训练")
            return
        
        self.training = True
        self.training_thread = threading.Thread(
            target=self._training_loop,
            args=(batch_size, train_interval)
        )
        self.training_thread.daemon = True
        self.training_thread.start()
        print(f"[训练] 训练线程已启动 (批次大小: {batch_size}, 间隔: {train_interval}秒)")
    
    def _training_loop(self, batch_size, train_interval):
        """训练循环"""
        step = 0
        
        while self.training:
            time.sleep(train_interval)
            
            # 从服务器内存缓冲区采样
            batch_data = self.server.sample_batch(batch_size)
            
            if batch_data is None:
                continue  # 数据不足，跳过
            
            states, policies, values = batch_data
            
            # 转换为PyTorch张量
            states_tensor = torch.FloatTensor(states).to(self.device)
            policies_tensor = torch.FloatTensor(policies).to(self.device)
            values_tensor = torch.FloatTensor(values).unsqueeze(1).to(self.device)
            
            # 训练一步
            loss, policy_loss, value_loss = self._train_step(
                states_tensor, policies_tensor, values_tensor
            )
            
            step += 1
            if step % 10 == 0:
                stats = self.server.get_stats()
                print(f"\n📈 训练步骤 {step}")
                print(f"   总损失: {loss:.4f} | 策略: {policy_loss:.4f} | 价值: {value_loss:.4f}")
                print(f"   数据: {stats['total_positions']} 位置, {stats['total_games']} 游戏")
            
            # 定期保存模型
            if step % 100 == 0:
                self.save_model(f'checkpoint_step_{step}.pth')
    
    def _train_step(self, states, policy_targets, value_targets):
        """执行一步训练"""
        self.model.train()
        
        # 前向传播
        policy_pred, value_pred = self.model(states)
        
        # 计算损失
        # 策略损失: 使用KL散度（交叉熵的正确形式）
        log_probs = torch.nn.functional.log_softmax(policy_pred, dim=1)
        policy_loss = -(policy_targets * log_probs).sum(dim=1).mean()
        
        value_loss = nn.MSELoss()(value_pred, value_targets)
        total_loss = policy_loss + 2.0 * value_loss
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # 梯度裁剪
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return total_loss.item(), policy_loss.item(), value_loss.item()
    
    def train_on_disk_data(self, epochs=10, batch_size=32):
        """在磁盘数据上训练（离线训练）"""
        if not PYTORCH_AVAILABLE:
            print("[错误] PyTorch不可用")
            return
        
        print(f"📚 加载磁盘数据...")
        states, policies, values = self.server.load_all_data()
        
        if states is None:
            print("[错误] 没有可用的训练数据")
            return
        
        print(f"[数据] 加载了 {len(states)} 个训练样本")
        
        # 转换为PyTorch张量
        dataset_size = len(states)
        indices = np.arange(dataset_size)
        
        for epoch in range(epochs):
            np.random.shuffle(indices)
            
            total_loss = 0
            num_batches = 0
            
            for i in range(0, dataset_size, batch_size):
                batch_indices = indices[i:i+batch_size]
                
                batch_states = torch.FloatTensor(states[batch_indices]).to(self.device)
                batch_policies = torch.FloatTensor(policies[batch_indices]).to(self.device)
                batch_values = torch.FloatTensor(values[batch_indices]).unsqueeze(1).to(self.device)
                
                loss, _, _ = self._train_step(batch_states, batch_policies, batch_values)
                total_loss += loss
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}/{epochs} - 平均损失: {avg_loss:.4f}")
            
            # 每个epoch保存一次
            self.save_model(f'epoch_{epoch+1}.pth')
        
        print("[完成] 训练完成")
    
    def save_model(self, filename):
        """保存模型"""
        if self.model is None:
            return
        
        filepath = self.model_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
        print(f"[保存] 模型已保存: {filename}")
    
    def load_model(self, filename):
        """加载模型"""
        if self.model is None:
            self._init_model()
        
        filepath = self.model_dir / filename
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"[加载] 模型已加载: {filename}")
    
    def stop(self):
        """停止所有服务"""
        print("\n🛑 正在停止...")
        
        # 停止训练
        self.training = False
        if self.training_thread:
            self.training_thread.join(timeout=5)
        
        # 停止服务器
        self.server.stop()
        
        print("[停止] 已停止所有服务")


def main():
    """主函数 - 一体化训练"""
    import argparse
    
    parser = argparse.ArgumentParser(description='TCP训练管理器')
    parser.add_argument('--mode', choices=['online', 'offline'], default='online',
                       help='训练模式: online=实时训练, offline=离线训练')
    parser.add_argument('--port', type=int, default=9999, help='服务器端口')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数（仅离线模式）')
    
    args = parser.parse_args()
    
    manager = TCPTrainingManager(server_port=args.port)
    manager.start_server()
    
    try:
        if args.mode == 'online':
            # 实时训练模式
            manager.start_training(batch_size=args.batch_size, train_interval=10)
            print("\n💡 提示: 启动客户端开始自我对弈")
            print("   python tcp_selfplay_client.py --games 100\n")
            
            while True:
                time.sleep(1)
                
        else:
            # 离线训练模式
            print("等待收集数据... (按Ctrl+C开始训练)")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
            
            manager.train_on_disk_data(epochs=args.epochs, batch_size=args.batch_size)
            
    except KeyboardInterrupt:
        pass
    finally:
        manager.stop()


if __name__ == '__main__':
    main()
