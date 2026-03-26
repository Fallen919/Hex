import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


class ResidualBlock(nn.Module):
    """残差块"""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class HexCNN(nn.Module):
    """Hex棋盘CNN模型"""

    def __init__(self, board_size=11):
        super().__init__()
        self.board_size = board_size

        # 初始卷积层 (3通道 -> 128通道)
        self.conv1 = nn.Conv2d(3, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

        # 12个残差块
        self.res_blocks = nn.ModuleList([
            ResidualBlock(128) for _ in range(12)
        ])

        # 策略头
        self.policy_conv = nn.Conv2d(128, 4, 1)
        self.policy_bn = nn.BatchNorm2d(4)
        self.policy_fc = nn.Linear(4 * board_size * board_size, board_size * board_size)

        # 价值头
        self.value_conv = nn.Conv2d(128, 4, 1)
        self.value_bn = nn.BatchNorm2d(4)
        self.value_fc1 = nn.Linear(4 * board_size * board_size, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # 初始卷积
        x = F.relu(self.bn1(self.conv1(x)))

        # 通过残差块
        for block in self.res_blocks:
            x = block(x)

        # 策略头
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 4 * self.board_size * self.board_size)
        policy = self.policy_fc(policy)

        # 价值头
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 4 * self.board_size * self.board_size)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value


class HexCNNTrainer:
    """CNN训练器"""

    def __init__(self, board_size=11, device=None):
        self.board_size = board_size

        # 设备选择
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        print(f"使用设备: {self.device}")

        # 初始化模型
        self.model = HexCNN(board_size).to(self.device)

        # 优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.001,
            weight_decay=1e-4
        )

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )

        # 经验回放缓冲区
        self.replay_buffer = []
        self.batch_size = 128
        self.max_buffer_size = 50000

        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")

    def state_to_tensor(self, state, current_player):
        """将游戏状态转换为张量（C++版本）"""
        try:
            board = state.get_board()
            size = state.get_size()

            # 创建三通道输入
            tensor = np.zeros((3, size, size), dtype=np.float32)

            for i in range(size):
                for j in range(size):
                    cell = board[i * size + j]
                    if cell == 1:  # RED
                        tensor[0, i, j] = 1.0
                    elif cell == 2:  # BLUE
                        tensor[1, i, j] = 1.0

            # 当前玩家通道
            if current_player == 1:
                tensor[2, :, :] = 1.0

            return torch.FloatTensor(tensor).unsqueeze(0).to(self.device)

        except Exception as e:
            print(f"[错误] 状态转换失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _convert_gamestate_to_tensor(self, game, current_player):
        """将Python游戏状态转换为张量"""
        try:
            board = game.board
            size = game.size

            # 创建三通道输入
            tensor = np.zeros((3, size, size), dtype=np.float32)

            for i in range(size):
                for j in range(size):
                    cell = board[i, j]
                    if cell == 1:  # RED
                        tensor[0, i, j] = 1.0
                    elif cell == 2:  # BLUE
                        tensor[1, i, j] = 1.0

            # 当前玩家通道
            if current_player == 1:
                tensor[2, :, :] = 1.0

            return torch.FloatTensor(tensor).unsqueeze(0)

        except Exception as e:
            print(f"状态转换失败: {e}")
            return None

    def get_best_move(self, state, current_player, temperature=1.0):
        """获取最佳走法（修复掩码）"""
        try:
            self.model.eval()

            with torch.no_grad():
                tensor = self.state_to_tensor(state, current_player)
                if tensor is None:
                    return None, 0.0

                policy_logits, value = self.model(tensor)

                # 获取合法走法
                legal_moves = state.moves()
                if not legal_moves:
                    return None, value.item()

                # 创建强掩码（设为极小值）
                mask = torch.full(
                    (self.board_size * self.board_size,),
                    -1e9,  # ← 改为更大的负数
                    device=self.device
                )

                #  仅对合法走法设为0
                for move in legal_moves:
                    x, y = move
                    idx = x * self.board_size + y
                    mask[idx] = 0

                # 应用掩码
                masked_logits = policy_logits[0] + mask

                #  检查掩码后的有效性
                valid_logits = masked_logits[mask == 0]
                if len(valid_logits) == 0:
                    print("[警告] 掩码后无有效走法")
                    move_idx = random.choice([m[0] * self.board_size + m[1] for m in legal_moves])
                else:
                    # 温度采样
                    if temperature > 0:
                        probs = F.softmax(masked_logits / temperature, dim=0)
                        probs_np = probs.cpu().numpy()

                        # 严格检查概率分布
                        if np.isnan(probs_np).any():
                            print("[警告] 出现NaN，使用随机走法")
                            move_idx = random.choice([m[0] * self.board_size + m[1] for m in legal_moves])
                        elif np.sum(probs_np) == 0:
                            print("[警告] 概率全为0，使用随机走法")
                            move_idx = random.choice([m[0] * self.board_size + m[1] for m in legal_moves])
                        else:
                            # 重新归一化概率
                            probs_np = probs_np / np.sum(probs_np)
                            move_idx = np.random.choice(len(probs_np), p=probs_np)
                    else:
                        # Greedy选择
                        move_idx = masked_logits.argmax().item()

                x = move_idx // self.board_size
                y = move_idx % self.board_size

                if (x, y) not in legal_moves:
                    print(f"[警告] 预测的走法 ({x},{y}) 不合法，随机选择")
                    x, y = random.choice(legal_moves)

                return (x, y), value.item()

        except Exception as e:
            print(f"[错误] 预测失败: {e}")
            import traceback
            traceback.print_exc()


            legal_moves = state.moves()
            if legal_moves:
                return random.choice(legal_moves), 0.0
            return None, 0.0

    def generate_training_data_from_mcts(self, num_games=100, mcts_time=1.0):
        """使用MCTS生成高质量训练数据"""
        print(f"\n{'='*60}")
        print(f"使用MCTS生成训练数据")
        print(f"游戏数量: {num_games}")
        print(f"MCTS思考时间: {mcts_time}秒")
        print(f"{'='*60}\n")

        try:
            # 动态导入MCTS代理
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

            # 尝试导入C++版本
            try:
                import hex_cpp
                use_cpp = True
                print("[信息] 使用C++加速版本")
            except ImportError:
                use_cpp = False
                print("[警告] 使用Python版本")

            from Hex_cnn import rave_mctsagent, gamestate

            total_positions = 0

            for game_idx in range(num_games):
                # 创建新游戏
                game = gamestate(self.board_size)
                mcts_agent = rave_mctsagent(game)

                game_data = []
                move_count = 0

                while game.winner() == gamestate.PLAYERS["none"]:
                    current_player = game.turn()

                    # 使用MCTS搜索
                    mcts_agent.search(mcts_time)
                    stats = mcts_agent.statistics()

                    # 获取MCTS的访问次数分布作为策略目标
                    visit_counts = np.zeros(self.board_size * self.board_size, dtype=np.float32)

                    if hasattr(mcts_agent.root, 'children') and mcts_agent.root.children:
                        for move, child in mcts_agent.root.children.items():
                            idx = move[0] * self.board_size + move[1]
                            visit_counts[idx] = child.N

                    # 归一化为概率分布
                    total_visits = visit_counts.sum()
                    if total_visits > 0:
                        policy_target = visit_counts / total_visits
                    else:
                        # 如果没有访问数据，使用均匀分布
                        legal_moves = game.moves()
                        policy_target = np.zeros(self.board_size * self.board_size, dtype=np.float32)
                        for move in legal_moves:
                            idx = move[0] * self.board_size + move[1]
                            policy_target[idx] = 1.0 / len(legal_moves)

                    # 选择最佳走法
                    move = mcts_agent.best_move()

                    if move == gamestate.GAMEOVER:
                        break

                    # 转换游戏状态为张量
                    tensor = self._convert_gamestate_to_tensor(game, current_player)

                    if tensor is not None:
                        game_data.append({
                            'state': tensor.cpu(),
                            'player': current_player,
                            'policy': policy_target,
                            'move': move,
                            'value': 0.0  # 临时值，稍后更新
                        })
                        move_count += 1

                    # 执行走法
                    game.play(move)
                    mcts_agent.move(move)

                # 更新价值标签
                winner = game.winner()
                if winner != gamestate.PLAYERS["none"]:
                    for i, data in enumerate(game_data):
                        if data['player'] == winner:
                            # 胜者：使用折扣因子
                            discount = 0.99 ** (len(game_data) - i - 1)
                            data['value'] = discount
                        else:
                            # 败者
                            discount = 0.99 ** (len(game_data) - i - 1)
                            data['value'] = -discount

                    self.replay_buffer.extend(game_data)
                    total_positions += len(game_data)

                # 进度显示
                if (game_idx + 1) % 10 == 0:
                    avg_moves = total_positions / (game_idx + 1)
                    print(f"完成 {game_idx + 1}/{num_games} 局 | "
                          f"总位置: {total_positions} | "
                          f"平均每局: {avg_moves:.1f}步")

            # 限制缓冲区大小
            if len(self.replay_buffer) > self.max_buffer_size:
                self.replay_buffer = self.replay_buffer[-self.max_buffer_size:]

            print(f"\n[信息] 数据生成完成!")
            print(f"总训练样本: {len(self.replay_buffer)}")
            print(f"{'='*60}\n")

            return len(self.replay_buffer)

        except Exception as e:
            print(f"[错误] 数据生成失败: {e}")
            import traceback
            traceback.print_exc()
            return 0

    def train_on_batch_with_policy_target(self):
        """使用策略分布目标训练"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0, 0.0

        self.model.train()
        batch = random.sample(self.replay_buffer, self.batch_size)

        states = torch.cat([item['state'] for item in batch]).to(self.device)

        # 策略目标是概率分布
        policy_targets = torch.FloatTensor(
            np.array([item['policy'] for item in batch])
        ).to(self.device)

        values = torch.FloatTensor([item['value'] for item in batch]).unsqueeze(1).to(self.device)

        # 前向传播
        policy_logits, value_pred = self.model(states)

        # 策略损失: 使用交叉熵（更稳定）
        log_probs = F.log_softmax(policy_logits, dim=1)
        policy_loss = -(policy_targets * log_probs).sum(dim=1).mean()

        # 价值损失
        value_loss = F.mse_loss(value_pred, values)

        # 总损失（价值损失权重更高）
        total_loss = policy_loss + 2.0 * value_loss

        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return total_loss.item(), policy_loss.item(), value_loss.item()

    def train_from_mcts_data(self, num_epochs=50, verbose=True):
        """使用MCTS数据训练模型"""
        if len(self.replay_buffer) == 0:
            print("[错误] 没有训练数据！请先运行 generate_training_data_from_mcts()")
            return

        print(f"\n{'='*60}")
        print(f"开始训练神经网络")
        print(f"训练样本: {len(self.replay_buffer)}")
        print(f"训练轮数: {num_epochs}")
        print(f"批次大小: {self.batch_size}")
        print(f"{'='*60}\n")

        best_loss = float('inf')

        for epoch in range(num_epochs):
            epoch_losses = []
            policy_losses = []
            value_losses = []

            # 多次采样训练
            num_batches = max(1, len(self.replay_buffer) // self.batch_size)

            for _ in range(num_batches):
                total_loss, policy_loss, value_loss = self.train_on_batch_with_policy_target()
                epoch_losses.append(total_loss)
                policy_losses.append(policy_loss)
                value_losses.append(value_loss)

            avg_total_loss = np.mean(epoch_losses)
            avg_policy_loss = np.mean(policy_losses)
            avg_value_loss = np.mean(value_losses)

            # 更新学习率
            self.scheduler.step(avg_total_loss)

            # 保存最佳模型
            if avg_total_loss < best_loss:
                best_loss = avg_total_loss
                self.save_model('hex_cnn_best.pth')

            # 打印进度
            if verbose and (epoch + 1) % 5 == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{num_epochs} | "
                      f"总损失: {avg_total_loss:.4f} | "
                      f"策略: {avg_policy_loss:.4f} | "
                      f"价值: {avg_value_loss:.4f} | "
                      f"学习率: {lr:.6f}")

        print(f"\n[信息] 训练完成！最佳损失: {best_loss:.4f}")
        print(f"{'='*60}\n")

    def self_play_game(self, temperature=1.0):
        """自我对弈一局"""
        try:
            import hex_cpp
            state = hex_cpp.GameState(self.board_size)
            game_data = []
            while state.winner() == 0:
                current_player = state.turn()

                move, value = self.get_best_move(state, current_player, temperature=temperature)

                if move is None:
                    break

                tensor = self.state_to_tensor(state, current_player)
                if tensor is not None:
                    game_data.append({
                        'state': tensor.cpu(),
                        'player': current_player,
                        'move': move,
                        'value': value
                    })

                state.play(move[0], move[1])

            # 游戏结束，更新价值
            winner = state.winner()
            if winner != 0:
                for data in game_data:
                    if data['player'] == winner:
                        data['value'] = 1.0
                    else:
                        data['value'] = -1.0

            return game_data

        except Exception as e:
            print(f"[错误] 自我对弈失败: {e}")
            import traceback
            traceback.print_exc()
            return []

    def train_on_batch(self):
        """在一个批次上训练"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        self.model.train()

        # 随机采样
        batch = random.sample(self.replay_buffer, self.batch_size)

        states = torch.cat([item['state'] for item in batch]).to(self.device)
        moves = torch.LongTensor([item['move'][0] * self.board_size + item['move'][1] for item in batch]).to(
            self.device)
        values = torch.FloatTensor([item['value'] for item in batch]).unsqueeze(1).to(self.device)

        # 前向传播
        policy_logits, value_pred = self.model(states)

        # 计算损失
        policy_loss = F.cross_entropy(policy_logits, moves)
        value_loss = F.mse_loss(value_pred, values)
        total_loss = policy_loss + value_loss

        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return total_loss.item()

    def train_from_self_play(self, num_games=100, epochs_per_game=5, temperature=1.0):
        """从自我对弈中训练（旧版本，保留用于兼容性）"""
        print(f"\n开始自我对弈训练: {num_games}局游戏")
        print(f"每局游戏后训练 {epochs_per_game} 个epoch")
        print(f"温度参数: {temperature}\n")

        for game_idx in range(num_games):
            # 自我对弈
            game_data = self.self_play_game(temperature=temperature)

            if game_data:
                self.replay_buffer.extend(game_data)

                # 限制缓冲区大小
                if len(self.replay_buffer) > self.max_buffer_size:
                    self.replay_buffer = self.replay_buffer[-self.max_buffer_size:]

                # 训练
                losses = []
                for _ in range(epochs_per_game):
                    loss = self.train_on_batch()
                    if loss > 0:
                        losses.append(loss)

                # 打印进度
                if (game_idx + 1) % 10 == 0:
                    avg_loss = sum(losses) / len(losses) if losses else 0
                    print(f"游戏 {game_idx + 1}/{num_games}, "
                          f"缓冲区: {len(self.replay_buffer)}, "
                          f"平均损失: {avg_loss:.4f}")

        print(f"\n[信息] 训练完成！最终缓冲区大小: {len(self.replay_buffer)}")

    def train(self, num_iterations=100, games_per_iteration=50):
        """训练循环"""
        print(f"\n开始训练: {num_iterations}轮迭代, 每轮{games_per_iteration}局游戏")
        print(f"批次大小: {self.batch_size}")
        print(f"最大缓冲区大小: {self.max_buffer_size}\n")

        previous_lr = self.optimizer.param_groups[0]['lr']

        for iteration in range(num_iterations):
            print(f"\n{'=' * 60}")
            print(f"迭代 {iteration + 1}/{num_iterations}")
            print(f"{'=' * 60}")

            print("生成训练数据...")
            new_games = 0

            # 动态调整温度
            if iteration < 20:
                temp = 1.5
            elif iteration < 50:
                temp = 1.0
            else:
                temp = 0.5

            for game_idx in range(games_per_iteration):
                game_data = self.self_play_game(temperature=temp)
                if game_data:
                    self.replay_buffer.extend(game_data)
                    new_games += 1

                if (game_idx + 1) % 10 == 0:
                    print(f"  完成 {game_idx + 1}/{games_per_iteration} 局游戏")

            print(f"[信息] 成功生成 {new_games} 局游戏")

            # 限制缓冲区大小
            if len(self.replay_buffer) > self.max_buffer_size:
                self.replay_buffer = self.replay_buffer[-self.max_buffer_size:]
                print(f"[警告] 缓冲区已满，保留最新的 {self.max_buffer_size} 条数据")

            print(f"当前训练数据池大小: {len(self.replay_buffer)}")

            # 训练神经网络
            print("\n训练神经网络...")
            epoch_losses = []
            num_epochs = 10

            for epoch in range(num_epochs):
                loss = self.train_on_batch()
                epoch_losses.append(loss)

                if (epoch + 1) % 3 == 0:
                    print(f"  Epoch {epoch + 1}/{num_epochs}, 损失: {loss:.4f}")

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"\n本轮平均损失: {avg_loss:.4f}")

            # 更新学习率
            self.scheduler.step(avg_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            if current_lr != previous_lr:
                print(f"[调整] 学习率已调整: {previous_lr:.6f} -> {current_lr:.6f}")
                previous_lr = current_lr
            else:
                print(f"当前学习率: {current_lr:.6f}")

            # 定期保存模型
            if (iteration + 1) % 10 == 0:
                save_path = f'hex_cnn_iter_{iteration + 1}.pth'
                self.save_model(save_path)
                print(f"[信息] 模型已保存到 {save_path}")

            # 定期诊断
            if (iteration + 1) % 5 == 0:
                self.diagnose_training()

    def diagnose_training(self):
        """训练诊断"""
        print("\n" + "=" * 60)
        print("训练诊断")
        print("=" * 60)

        print(f"缓冲区大小: {len(self.replay_buffer)}")
        print(f"当前学习率: {self.optimizer.param_groups[0]['lr']:.6f}")

        if len(self.replay_buffer) > 0:
            sample = random.sample(self.replay_buffer, min(10, len(self.replay_buffer)))
            values = [item['value'] for item in sample]
            print(f"样本价值分布: 均值={np.mean(values):.3f}, 标准差={np.std(values):.3f}")

        print("=" * 60 + "\n")

    def save_model(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, path)
        print(f"[信息] 模型已保存: {path}")

    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"[信息] 模型已加载: {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("Hex CNN 模型测试")
    print("=" * 60)

    trainer = HexCNNTrainer(board_size=11)

    print("\n可用的训练方法:")
    print("1. trainer.generate_training_data_from_mcts(num_games=100, mcts_time=2.0)")
    print("   然后 trainer.train_from_mcts_data(num_epochs=50)")
    print("2. trainer.train_from_self_play(num_games=100)  # 简单接口（不推荐）")
    print("3. trainer.train(num_iterations=100, games_per_iteration=50)  # 完整接口（不推荐）")