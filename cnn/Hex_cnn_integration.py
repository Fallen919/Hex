"""
海克斯棋CNN模型集成模块
将深度学习模型集成到现有的GUI和游戏引擎中
"""

import numpy as np
from copy import deepcopy
import torch
import sys
import importlib
import inspect
from pathlib import Path
from time import perf_counter
import os

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False



# ========================================
# 动态导入父模块
# ========================================
def _import_game_core():
    """
    动态导入game_core模块

    Returns:
        GameState类，如果导入失败则返回None
    """
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent

    # 添加项目根目录到sys.path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        print(f"✓ 已添加项目根目录: {project_root}")

    try:
        import game_core
        print(f"✓ 已导入game_core模块: {game_core.__file__}")
        return game_core.GameState
    except ImportError as e:
        print(f"✗ 无法导入game_core: {e}")
        print(f"当前sys.path: {sys.path[:3]}")
        return None


def _import_hex_cpp():
    """
    动态导入hex_cpp模块

    Returns:
        hex_cpp模块，如果导入失败则返回None
    """
    try:
        import hex_cpp
        print(f"✓ C++模块已加载: {hex_cpp.__file__}")
        return hex_cpp
    except ImportError as e:
        print(f"✗ C++模块加载失败: {e}")
        return None


# ========================================
# 全局导入
# ========================================
GameState = _import_game_core()
_GAME_CORE_AVAILABLE = GameState is not None

HEX_CPP_MODULE = _import_hex_cpp()
HAS_CPP_AVAILABLE = HEX_CPP_MODULE is not None

# ========================================
# 强制重新加载CNN模型模块
# ========================================
if 'cnn.Hex_cnn_model' in sys.modules:
    importlib.reload(sys.modules['cnn.Hex_cnn_model'])
elif 'Hex_cnn_model' in sys.modules:
    importlib.reload(sys.modules['Hex_cnn_model'])

# ========================================
# 导入CNN模型
# ========================================
try:
    from .Hex_cnn_model import HexCNNTrainer

    print("✓ 已导入HexCNNTrainer（相对导入）")
except ImportError:
    try:
        from Hex_cnn_model import HexCNNTrainer

        print("✓ 已导入HexCNNTrainer（直接导入）")
    except ImportError as e:
        print(f"✗ 导入HexCNNTrainer失败: {e}")
        HexCNNTrainer = None

# 验证方法签名
if HexCNNTrainer:
    print("=" * 60)
    print("验证 train_from_mcts_data 方法签名:")
    try:
        sig = inspect.signature(HexCNNTrainer.train_from_mcts_data)
        print(f"  {sig}")
    except Exception as e:
        print(f"  方法未找到: {e}")
    print("=" * 60)


class CNNAgent:
    """CNN代理 - 可以替代MCTS代理"""

    def __init__(self, board_size=11, model_path=None):
        """
        初始化CNN代理

        Args:
            board_size: 棋盘大小
            model_path: 预训练模型路径
        """
        # 使用全局C++模块变量
        self.has_cpp = HAS_CPP_AVAILABLE
        self.hex_cpp = HEX_CPP_MODULE

        if not self.has_cpp:
            print("⚠ 警告: C++模块不可用，将使用Python实现（速度较慢）")

        self.board_size = board_size
        self.learning_rate = 1e-3

        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.device = device
            self.trainer = HexCNNTrainer(board_size=board_size, device=device)

            if model_path:
                self.load_model(model_path)
                print(f"✓ 已加载CNN模型: {model_path}")
            else:
                print("⚠ 使用未训练的CNN模型")

            self.model_available = True

        except Exception as e:
            print(f"⚠ CNN模型初始化失败: {e}")
            print("将回退到MCTS模式")
            self.model_available = False

        self.rootstate = None
        self.red_move_count = 0
        self.run_time = 0
        self.num_rollouts = 0
        self.node_count = 0

        # 运行时对象
        self.gamestate = None

    def set_gamestate(self, state):
        """设置游戏状态（修复版）"""
        try:
            # 检查是否为C++状态
            if hasattr(state, 'use_cpp') and state.use_cpp:
                print("检测到C++状态对象，转换为Python状态...")
                from game_core import GameState

                # 创建Python版本的状态
                new_state = GameState(size=state.size, use_cpp=False)

                # 获取棋盘
                if hasattr(state, '_cpp_state'):
                    board_list = state._cpp_state.get_board()
                    new_state.board = np.array(board_list).reshape(state.size, state.size)
                else:
                    new_state.board = state.board.copy()

                new_state.toplay = state.toplay
                new_state.size = state.size

                self.rootstate = new_state
                print(f"✓ 已复制游戏状态（C++ -> Python）")

            else:
                # Python状态直接深拷贝
                import copy
                self.rootstate = copy.deepcopy(state)
                print("✓ 已复制游戏状态（Python）")

        except Exception as e:
            print(f"✗ 复制状态失败: {e}")
            import traceback
            traceback.print_exc()

    def search(self, time_budget):
        """
        执行搜索（CNN模型版本）

        Args:
            time_budget: 时间预算
        """
        if not self.model_available:
            print("CNN模型不可用")
            return

        start_time = perf_counter()

        # CNN模型的"搜索"实际上是神经网络前向传播
        self.num_rollouts = 1
        self.node_count = 1
        self.run_time = perf_counter() - start_time

    def best_move(self, think_time=1):
        """获取最佳走法"""
        if not self.model_available or self.rootstate is None:
            print("⚠️ 模型或状态不可用")
            return None

        try:
            with torch.no_grad():
                # 获取合法走法
                if self.has_cpp and self.hex_cpp is not None:
                    cpp_state = self._convert_to_cpp_state(self.rootstate)
                    legal_moves = cpp_state.moves()
                    current_player = cpp_state.turn()
                else:
                    legal_moves = [(x, y) for x in range(self.rootstate.size)
                                   for y in range(self.rootstate.size)
                                   if self.rootstate.board[x, y] == 0]
                    current_player = 1 if self.rootstate.playerJustMoved == 2 else 2

                #  严格验证合法走法
                if not legal_moves:
                    print("⚠️ 没有合法走法")
                    return None

                # 获取模型预测
                move, value = self.trainer.get_best_move(
                    cpp_state if self.has_cpp else self.rootstate,
                    current_player,
                    temperature=0.1
                )

                # 二次验证
                if move is not None:
                    x, y = move

                    # 验证坐标范围
                    if not (0 <= x < self.rootstate.size and 0 <= y < self.rootstate.size):
                        print(f"⚠️ 坐标越界: {move}")
                        return self._get_random_valid_move()

                    # 验证位置是否空闲
                    if self.rootstate.board[x, y] != 0:
                        print(f"⚠️ 位置已占用: {move}, 值={self.rootstate.board[x, y]}")
                        print(f"   当前棋盘状态（部分）:")
                        for i in range(max(0, x - 1), min(self.rootstate.size, x + 2)):
                            print(f"   第{i}行: {self.rootstate.board[i, :]}")
                        return self._get_random_valid_move()

                    # 验证是否在合法走法列表中
                    if move not in legal_moves:
                        print(f"⚠️ 走法不在合法列表中: {move}")
                        return self._get_random_valid_move()

                    print(f"✓ CNN走法: {move}, 评分: {value:.3f}")
                    return move

                return self._get_random_valid_move()

        except Exception as e:
            print(f"✗ CNN异常: {e}")
            import traceback
            traceback.print_exc()
            return self._get_random_valid_move()

    def move(self, move):
        """执行走法"""
        if self.rootstate is None:
            print("⚠️ rootstate未初始化")
            return

        try:
            # 方案1：使用C++状态
            if self.has_cpp and self.hex_cpp is not None:
                cpp_state = self._convert_to_cpp_state(self.rootstate)
                cpp_state.play(move[0], move[1])

                # 同步回Python状态
                self.rootstate.board = np.array(cpp_state.board()).reshape(
                    self.rootstate.size, self.rootstate.size
                )
                self.rootstate.toplay = cpp_state.turn()

                print(f"✓ 状态已更新: 走法 {move}, 下一手 {self.rootstate.toplay}")

            # 方案2：纯Python状态
            else:
                if not (0 <= move[0] < self.rootstate.size and
                        0 <= move[1] < self.rootstate.size):
                    print(f"✗ 无效走法: {move}")
                    return

                if self.rootstate.board[move[0], move[1]] != 0:
                    print(f"✗ 位置已占用: {move}")
                    return

                # 落子
                self.rootstate.board[move[0], move[1]] = self.rootstate.toplay

                # 切换玩家
                self.rootstate.toplay = 3 - self.rootstate.toplay

                print(f"✓ 状态已更新: 走法 {move}, 下一手 {self.rootstate.toplay}")

        except AttributeError as e:
            print(f"更新状态失败（属性错误）: {e}")

        except Exception as e:
            print(f"✗ 更新状态失败: {e}")
            import traceback
            traceback.print_exc()

    def _get_random_valid_move(self):
        """随机合法走法"""
        import random
        if self.rootstate is None:
            return None
        valid_moves = [(x, y) for x in range(self.rootstate.size) for y in range(self.rootstate.size) if
                       self.rootstate.board[x, y] == 0]
        if valid_moves:
            move = random.choice(valid_moves)
            print(f"→ 随机走法: {move}")
            return move
        return None


    def statistics(self):
        """返回搜索统计信息"""
        return (self.num_rollouts, self.node_count, self.run_time)

    def _convert_to_cpp_state(self, python_state):
        """
        将Python游戏状态转换为C++版本

        Args:
            python_state: Python 的 GameState 对象

        Returns:
            C++ 的 GameState 对象
        """
        if not self.has_cpp or self.hex_cpp is None:
            raise ImportError("C++ 模块不可用")

        cpp_state = self.hex_cpp.GameState(python_state.size)

        # 重放所有走法
        for x in range(python_state.size):
            for y in range(python_state.size):
                cell_value = python_state.board[x, y]
                if cell_value == 1:  # RED
                    if cpp_state.get_board()[x * python_state.size + y] == 0:
                        cpp_state.play(x, y)
                elif cell_value == 2:  # BLUE
                    if cpp_state.get_board()[x * python_state.size + y] == 0:
                        cpp_state.play(x, y)

        return cpp_state

    # ========================================
    # 训练检查点（训练时用）
    # ========================================

    def save_checkpoint(self, checkpoint_path, epoch=None, loss=None):
        """
        保存训练检查点（用于恢复训练）

        Args:
            checkpoint_path: 检查点保存路径
            epoch: 当前轮数（可选）
            loss: 当前损失（可选）
        """
        checkpoint = {
            'model_state_dict': self.trainer.model.state_dict(),
            'board_size': self.board_size,
            'learning_rate': self.learning_rate,
        }

        if epoch is not None:
            checkpoint['epoch'] = epoch
        if loss is not None:
            checkpoint['loss'] = loss

        if hasattr(self.trainer, 'optimizer'):
            checkpoint['optimizer_state_dict'] = self.trainer.optimizer.state_dict()
            checkpoint['optimizer_lr'] = self.trainer.optimizer.param_groups[0]['lr']

        torch.save(checkpoint, checkpoint_path)
        print(f"✓ 训练检查点已保存: {checkpoint_path}")

    def load_checkpoint(self, path):
        """
        加载训练检查点（恢复训练）

        Args:
            path: 检查点文件路径

        Returns:
            bool: 是否加载成功
        """
        if not os.path.exists(path):
            print(f"检查点文件不存在: {path}")
            return False

        try:
            # 加载检查点
            checkpoint = torch.load(
                path,
                map_location=self.device,
                weights_only=False
            )

            self.trainer.model.load_state_dict(checkpoint['model_state_dict'])

            if 'optimizer_state_dict' in checkpoint and hasattr(self.trainer, 'optimizer'):
                self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if 'board_size' in checkpoint:
                self.board_size = checkpoint['board_size']
            if 'learning_rate' in checkpoint:
                self.learning_rate = checkpoint['learning_rate']

            # 重新创建优化器和调度器
            if not hasattr(self.trainer, 'optimizer'):
                self.trainer.optimizer = torch.optim.Adam(
                    self.trainer.model.parameters(),
                    lr=checkpoint.get('optimizer_lr', self.learning_rate),
                    weight_decay=1e-4
                )

            if not hasattr(self.trainer, 'scheduler'):
                self.trainer.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.trainer.optimizer,
                    mode='min',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6
                )

            epoch = checkpoint.get('epoch', 0)
            loss = checkpoint.get('loss', 0)

            print(f"✓ 检查点已加载: epoch {epoch}, loss {loss:.4f}")
            return True

        except Exception as e:
            print(f"✗ 加载检查点失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    # ========================================
    # 推理模型（GUI用）
    # ========================================

    def save_model(self, path):
        """
        保存纯净的推理模型

        Args:
            path: 模型保存路径
        """
        torch.save(
            self.trainer.model.state_dict(),
            path
        )
        print(f"✓ 推理模型已保存: {path}")

    def load_model(self, path):
        """
        加载推理模型

        Args:
            path: 模型文件路径

        Returns:
            bool: 是否加载成功
        """
        if not os.path.exists(path):
            print(f"模型文件不存在: {path}")
            return False

        try:
            # 加载文件
            print(f"正在加载模型: {path}")
            loaded_data = torch.load(
                path,
                map_location=self.device,
                weights_only=False
            )

            # 智能判断文件格式
            if isinstance(loaded_data, dict):
                if 'model_state_dict' in loaded_data:
                    # Checkpoint格式
                    print("✓ 检测到checkpoint格式，正在提取模型权重...")
                    state_dict = loaded_data['model_state_dict']

                    # 打印额外信息
                    if 'epoch' in loaded_data:
                        print(f"  训练轮数: {loaded_data['epoch']}")
                    if 'loss' in loaded_data:
                        print(f"  训练损失: {loaded_data['loss']:.4f}")
                    if 'board_size' in loaded_data:
                        expected_size = loaded_data['board_size']
                        if expected_size != self.board_size:
                            print(f"⚠️ 警告: 模型棋盘大小({expected_size})与当前({self.board_size})不匹配")
                            return False
                else:
                    # 直接的state_dict
                    print("✓ 检测到纯模型格式")
                    state_dict = loaded_data
            else:
                # 旧格式
                print("✓ 检测到旧格式模型")
                state_dict = loaded_data

            # 验证state_dict的keys
            model_keys = set(self.trainer.model.state_dict().keys())
            loaded_keys = set(state_dict.keys())

            # 检查缺失的keys
            missing_keys = model_keys - loaded_keys
            unexpected_keys = loaded_keys - model_keys

            if missing_keys:
                print(f"⚠️ 警告: 模型缺少以下参数:")
                for key in list(missing_keys)[:5]:
                    print(f"    {key}")
                if len(missing_keys) > 5:
                    print(f"    ... 还有 {len(missing_keys) - 5} 个")
                return False

            if unexpected_keys:
                print(f"⚠️ 警告: 发现意外的参数:")
                for key in list(unexpected_keys)[:5]:
                    print(f"    {key}")
                if len(unexpected_keys) > 5:
                    print(f"    ... 还有 {len(unexpected_keys) - 5} 个")

            # 加载权重
            self.trainer.model.load_state_dict(state_dict, strict=True)
            self.trainer.model.eval()

            # 清空状态
            self.gamestate = None
            self.rootstate = None

            print(f"✓ 模型权重加载成功！")
            print(f"  参数总数: {sum(p.numel() for p in self.trainer.model.parameters()):,}")

            return True

        except Exception as e:
            print(f"✗ 加载模型失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def convert_checkpoint_to_model(self, checkpoint_path, output_path):
        """
        将checkpoint转换为纯模型文件（用于分发和部署）

        Args:
            checkpoint_path: 检查点文件路径
            output_path: 输出的纯模型路径

        Returns:
            bool: 是否转换成功
        """
        try:
            print(f"正在转换checkpoint: {checkpoint_path}")
            print(f"目标文件: {output_path}")

            # 加载checkpoint
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=False
            )

            # 提取state_dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']

                # 显示checkpoint信息
                print("\nCheckpoint信息:")
                if 'epoch' in checkpoint:
                    print(f"  训练轮数: {checkpoint['epoch']}")
                if 'loss' in checkpoint:
                    print(f"  训练损失: {checkpoint['loss']:.4f}")
                if 'board_size' in checkpoint:
                    print(f"  棋盘大小: {checkpoint['board_size']}")
                if 'learning_rate' in checkpoint:
                    print(f"  学习率: {checkpoint['learning_rate']}")
            else:
                # 已经是纯模型
                print("✓ 输入文件已经是纯模型格式")
                state_dict = checkpoint

            # 保存为纯模型
            torch.save(state_dict, output_path)

            # 验证转换
            print("\n验证转换结果...")
            test_load = torch.load(output_path, map_location=self.device, weights_only=False)

            if isinstance(test_load, dict) and 'model_state_dict' in test_load:
                print("⚠️ 警告: 转换后仍包含checkpoint结构")
                return False

            # 检查文件大小
            original_size = os.path.getsize(checkpoint_path) / 1024 / 1024
            converted_size = os.path.getsize(output_path) / 1024 / 1024

            print(f"\n✓ 模型转换成功！")
            print(f"  原始文件: {original_size:.2f} MB")
            print(f"  转换后: {converted_size:.2f} MB")
            print(
                f"  节省: {original_size - converted_size:.2f} MB ({(1 - converted_size / original_size) * 100:.1f}%)")

            return True

        except Exception as e:
            print(f"✗ 转换失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def train_model(self, num_games=100, save_path='hex_cnn_model.pth',
                    learning_rate=1e-3, num_epochs=50,
                    progress_callback=None, **kwargs):
        """
        训练CNN模型（使用MCTS数据）

        Args:
            num_games: 自我对弈游戏数量
            save_path: 最终模型保存路径
            learning_rate: 学习率
            num_epochs: 训练轮数
            progress_callback: 进度回调函数 callback(current, total, message)
            **kwargs: 捕获额外参数
        """
        # ========================================
        # 添加：设置日志系统
        # ========================================
        import logging
        from datetime import datetime

        # 创建日志目录
        log_dir = Path("training_logs")
        log_dir.mkdir(exist_ok=True)

        # 生成时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'training_{timestamp}.log'

        # 配置日志
        logger = logging.getLogger(f'HexCNN_{timestamp}')
        logger.setLevel(logging.INFO)
        logger.handlers = []  # 清除已有处理器

        # 文件处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 格式化
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        # ========================================
        # 原有代码：参数验证
        # ========================================
        if kwargs:
            warning_msg = f"⚠️ 警告：收到未使用的参数: {kwargs}"
            print(warning_msg)
            logger.warning(warning_msg)

        if not self.model_available:
            error_msg = "CNN模型不可用，无法训练"
            print(error_msg)
            logger.error(error_msg)
            return

        # ========================================
        # 添加：记录训练配置
        # ========================================
        logger.info("=" * 70)
        logger.info("开始训练 CNN 模型")
        logger.info("=" * 70)
        logger.info(f"训练配置:")
        logger.info(f"  游戏数量: {num_games}")
        logger.info(f"  学习率: {learning_rate}")
        logger.info(f"  训练轮数: {num_epochs}")
        logger.info(f"  保存路径: {save_path}")
        logger.info(f"  日志文件: {log_file}")
        logger.info(f"  棋盘大小: {self.board_size}")
        logger.info(f"  设备: {self.device}")
        logger.info("=" * 70)

        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)

        # ========================================
        # 阶段1: 生成MCTS数据
        # ========================================
        if progress_callback:
            progress_callback(0, num_games, "准备生成MCTS训练数据...")

        logger.info(f"\n阶段1: 使用MCTS生成训练数据...")
        logger.info(f"目标游戏数: {num_games}")

        num_positions = self.trainer.generate_training_data_from_mcts(
            num_games=num_games,
            mcts_time=1.0
        )

        if num_positions == 0:
            error_msg = "✗ 数据生成失败"
            print(error_msg)
            logger.error(error_msg)
            if progress_callback:
                progress_callback(0, 1, "数据生成失败")
            return

        logger.info(f"✓ 成功生成 {num_positions} 个训练样本")
        logger.info(f"经验池大小: {len(self.trainer.replay_buffer)}")

        # ========================================
        # 阶段2: 训练网络
        # ========================================
        logger.info(f"\n阶段2: 训练神经网络...")
        logger.info("-" * 70)

        if progress_callback:
            progress_callback(0, num_epochs, f"开始训练，共 {num_positions} 个样本...")

        # 用于绘图
        history = {
            'loss': [],
            'policy_loss': [],
            'value_loss': []
        }

        best_loss = float('inf')

        for epoch in range(num_epochs):
            epoch_losses = []
            policy_losses = []
            value_losses = []

            num_batches = max(1, len(self.trainer.replay_buffer) // self.trainer.batch_size)

            for _ in range(num_batches):
                total_loss, policy_loss, value_loss = self.trainer.train_on_batch_with_policy_target()
                epoch_losses.append(total_loss)
                policy_losses.append(policy_loss)
                value_losses.append(value_loss)

            avg_total_loss = np.mean(epoch_losses)
            avg_policy_loss = np.mean(policy_losses)
            avg_value_loss = np.mean(value_losses)

            # 保存历史
            history['loss'].append(avg_total_loss)
            history['policy_loss'].append(avg_policy_loss)
            history['value_loss'].append(avg_value_loss)

            self.trainer.scheduler.step(avg_total_loss)

            # 获取当前学习率
            current_lr = self.trainer.optimizer.param_groups[0]['lr']

            # ========================================
            # 添加：详细日志输出
            # ========================================
            log_msg = f"Epoch {epoch + 1}/{num_epochs} - "
            log_msg += f"总损失: {avg_total_loss:.4f}, "
            log_msg += f"策略损失: {avg_policy_loss:.4f}, "
            log_msg += f"价值损失: {avg_value_loss:.4f}"

            # 检查是否是最佳模型
            if avg_total_loss < best_loss:
                best_loss = avg_total_loss
                log_msg += " ⭐ 最佳"

                # 保存最佳模型
                best_model_path = checkpoint_dir / f"best_model_{timestamp}.pth"
                self.save_checkpoint(best_model_path, epoch=epoch + 1, loss=avg_total_loss)

            logger.info(log_msg)

            # 保存检查点（每5轮）
            if (epoch + 1) % 5 == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}_{timestamp}.pth"
                self.save_checkpoint(checkpoint_path, epoch=epoch + 1, loss=avg_total_loss)
                logger.info(f"  → 已保存检查点: {checkpoint_path}")

            # 更新进度回调
            if progress_callback:
                progress_callback(
                    epoch + 1,
                    num_epochs,
                    f"训练进度 {epoch + 1}/{num_epochs}\n"
                    f"总损失: {avg_total_loss:.4f}\n"
                    f"策略损失: {avg_policy_loss:.4f}\n"
                    f"价值损失: {avg_value_loss:.4f}"
                )

            # 每5轮打印详细信息
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs} | "
                      f"总损失: {avg_total_loss:.4f} | "
                      f"策略: {avg_policy_loss:.4f} | "
                      f"价值: {avg_value_loss:.4f} | "
                      f"学习率: {current_lr:.6f}")

        # ========================================
        # 阶段3: 保存模型和生成报告
        # ========================================
        logger.info("-" * 70)
        logger.info("训练完成！")
        logger.info(f"最终总损失: {history['loss'][-1]:.4f}")
        logger.info(f"最终策略损失: {history['policy_loss'][-1]:.4f}")
        logger.info(f"最终价值损失: {history['value_loss'][-1]:.4f}")
        logger.info(f"最佳总损失: {best_loss:.4f}")
        logger.info(f"总训练样本: {len(self.trainer.replay_buffer)}")
        logger.info("=" * 70)

        # 保存最终模型
        if save_path:
            self.save_model(save_path)
            final_msg = f"🎉 最终推理模型已导出: {save_path}"
            print(final_msg)
            logger.info(final_msg)

        # ========================================
        # 添加：保存训练曲线
        # ========================================
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

            epochs = range(1, len(history['loss']) + 1)

            # 总损失
            ax1.plot(epochs, history['loss'], 'b-', linewidth=2, marker='o', markersize=3)
            ax1.set_xlabel('Epoch', fontsize=12)
            ax1.set_ylabel('Total Loss', fontsize=12)
            ax1.set_title('总损失曲线', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)

            # 策略损失
            ax2.plot(epochs, history['policy_loss'], 'g-', linewidth=2, marker='s', markersize=3)
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('Policy Loss', fontsize=12)
            ax2.set_title('策略损失曲线', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)

            # 价值损失
            ax3.plot(epochs, history['value_loss'], 'r-', linewidth=2, marker='^', markersize=3)
            ax3.set_xlabel('Epoch', fontsize=12)
            ax3.set_ylabel('Value Loss', fontsize=12)
            ax3.set_title('价值损失曲线', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)

            plt.tight_layout()

            plot_path = log_dir / f'training_curve_{timestamp}.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()

            logger.info(f"训练曲线已保存: {plot_path}")

        except Exception as e:
            logger.warning(f"保存训练曲线失败: {e}")

        # ========================================
        # 添加：生成训练报告
        # ========================================
        try:
            report_path = log_dir / f'training_report_{timestamp}.txt'

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 70 + "\n")
                f.write("CNN训练报告\n")
                f.write("=" * 70 + "\n\n")

                f.write(f"训练时间: {timestamp}\n")
                f.write(f"游戏数量: {num_games}\n")
                f.write(f"训练轮数: {num_epochs}\n")
                f.write(f"学习率: {learning_rate}\n")
                f.write(f"训练样本数: {len(self.trainer.replay_buffer)}\n")
                f.write(f"棋盘大小: {self.board_size}\n\n")

                f.write("-" * 70 + "\n")
                f.write("训练结果\n")
                f.write("-" * 70 + "\n")
                f.write(f"最终总损失: {history['loss'][-1]:.4f}\n")
                f.write(f"最终策略损失: {history['policy_loss'][-1]:.4f}\n")
                f.write(f"最终价值损失: {history['value_loss'][-1]:.4f}\n")
                f.write(f"最佳总损失: {best_loss:.4f}\n\n")

                f.write("-" * 70 + "\n")
                f.write("详细历史记录\n")
                f.write("-" * 70 + "\n")
                f.write(f"{'Epoch':<8} {'Total Loss':<12} {'Policy Loss':<14} {'Value Loss':<12}\n")
                f.write("-" * 70 + "\n")

                for i in range(len(history['loss'])):
                    epoch = i + 1
                    total_loss = history['loss'][i]
                    policy_loss = history['policy_loss'][i]
                    value_loss = history['value_loss'][i]

                    f.write(f"{epoch:<8} {total_loss:<12.4f} {policy_loss:<14.4f} {value_loss:<12.4f}\n")

                f.write("=" * 70 + "\n")

            logger.info(f"训练报告已保存: {report_path}")

        except Exception as e:
            logger.warning(f"生成训练报告失败: {e}")

        print("=" * 60)
        print("训练完成！")
        print(f"总训练样本: {len(self.trainer.replay_buffer)}")
        print(f"日志文件: {log_file}")
        print(f"训练报告: {report_path}")
        print("=" * 60)


# 集成到Gui类的方法
class CNNGuiIntegration:
    """GUI集成助手"""

    @staticmethod
    def add_cnn_controls(gui_instance):
        """
        为现有GUI添加CNN相关控制

        Args:
            gui_instance: Gui类的实例
        """
        from tkinter import Frame, Button, Label

        # 在开局库面板后添加CNN面板
        panel_cnn = Frame(gui_instance.notebook, bg=gui_instance.colors['bg'])
        gui_instance.notebook.add(panel_cnn, text='CNN模型')

        # 标题
        Label(
            panel_cnn, text='深度学习模型', font=('KaiTi', 14, 'bold'),
            foreground='white', bg=gui_instance.colors['bg'], pady=10
        ).pack(side='top', fill='x')

        # 模型状态
        model_status = Label(
            panel_cnn, font=('KaiTi', 12), fg='white', bg=gui_instance.colors['bg'],
            text="CNN模型状态:\n未加载", justify='left'
        )
        model_status.pack(side='top', fill='x', pady=5)
        gui_instance.cnn_status_label = model_status

        # 控制按钮框架
        button_frame = Frame(panel_cnn, bg=gui_instance.colors['bg'])
        button_frame.pack(side='top', fill='x', pady=10)

        # 加载模型按钮
        load_btn = Button(
            button_frame, text='加载模型', font=('KaiTi', 10),
            bg=gui_instance.colors['button'], fg='white',
            command=lambda: CNNGuiIntegration.load_cnn_model(gui_instance)
        )
        load_btn.pack(side='left', padx=2, pady=5)

        # 训练模型按钮
        train_btn = Button(
            button_frame, text='训练模型', font=('KaiTi', 10),
            bg=gui_instance.colors['button'], fg='white',
            command=lambda: CNNGuiIntegration.train_cnn_model(gui_instance)
        )
        train_btn.pack(side='left', padx=2, pady=5)

        # 切换到CNN按钮
        switch_btn = Button(
            button_frame, text='使用CNN', font=('KaiTi', 10),
            bg=gui_instance.colors['button'], fg='white',
            command=lambda: CNNGuiIntegration.switch_to_cnn(gui_instance)
        )
        switch_btn.pack(side='left', padx=2, pady=5)

        # 转换按钮
        convert_btn = Button(
            button_frame, text='转换模型', font=('KaiTi', 10),
            bg=gui_instance.colors['button'], fg='white',
            command=lambda: CNNGuiIntegration.convert_checkpoint(gui_instance)
        )
        convert_btn.pack(side='left', padx=2, pady=5)

        # 训练信息显示
        Label(
            panel_cnn, text='训练设置', font=('KaiTi', 14, 'bold'),
            foreground='white', bg=gui_instance.colors['bg'], pady=10
        ).pack(side='top', fill='x')

        from tkinter import Entry

        settings_frame = Frame(panel_cnn, bg=gui_instance.colors['bg'])
        settings_frame.pack(side='top', fill='x', pady=5)

        Label(settings_frame, text='游戏数:', fg='white', bg=gui_instance.colors['bg']).pack(side='left')
        games_entry = Entry(settings_frame, width=10)
        games_entry.insert(0, '100')
        games_entry.pack(side='left', padx=5)
        gui_instance.cnn_games_entry = games_entry

        # 训练进度
        progress_label = Label(
            panel_cnn, font=('KaiTi', 12), fg='white', bg=gui_instance.colors['bg'],
            text="训练进度:\n等待开始", justify='left'
        )
        progress_label.pack(side='top', fill='x', pady=5)
        gui_instance.cnn_progress_label = progress_label

    @staticmethod
    def load_cnn_model(gui_instance):
        """加载CNN模型"""
        from tkinter import filedialog, messagebox

        filename = filedialog.askopenfilename(
            filetypes=[("PyTorch模型", "*.pth"), ("所有文件", "*.*")],             title="加载CNN模型"
        )

        if filename:
            try:
                cnn_agent = CNNAgent(board_size=gui_instance.game.size, model_path=filename)
                cnn_agent.set_gamestate(gui_instance.game)

                gui_instance.AGENTS['CNN'] = lambda state=None: cnn_agent
                gui_instance.cnn_agent = cnn_agent

                gui_instance.cnn_status_label.config(
                    text=f"CNN模型状态:\n已加载\n文件: {os.path.basename(filename)}"
                )

                messagebox.showinfo("成功", f"CNN模型已从 {filename} 加载")

            except Exception as e:
                messagebox.showerror("错误", f"加载失败: {str(e)}")
                import traceback
                traceback.print_exc()

    @staticmethod
    def convert_checkpoint(gui_instance):
        """转换checkpoint为纯模型"""
        from tkinter import filedialog, messagebox

        # 选择checkpoint文件
        checkpoint_path = filedialog.askopenfilename(
            filetypes=[("PyTorch Checkpoint", "*.pth"), ("所有文件", "*.*")],
            title="选择要转换的Checkpoint文件"
        )

        if not checkpoint_path:
            return

        # 选择输出路径
        output_path = filedialog.asksaveasfilename(
            defaultextension=".pth",
            filetypes=[("PyTorch模型", "*.pth"), ("所有文件", "*.*")],
            title="保存转换后的模型",
            initialfile="converted_model.pth"
        )

        if not output_path:
            return

        try:
            # 创建临时agent进行转换
            if not hasattr(gui_instance, 'cnn_agent'):
                gui_instance.cnn_agent = CNNAgent(board_size=gui_instance.game.size)

            # 执行转换
            success = gui_instance.cnn_agent.convert_checkpoint_to_model(
                checkpoint_path,
                output_path
            )

            if success:
                messagebox.showinfo(
                    "转换成功",
                    f"Checkpoint已成功转换为纯模型！\n\n"
                    f"输出文件: {output_path}\n\n"
                    f"现在可以在GUI中加载这个模型了。"
                )
            else:
                messagebox.showerror("转换失败", "转换过程中出现错误，请查看控制台输出。")

        except Exception as e:
            messagebox.showerror("错误", f"转换失败: {str(e)}")
            import traceback
            traceback.print_exc()

    @staticmethod
    def train_cnn_model(gui_instance):
        """训练CNN模型"""
        from tkinter import messagebox, filedialog
        import threading

        try:
            num_games = int(gui_instance.cnn_games_entry.get())
        except ValueError:
            messagebox.showerror("错误", "请输入有效的游戏数量")
            return

        # 选择保存路径
        save_path = filedialog.asksaveasfilename(
            defaultextension=".pth",
            filetypes=[("PyTorch模型", "*.pth"), ("所有文件", "*.*")],
            title="保存训练后的模型"
        )

        if not save_path:
            return

        def progress_callback(current, total, message):
            """更新训练进度"""
            try:
                gui_instance.cnn_progress_label.config(
                    text=f"训练进度: {current}/{total}\n{message}"
                )
                gui_instance.root.update()
            except:
                pass

        def train_thread():
            """训练线程"""
            try:
                if not hasattr(gui_instance, 'cnn_agent'):
                    gui_instance.cnn_agent = CNNAgent(board_size=gui_instance.game.size)

                gui_instance.cnn_agent.train_model(
                    num_games=num_games,
                    save_path=save_path,
                    progress_callback=progress_callback
                )

                gui_instance.cnn_progress_label.config(
                    text=f"训练完成！\n模型已保存到:\n{os.path.basename(save_path)}"
                )

                messagebox.showinfo("成功", f"训练完成！模型已保存到 {save_path}")

            except Exception as e:
                gui_instance.cnn_progress_label.config(
                    text=f"训练失败:\n{str(e)}"
                )
                messagebox.showerror("错误", f"训练失败: {str(e)}")
                import traceback
                traceback.print_exc()

        # 在新线程中训练
        thread = threading.Thread(target=train_thread, daemon=True)
        thread.start()

    @staticmethod
    def switch_to_cnn(gui_instance):
        """切换到CNN模式"""
        from tkinter import messagebox

        if not hasattr(gui_instance, 'cnn_agent'):
            messagebox.showerror("错误", "请先加载CNN模型")
            return

        # 切换AI
        if gui_instance.player1 == 'AI':
            gui_instance.ai1 = 'CNN'
            messagebox.showinfo("成功", "红方AI已切换到CNN模式")

        elif gui_instance.player2 == 'AI':
            gui_instance.ai2 = 'CNN'
            messagebox.showinfo("成功", "蓝方AI已切换到CNN模式")

        else:
            messagebox.showinfo("提示", "当前没有AI玩家")


# ========================================
# 模块测试和诊断函数
# ========================================

def test_imports():
    """测试所有导入是否正常"""
    print("=" * 60)
    print("模块导入测试")
    print("=" * 60)

    print(f"\n项目根目录: {Path(__file__).parent.parent}")
    print(f"CNN目录: {Path(__file__).parent}")
    print(f"\nsys.path 前3项:")
    for i, p in enumerate(sys.path[:3], 1):
        print(f"  {i}. {p}")

    # 测试GameState
    print(f"\n{'='*60}")
    print("GameState导入测试:")
    if _GAME_CORE_AVAILABLE and GameState is not None:
        print(f"  ✓ GameState可用")
        print(f"  模块位置: {GameState.__module__}")
        try:
            test_game = GameState(size=11)
            print(f"  ✓ 成功创建 GameState(size=11)")
            print(f"  棋盘形状: {test_game.board.shape}")
        except Exception as e:
            print(f"  ✗ 创建GameState失败: {e}")
    else:
        print(f"  ✗ GameState不可用")

    # 测试hex_cpp
    print(f"\n{'='*60}")
    print("hex_cpp导入测试:")
    if HAS_CPP_AVAILABLE and HEX_CPP_MODULE is not None:
        print(f"  ✓ hex_cpp可用")
        print(f"  模块位置: {HEX_CPP_MODULE.__file__}")
        try:
            test_cpp = HEX_CPP_MODULE.GameState(11)
            print(f"  ✓ 成功创建 hex_cpp.GameState(11)")
        except Exception as e:
            print(f"  ✗ 创建C++ GameState失败: {e}")
    else:
        print(f"  ✗ hex_cpp不可用")

    # 测试HexCNNTrainer
    print(f"\n{'='*60}")
    print("HexCNNTrainer导入测试:")
    if HexCNNTrainer is not None:
        print(f"  ✓ HexCNNTrainer可用")
        try:
            trainer = HexCNNTrainer(board_size=11)
            print(f"  ✓ 成功创建 HexCNNTrainer(board_size=11)")
            print(f"  设备: {trainer.device}")
        except Exception as e:
            print(f"  ✗ 创建HexCNNTrainer失败: {e}")
    else:
        print(f"  ✗ HexCNNTrainer不可用")

    # 测试CNNAgent
    print(f"\n{'='*60}")
    print("CNNAgent创建测试:")
    try:
        agent = CNNAgent(board_size=11)
        print(f"  ✓ 成功创建 CNNAgent(board_size=11)")
        print(f"  模型可用: {agent.model_available}")
        print(f"  C++可用: {agent.has_cpp}")

        # 测试set_gamestate
        if _GAME_CORE_AVAILABLE and GameState is not None:
            test_game = GameState(size=11)
            agent.set_gamestate(test_game)
            print(f"  ✓ set_gamestate执行成功")

    except Exception as e:
        print(f"  ✗ 创建CNNAgent失败: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*60}")
    print("测试完成！")
    print("=" * 60)


def test_model_conversion():
    """测试模型转换功能"""
    print("\n" + "=" * 60)
    print("模型转换测试")
    print("=" * 60)

    try:
        agent = CNNAgent(board_size=11)

        # 创建临时checkpoint
        temp_checkpoint = "test_checkpoint.pth"
        agent.save_checkpoint(temp_checkpoint, epoch=1, loss=0.5)
        print(f"✓ 临时checkpoint已创建: {temp_checkpoint}")

        # 转换为纯模型
        temp_model = "test_model.pth"
        success = agent.convert_checkpoint_to_model(temp_checkpoint, temp_model)

        if success:
            print(f"✓ 转换成功")

            # 测试加载转换后的模型
            agent2 = CNNAgent(board_size=11)
            load_success = agent2.load_model(temp_model)

            if load_success:
                print(f"✓ 转换后的模型加载成功")
            else:
                print(f"✗ 转换后的模型加载失败")

            # 清理临时文件
            if os.path.exists(temp_checkpoint):
                os.remove(temp_checkpoint)
            if os.path.exists(temp_model):
                os.remove(temp_model)
            print(f"✓ 临时文件已清理")
        else:
            print(f"✗ 转换失败")

    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()

    print("=" * 60)


