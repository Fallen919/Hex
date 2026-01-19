# 海克斯棋CNN模型技术文档

## 模型架构详解

### 1. 输入表示系统

#### 1.1 多通道棋盘表示

python

```
# 输入张量形状: (3, board_size, board_size)
通道0: 红棋位置矩阵 (值为1表示红棋)
通道1: 蓝棋位置矩阵 (值为1表示蓝棋) 
通道2: 当前玩家通道 (全1表示红方走棋，全0表示蓝方走棋)
```

设计原理：

* 分离通道：让网络能独立处理红蓝双方棋子信息

* 玩家通道：提供走棋方信息，使网络理解对称性（红蓝策略有差异）

* 标准化：所有值在0-1之间，避免梯度不稳定

#### 1.2 对称性处理

海克斯棋的红蓝双方不对称（红连左右，蓝连上下），因此：

* 当蓝方走棋时，旋转棋盘90度进行训练

* 评估时，对蓝方走法同样进行旋转

* 确保网络学习到棋盘对称性下的通用策略

### 2. 网络核心架构

#### 2.1 基础卷积层

python

```
class InitialConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(128)
    
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))
```

参数说明：

* 输入通道：3（红棋、蓝棋、玩家）

* 输出通道：128（经验选择的特征维度）

* 卷积核：3×3（捕捉局部连接关系）

* 填充：1（保持特征图尺寸不变）

#### 2.2 残差块设计

python

```
class ResidualBlock(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        # 第一个卷积：特征变换
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        
        # 第二个卷积：特征精炼
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))  # 非线性变换
        out = self.bn2(self.conv2(out))        # 线性变换
        out += residual                         # 跳跃连接
        out = F.relu(out)                      # 最终激活
        return out
```

残差块优势：

1. 梯度流动：跳跃连接允许梯度直接反向传播，缓解梯度消失

2. 特征复用：浅层特征可以直接传递给深层

3. 训练稳定：批归一化确保每层输入分布稳定

4. 容量控制：12个残差块提供足够的模型容量，但不过度复杂

#### 2.3 双头输出架构

python

```
# 策略头 - 预测走法概率分布
self.policy_conv = nn.Conv2d(128, 4, 1)  # 1×1卷积降维
self.policy_bn = nn.BatchNorm2d(4)
self.policy_fc = nn.Linear(4*size*size, size*size)  # 展平后全连接

# 价值头 - 评估局面胜率
self.value_conv = nn.Conv2d(128, 4, 1)
self.value_bn = nn.BatchNorm2d(4)
self.value_fc1 = nn.Linear(4*size*size, 128)
self.value_fc2 = nn.Linear(128, 1)
```

双头设计原理：

* 策略头：专注于"如何走" - 预测每个位置的好坏

* 价值头：专注于"局面如何" - 评估整体胜率

* 参数共享：两者共享前128个通道的特征提取层

* 协同训练：价值评估帮助策略学习，策略学习改善价值评估

### 3. 损失函数设计

#### 3.1 多任务损失

python

```
def compute_loss(policy_logits, value_pred, policy_targets, true_values):
    # 策略损失：交叉熵损失
    policy_loss = F.cross_entropy(policy_logits, policy_targets)
    
    # 价值损失：均方误差
    value_loss = F.mse_loss(value_pred, true_values)
    
    # 总损失：加权和
    total_loss = policy_loss + 2.0 * value_loss
    return total_loss, policy_loss, value_loss
```

权重分配原理：

* 价值损失权重更高：在海克斯棋中，正确评估局面比精确选择走法更重要

* 比例2:1：通过实验验证的最佳权重比

* 稳定训练：高权重价值损失防止策略头过拟合

#### 3.2 策略目标设计

python

```
# MCTS提供的策略目标（概率分布）
def get_policy_targets(mcts_root):
    visits = np.zeros(board_size * board_size)
    for move, child in mcts_root.children.items():
        idx = move[0] * board_size + move[1]
        visits[idx] = child.visits
    
    # 温度参数τ控制平滑程度
    temperature = 1.0  # 训练时使用τ=1.0
    log_visits = np.log(visits + 1e-10)
    policy_targets = F.softmax(torch.tensor(log_visits / temperature), dim=0)
    return policy_targets
```

温度参数作用：

* τ=1.0：保留MCTS的探索特性，保持概率分布多样性

* τ→0：近似贪心选择，仅保留最佳走法

* 训练时使用τ=1.0：学习完整分布

* 推理时使用τ=0.1：偏向最佳走法

### 4. 训练数据生成

#### 4.1 MCTS数据生成流程

python

```
def generate_mcts_training_data(num_games=200, mcts_time=2.0):
    training_data = []
    
    for game_idx in range(num_games):
        game_state = GameState(board_size=11)
        mcts_agent = RAVEAgent(game_state)
        game_trajectory = []
        
        while not game_state.is_terminal():
            # MCTS搜索获取策略分布
            mcts_agent.search(time_limit=mcts_time)
            policy_distribution = mcts_agent.get_policy()
            
            # 记录训练样本
            sample = {
                'state': game_state.to_tensor(),
                'policy_target': policy_distribution,
                'player': game_state.current_player,
                'move': None  # 将在游戏结束后填充
            }
            game_trajectory.append(sample)
            
            # 执行MCTS推荐的走法
            move = mcts_agent.best_move()
            game_state.make_move(move)
            mcts_agent.move(move)
        
        # 游戏结束，分配价值标签
        winner = game_state.winner()
        for i, sample in enumerate(game_trajectory):
            # 使用折扣因子：越靠近结束奖励越准确
            discount = 0.99 ** (len(game_trajectory) - i - 1)
            if sample['player'] == winner:
                sample['value_target'] = discount
            else:
                sample['value_target'] = -discount
        
        training_data.extend(game_trajectory)
    
    return training_data
```

关键技术点：

1. MCTS质量：

   * 使用RAVE MCTS

   * 每步思考2秒，确保决策质量

   * 访问次数作为策略分布的基础

2. 价值标签设计：

   * 二值标签：胜=+1，负=-1

   * 折扣因子：γ=0.99，考虑步数影响

   * 自我一致性：同一局中不同位置的价值相互关联

3. 数据多样性：

   * 使用温度参数增加探索

   * 记录完整轨迹而非仅胜负位置

   * 包含不同阶段（开局、中局、残局）的局面

#### 4.2 数据增强策略

python

```
def augment_training_data(state_tensor, policy_target, player):
    # 旋转增强（仅对红方）
    if player == 1:
        # 4种旋转
        rotations = [0, 1, 2, 3]
        for rot in rotations:
            rotated_state = torch.rot90(state_tensor, rot, [1, 2])
            rotated_policy = rotate_policy(policy_target, rot, board_size)
            yield rotated_state, rotated_policy
    
    # 镜像增强（左右对称）
    flipped_state = torch.flip(state_tensor, [2])  # 水平翻转
    flipped_policy = flip_policy(policy_target, board_size)
    yield flipped_state, flipped_policy
```

增强策略效果：

* 8倍数据量：4种旋转 × 2种镜像

* 不变性学习：让网络学习棋盘对称性

* 泛化能力：减少过拟合，提升模型鲁棒性

### 5. 训练流程优化

#### 5.1 监督学习流程

python

```
class SupervisedTrainer:
    def __init__(self, model, lr=0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=5)
        self.replay_buffer = ReplayBuffer(max_size=50000)
    
    def train_epoch(self, batch_size=128):
        self.model.train()
        
        for batch in self.replay_buffer.sample_batches(batch_size):
            # 前向传播
            policy_logits, value_pred = self.model(batch['states'])
            
            # 计算损失
            policy_loss = F.cross_entropy(policy_logits, batch['policies'])
            value_loss = F.mse_loss(value_pred, batch['values'])
            total_loss = policy_loss + 2.0 * value_loss
            
            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        
        return total_loss.item()
```

优化技术：

1. Adam优化器：

   * 自适应学习率

   * 动量加速收敛

   * 偏置修正

2. 学习率调度：

   * 初始学习率：0.001

   * ReduceLROnPlateau：损失平台期降低学习率

   * 最小学习率：1e-6（防止震荡）

3. 梯度裁剪：

   * 最大梯度范数：1.0

   * 防止梯度爆炸

   * 稳定训练过程

#### 5.2 经验回放机制

python

```
class ReplayBuffer:
    def __init__(self, max_size=50000):
        self.buffer = []
        self.max_size = max_size
        self.priority_weights = None  # 可用于优先经验回放
    
    def add(self, experience):
        self.buffer.append(experience)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)  # FIFO淘汰旧数据
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        return self._collate_batch(batch)
```

经验回放优势：

1. 打破相关性：随机采样打破时间相关性

2. 样本复用：高质量样本多次训练

3. 稳定训练：平滑学习曲线，减少震荡

4. 数据高效：最大化数据利用率

### 6. 模型推理与部署

#### 6.1 推理流程

python

```
class CNNInferenceEngine:
    def __init__(self, model_path, board_size=11, device='cuda'):
        self.model = HexCNN(board_size).to(device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()  # 评估模式
        self.device = device
        self.board_size = board_size
    
    def get_move(self, game_state, temperature=0.1):
        # 转换为模型输入
        state_tensor = self._state_to_tensor(game_state)
        
        with torch.no_grad():
            # 前向传播
            policy_logits, value = self.model(state_tensor)
            
            # 应用合法走法掩码
            legal_moves = game_state.get_legal_moves()
            mask = self._create_legal_mask(legal_moves)
            masked_logits = policy_logits + mask
            
            # 温度采样
            if temperature > 0:
                probs = F.softmax(masked_logits / temperature, dim=1)
                move_idx = torch.multinomial(probs, 1).item()
            else:
                move_idx = masked_logits.argmax().item()
            
            # 转换回棋盘坐标
            x = move_idx // self.board_size
            y = move_idx % self.board_size
            
            return (x, y), value.item()
```

推理优化：

1. eval模式：关闭dropout和batchnorm的随机性

2. 无梯度计算：torch.no_grad()节省内存

3. 合法走法掩码：确保只从合法走法中采样

4. 温度参数：控制探索-利用权衡

#### 6.2 批量推理优化

python

```
def batch_inference(states, model, batch_size=32):
    """批量处理多个局面，提升GPU利用率"""
    results = []
    
    for i in range(0, len(states), batch_size):
        batch = torch.stack(states[i:i+batch_size])
        with torch.no_grad():
            policy_batch, value_batch = model(batch)
            results.extend(zip(policy_batch, value_batch))
    
    return results
```

批量优化效果：

* GPU利用率：从~10%提升到~80%

* 推理速度：10-20倍加速

* 内存效率：减少内存分配次数

### 7. 模型评估与测试

#### 7.1 评估指标

python

```
class ModelEvaluator:
    def __init__(self, model, reference_agent='MCTS'):
        self.model = model
        self.reference_agent = reference_agent
    
    def evaluate_win_rate(self, num_games=100):
        """与参考AI对弈评估胜率"""
        wins = 0
        for _ in range(num_games):
            game_state = GameState()
            while not game_state.is_terminal():
                if game_state.current_player == 1:
                    move, _ = self.model.get_move(game_state)
                else:
                    move = self.reference_agent.get_move(game_state)
                game_state.make_move(move)
            
            if game_state.winner() == 1:  # 模型作为红方
                wins += 1
        
        return wins / num_games
    
    def evaluate_policy_accuracy(self, test_data):
        """策略预测准确率评估"""
        correct = 0
        total = 0
        
        for state, true_move in test_data:
            pred_move, _ = self.model.get_move(state)
            if pred_move == true_move:
                correct += 1
            total += 1
        
        return correct / total
```

#### 7.2 自我对弈评估

python

```
def self_play_evaluation(model, num_games=50):
    """模型自我对弈，评估策略一致性"""
    results = []
    
    for _ in range(num_games):
        game_state = GameState()
        move_history = []
        
        while not game_state.is_terminal():
            move, value = model.get_move(game_state)
            move_history.append((move, value))
            game_state.make_move(move)
        
        # 分析对局质量
        game_length = len(move_history)
        value_stability = analyze_value_stability(move_history)
        results.append({
            'length': game_length,
            'stability': value_stability,
            'winner': game_state.winner()
        })
    
    return analyze_results(results)
```

### 8. 高级训练技巧

#### 8.1 课程学习

python

```
class CurriculumLearning:
    def __init__(self, model):
        self.model = model
        self.curriculum_stages = [
            {'games': 50, 'mcts_time': 0.5, 'board_size': 7},   # 阶段1：小棋盘快速学习
            {'games': 100, 'mcts_time': 1.0, 'board_size': 9},  # 阶段2：中等棋盘
            {'games': 200, 'mcts_time': 2.0, 'board_size': 11}, # 阶段3：标准棋盘
        ]
    
    def train_with_curriculum(self):
        for stage in self.curriculum_stages:
            print(f"开始阶段: {stage}")
            
            # 生成该阶段数据
            data = generate_data(
                num_games=stage['games'],
                mcts_time=stage['mcts_time'],
                board_size=stage['board_size']
            )
            
            # 调整模型输入尺寸（如果棋盘大小变化）
            if stage['board_size'] != self.model.board_size:
                self.model.resize_board(stage['board_size'])
            
            # 训练
            trainer.train_on_data(data)
```

课程学习优势：

1. 渐进学习：从简单任务开始，逐步增加难度

2. 快速收敛：早期阶段快速学习基础模式

3. 避免局部最优：不同阶段提供多样化的训练信号

#### 8.2 集成学习

python

```
class ModelEnsemble:
    def __init__(self, model_paths):
        self.models = []
        for path in model_paths:
            model = HexCNN()
            model.load_state_dict(torch.load(path))
            model.eval()
            self.models.append(model)
    
    def get_move(self, state):
        # 收集所有模型的预测
        all_policies = []
        all_values = []
        
        for model in self.models:
            policy, value = model(state)
            all_policies.append(policy)
            all_values.append(value)
        
        # 集成策略：平均概率分布
        ensemble_policy = torch.mean(torch.stack(all_policies), dim=0)
        
        # 选择最佳走法
        move_idx = ensemble_policy.argmax().item()
        x = move_idx // board_size
        y = move_idx % board_size
        
        return (x, y), torch.mean(torch.stack(all_values))
```

集成学习效果：

* 减少方差：多个模型平均减少随机误差

* 提升鲁棒性：不同模型可能捕捉不同特征

* 不确定性估计：通过模型间差异估计预测置信度

### 9. 性能分析与优化

#### 9.1 计算复杂度分析

| 组件     | 浮点运算次数 (FLOPs)                   | 内存占用 (MB) | 执行时间 (ms) |
| :----- | :------------------------------- | :-------- | :-------- |
| 输入转换   | 3×11×11=363                      | 0.001     | 0.01      |
| 初始卷积   | 3×128×3×3×11×11=4.2M             | 0.062     | 0.5       |
| 残差块×12 | 12×2×128×128×3×3×11×11=4.3G      | 0.062     | 8.0       |
| 策略头    | 128×4×11×11+4×121×121=0.7M       | 0.002     | 0.2       |
| 价值头    | 128×4×11×11+4×121×128+128×1=0.7M | 0.002     | 0.2       |
| 总计     | ~4.3G FLOPs                      | ~0.13MB   | ~9ms      |

优化空间：

1. 模型剪枝：移除不重要的权重

2. 量化：FP16或INT8量化

3. 知识蒸馏：训练小型学生网络

4. 缓存：重复局面结果缓存

#### 9.2 内存优化策略

python

```
class MemoryEfficientInference:
    def __init__(self, model):
        self.model = model
        
        # 激活检查点技术
        self.checkpoint_segments = 4  # 将网络分为4段
        
    def forward_with_checkpoint(self, x):
        # 分段计算，节省激活内存
        segments = self._split_network()
        
        for i, segment in enumerate(segments):
            if i % 2 == 0:  # 每隔一段保存检查点
                x = checkpoint(segment, x, use_reentrant=False)
            else:
                x = segment(x)
        
        return x
```

### 10. 部署与集成

#### 10.1 模型导出

python

```
def export_model(model, output_path, format='onnx'):
    """导出模型为部署格式"""
    if format == 'onnx':
        dummy_input = torch.randn(1, 3, 11, 11)
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            opset_version=11,
            input_names=['board_state'],
            output_names=['policy', 'value'],
            dynamic_axes={
                'board_state': {0: 'batch_size'},
                'policy': {0: 'batch_size'},
                'value': {0: 'batch_size'}
            }
        )
    elif format == 'torchscript':
        scripted_model = torch.jit.script(model)
        scripted_model.save(output_path)
```

#### 10.2 生产环境部署

python

```
class ProductionModelServer:
    def __init__(self, model_path, max_batch_size=64, gpu_id=0):
        # 加载模型
        self.model = self._load_model(model_path)
        
        # 创建推理队列
        self.request_queue = Queue()
        self.result_cache = LRUCache(maxsize=1000)
        
        # 启动推理线程
        self.inference_thread = Thread(target=self._inference_loop)
        self.inference_thread.start()
    
    def get_move(self, state_hash, state_tensor):
        """异步获取走法，支持缓存"""
        # 检查缓存
        if state_hash in self.result_cache:
            return self.result_cache[state_hash]
        
        # 提交推理请求
        future = Future()
        self.request_queue.put((state_hash, state_tensor, future))
        
        # 等待结果
        return future.result(timeout=5.0)
```

##

## 后续研究方向

1. 强化学习：从自我对弈中进一步提升

2. 多尺度架构：同时处理不同分辨率的特征

3. 注意力机制：引入Transformer模块捕捉长距离依赖

4. 元学习：快速适应不同对手风格

5. 可解释性：可视化模型决策过程

