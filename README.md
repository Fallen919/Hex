# 海克斯棋 CNN训练系统

基于TCP分布式自我对弈的CNN模型训练系统，支持C++加速和GPU训练。

## &#x20;快速开始

### 方法1: GUI训练（推荐）

1. 在PyCharm中打开项目

2. 运行 `hex_gui.py`

3. 在GUI右侧"CNN功能区"中：

   * 设置"训练局数"（如100、1000）

   * 点击 训练模型 按钮

4. 观看实时对弈 + 后台训练



### 方法2: 命令行训练

```bash
# 1. 启动训练服务器
python tcp_training_manager.py --mode online --port 9999

# 2. 启动自我对弈客户端（可开启多个）
python tcp_selfplay_client.py --games 1000 --time 1.5
```

## &#x20;项目结构

```
hex_cnn/
├── hex_gui.py                  # 主GUI界面（含训练功能）
├── game_core.py                # 游戏核心（C++加速）
├── mcts_agent.py               # MCTS智能体
├── tcp_training_manager.py     # TCP训练管理器
├── tcp_selfplay_server.py      # TCP服务器
├── tcp_selfplay_client.py      # TCP客户端
├── opening_book.py             # 开局库
├── hex_cpp.cp312-win_amd64.pyd # C++加速模块
├── cnn/                        # CNN模型
│   ├── Hex_cnn_model.py        # 模型定义
│   ├── Hex_cnn_integration.py  # CNN智能体
│   └── train_cnn_mcts.py       # 训练脚本
├── training_data/              # 训练数据
├── models/                     # 模型检查点
└── checkpoints/                # CNN检查点
```

## &#x20;训练参数调整

在GUI训练前，可以调整：

* **训练局数**：每个客户端游戏数（默认1000）

* **学习率**：神经网络学习率（默认0.001）

* **批次大小**：训练批次大小（默认32）

* **训练轮数**：每批数据训练轮数（默认50）

## &#x20;性能优化



**GPU训练：**

* 系统自动检测CUDA

* 日志显示：`设备: cuda` 表示使用GPU

## &#x20;训练监控

**GUI日志：**

* 实时显示训练进度

* 对局统计（已完成局数、训练位置数）

**文件监控：**

* `training_data/training_stats.json` - 训练统计

* `training_data/batch_*.npz` - 训练数据批次

* `models/checkpoint_step_*.pth` - 模型检查点

## 高级配置

### 修改客户端数量

编辑 `hex_gui.py` 第744行（`train_cnn_model`方法）：

```python
for i in range(2):  # 改为更多客户端
```

### 修改思考时间

编辑 `hex_gui.py` 第749行：

```python
'--time', '1.5',  # 改为其他值（秒）
```

## 🐛 故障排除

**端口占用：**

```bash
# 查看9999端口占用
netstat -ano | findstr "9999"

# 清理占用进程（GUI会自动清理）
```

**训练未开始：**

* 检查控制台是否有 `[服务器] 服务器就绪` 消息

* 检查是否有防火墙阻止

**棋盘跳动：**

* 已修复：只显示第一个客户端的对局

* 其他客户端在后台训练

**观战切局说明：**

* 多客户端并行训练时，服务器会在主观战对局结束后自动切换到另一活跃对局

* GUI现在使用 `game_id` 区分不同对局，并在日志中提示“新对局/切换观战”

* 若看到胜者变化，请先确认是否已经切换到新的 `game_id`

## &#x20;技术文档

* [技术文档.md](技术文档.md) - 系统架构详解

* [cnn/cnn技术文档.md](cnn/cnn技术文档.md) - CNN模型说明

## &#x20;典型训练流程

1. **数据收集**：TCP分布式自我对弈（2个客户端 × 1000局）

2. **数据保存**：自动保存到 `training_data/batch_*.npz`

3. **模型训练**：每10秒从缓冲区采样批次训练

4. **模型保存**：每100步保存检查点到 `models/`

5. **GUI观战**：实时显示第一个客户端的对局

## 胜利规则说明

* 当前项目规则：**红方连接上下边界获胜，蓝方连接左右边界获胜**

* Python主逻辑已按该规则统一判定

* `cpp_module/gamestate.cpp` 也已同步该规则；若你本地重编译 `hex_cpp`，C++后端将与Python完全一致

## 依赖

```bash
pip install numpy torch tkinter
```

