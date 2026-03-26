"""
使用MCTS数据训练CNN模型
推荐的训练流程
"""

from Hex_cnn_model import HexCNNTrainer
import torch


def main():
    print("=" * 60)
    print("Hex CNN 模型训练（MCTS监督学习）")
    print("=" * 60)

    # 检查设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")

    if device == 'cuda':
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")

    # 创建训练器
    trainer = HexCNNTrainer(board_size=11, device=device)

    # ===================================================================
    # 阶段1: 生成MCTS训练数据
    # ===================================================================
    print("\n" + "=" * 60)
    print("阶段1: 生成MCTS训练数据")
    print("=" * 60)

    # 参数说明：
    # num_games: 游戏数量，越多越好（建议200-500）
    # mcts_time: 每步思考时间（秒），越长棋力越强（建议1-3秒）

    # 可以根据你的计算资源调整这些参数：
    # - 快速测试: num_games=50, mcts_time=1.0
    # - 标准训练: num_games=200, mcts_time=2.0
    # - 高质量训练: num_games=500, mcts_time=3.0

    num_positions = trainer.generate_training_data_from_mcts(
        num_games=200,  # 生成200局游戏
        mcts_time=2.0  # 每步思考2秒
    )

    if num_positions == 0:
        print("[错误] 数据生成失败，退出")
        return

    # ===================================================================
    # 阶段2: 训练神经网络
    # ===================================================================
    print("\n" + "=" * 60)
    print("阶段2: 训练神经网络")
    print("=" * 60)

    trainer.train_from_mcts_data(
        num_epochs=100,  # 训练100轮
        verbose=True  # 显示详细进度
    )

    # ===================================================================
    # 阶段3: 保存最终模型
    # ===================================================================
    print("\n" + "=" * 60)
    print("阶段3: 保存模型")
    print("=" * 60)

    trainer.save_model('hex_cnn_mcts_trained.pth')

    # ===================================================================
    # 完成
    # ===================================================================
    print("\n" + "=" * 60)
    print("[信息] 训练完成！")
    print("=" * 60)
    print(f"\n模型已保存:")
    print(f"  - hex_cnn_best.pth (最佳模型 - 推荐使用)")
    print(f"  - hex_cnn_mcts_trained.pth (最终模型)")
    print(f"\n使用方法:")
    print(f"  1. 运行 GUI: python Hex_cnn.py")
    print(f"  2. 切换到 'CNN模型' 选项卡")
    print(f"  3. 点击 '加载模型' 并选择 hex_cnn_best.pth")
    print(f"  4. 选择 'CNN' 算法开始对弈")
    print(f"\n训练统计:")
    print(f"  - 总训练样本: {len(trainer.replay_buffer)}")
    print(f"  - 批次大小: {trainer.batch_size}")
    print(f"  - 使用设备: {device}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()