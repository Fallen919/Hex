"""
CNN模块包
"""

try:
    from .Hex_cnn_integration import CNNAgent
    from .Hex_cnn_model import HexCNNTrainer

    __all__ = ['CNNAgent', 'HexCNNTrainer']

    print("[信息] CNN包初始化成功")

except ImportError as e:
    print(f"[警告] CNN包部分模块导入失败: {e}")
    __all__ = []
