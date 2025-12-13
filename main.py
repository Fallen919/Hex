"""
海克斯棋主程序入口
整合所有模块
"""

import sys
from hex_gui import main

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序已退出")
        sys.exit(0)
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
