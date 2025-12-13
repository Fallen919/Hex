"""
海克斯棋棋谱管理
支持标准HEX格式的读写
"""

from datetime import datetime
import re


class GameRecord:
    """海克斯棋棋谱类"""

    def __init__(self, board_size):
        self.board_size = board_size
        self.moves = []  # [(x, y, color, move_number), ...]
        self.game_info = {
            'start_time': datetime.now().isoformat(),
            'board_size': board_size,
            'red_player': 'Human',
            'blue_player': 'AI',
            'result': 'unknown',
            'location': '重庆',
            'tournament': '练习赛'
        }

    def add_move(self, position, color):
        """添加一步棋"""
        move_number = len(self.moves) + 1
        self.moves.append((position[0], position[1], color, move_number))

    def set_result(self, winner):
        """设置游戏结果"""
        self.game_info['result'] = winner
        self.game_info['end_time'] = datetime.now().isoformat()

    def set_game_info(self, red_player=None, blue_player=None, location=None, tournament=None):
        """设置比赛信息"""
        if red_player:
            self.game_info['red_player'] = red_player
        if blue_player:
            self.game_info['blue_player'] = blue_player
        if location:
            self.game_info['location'] = location
        if tournament:
            self.game_info['tournament'] = tournament

    def _pos_to_coordinate(self, x, y):
        """
        将位置转换为标准坐标格式
        例如: (0, 0) -> (A,1)
        """
        col_letter = chr(ord('A') + y)
        row_number = x + 1
        return f"({col_letter},{row_number})"

    def _coordinate_to_pos(self, coord_str):
        """
        将坐标字符串转换为位置
        例如: "(A,1)" -> (0, 0)
        """
        # 去除括号和空格
        coord_str = coord_str.strip("() ")
        parts = coord_str.split(",")

        if len(parts) != 2:
            raise ValueError(f"无效的坐标格式: {coord_str}")

        col_letter = parts[0].strip().upper()
        row_number = int(parts[1].strip())

        y = ord(col_letter) - ord('A')
        x = row_number - 1

        return (x, y)

    def to_hex_format(self):
        """
        转换为标准海克斯棋谱格式
        格式: {[HEX][红方 R][蓝方 B][结果][时间 地点][赛事]
               R(A,1);B(B,2);R(C,3);...}
        """
        start_time = datetime.fromisoformat(self.game_info['start_time'])
        time_str = start_time.strftime("%Y.%m.%d %H:%M")

        # 转换结果
        if self.game_info['result'] == '红色':
            result = '先手胜'
        elif self.game_info['result'] == '蓝色':
            result = '后手胜'
        else:
            result = '未完成'

        # 构建信息行
        info_line = (f"[HEX][{self.game_info['red_player']} R]"
                     f"[{self.game_info['blue_player']} B][{result}]"
                     f"[{time_str} {self.game_info['location']}]"
                     f"[{self.game_info['tournament']}]")

        # 构建走法行
        moves_parts = []
        for x, y, color, move_num in self.moves:
            coord = self._pos_to_coordinate(x, y)
            color_prefix = 'R' if color == 'red' else 'B'
            move_str = f"{color_prefix}{coord}"
            moves_parts.append(move_str)

        moves_line = ";".join(moves_parts)

        return f"{{{info_line}\n{moves_line}}}"

    def save_to_file(self, filename):
        """保存为标准海克斯棋谱文件"""
        if not filename.endswith('.txt'):
            filename = filename.rsplit('.', 1)[0] + '.txt'

        try:
            with open(filename, 'w', encoding='gb2312') as f:
                f.write(self.to_hex_format())
        except Exception:
            # 如果gb2312编码失败，使用utf-8
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(self.to_hex_format())

    def load_from_file(self, filename):
        """从标准海克斯棋谱文件加载"""
        content = None

        # 尝试多种编码
        for encoding in ['gb2312', 'gbk', 'utf-8']:
            try:
                with open(filename, 'r', encoding=encoding) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue

        if content is None:
            raise ValueError("无法读取文件，编码格式不支持")

        self.from_hex_format(content)

    def from_hex_format(self, hex_text):
        """从标准海克斯棋谱格式加载"""
        # 去除首尾的大括号
        hex_text = hex_text.strip()
        if hex_text.startswith('{') and hex_text.endswith('}'):
            hex_text = hex_text[1:-1]

        # 分割信息行和走法行
        lines = hex_text.strip().split('\n')
        if len(lines) < 2:
            raise ValueError("棋谱格式错误：至少需要两行（信息行和走法行）")

        info_line = lines[0]
        moves_line = lines[1] if len(lines) > 1 else ""

        # 解析信息行
        self._parse_info_line(info_line)

        # 解析走法行
        if moves_line.strip():
            self._parse_moves_line(moves_line)

    def _parse_info_line(self, info_line):
        """解析信息行"""
        # 使用正则表达式提取信息
        # 格式: [HEX][红方 R][蓝方 B][结果][时间 地点][赛事]

        pattern = r'\[([^\]]+)\]'
        matches = re.findall(pattern, info_line)

        if len(matches) < 6:
            # 尝试简化解析
            print(f"警告: 信息行格式不完整，使用默认值")
            return

        # matches[0] 应该是 "HEX"
        # matches[1] 是红方信息，格式: "玩家名 R"
        # matches[2] 是蓝方信息，格式: "玩家名 B"
        # matches[3] 是结果
        # matches[4] 是时间和地点
        # matches[5] 是赛事

        try:
            red_info = matches[1].strip()
            if red_info.endswith(' R'):
                self.game_info['red_player'] = red_info[:-2].strip()
            else:
                self.game_info['red_player'] = red_info

            blue_info = matches[2].strip()
            if blue_info.endswith(' B'):
                self.game_info['blue_player'] = blue_info[:-2].strip()
            else:
                self.game_info['blue_player'] = blue_info

            result = matches[3].strip()
            if result == '先手胜':
                self.game_info['result'] = '红色'
            elif result == '后手胜':
                self.game_info['result'] = '蓝色'
            else:
                self.game_info['result'] = result

            # 解析时间和地点
            time_location = matches[4].strip()
            parts = time_location.split()
            if len(parts) >= 2:
                # 假设格式是 "日期 时间 地点"
                self.game_info['location'] = parts[-1]

            self.game_info['tournament'] = matches[5].strip()

        except Exception as e:
            print(f"解析信息行时出错: {e}")

    def _parse_moves_line(self, moves_line):
        """解析走法行"""
        # 清空现有走法
        self.moves = []

        # 分割走法，格式: R(A,1);B(B,2);R(C,3);...
        move_strs = moves_line.split(';')

        for move_num, move_str in enumerate(move_strs, 1):
            move_str = move_str.strip()
            if not move_str:
                continue

            try:
                # 第一个字符是颜色 (R 或 B)
                if move_str[0] not in ['R', 'B', 'r', 'b']:
                    continue

                color_char = move_str[0].upper()
                color = 'red' if color_char == 'R' else 'blue'

                # 剩余部分是坐标
                coord_str = move_str[1:].strip()
                x, y = self._coordinate_to_pos(coord_str)

                self.moves.append((x, y, color, move_num))

            except Exception as e:
                print(f"解析走法 '{move_str}' 时出错: {e}")
                continue

    def get_moves_text(self):
        """获取走法的文本表示"""
        text = f"海克斯棋谱\n"
        text += f"=" * 40 + "\n"
        text += f"棋盘大小: {self.board_size}x{self.board_size}\n"
        text += f"红方: {self.game_info['red_player']}\n"
        text += f"蓝方: {self.game_info['blue_player']}\n"
        text += f"结果: {self.game_info['result']}\n"
        text += f"地点: {self.game_info['location']}\n"
        text += f"赛事: {self.game_info['tournament']}\n"
        text += f"=" * 40 + "\n\n"

        for x, y, color, move_num in self.moves:
            coord = chr(ord('A') + y) + str(x + 1)
            color_name = "红" if color == "red" else "蓝"
            text += f"{move_num:3d}. {color_name:2s} {coord:4s}\n"

        return text

    def clear(self):
        """清空棋谱"""
        self.moves = []
        self.game_info['start_time'] = datetime.now().isoformat()
        self.game_info['result'] = 'unknown'

    def get_move_count(self):
        """获取走法数量"""
        return len(self.moves)

    def get_last_move(self):
        """获取最后一步走法"""
        if len(self.moves) == 0:
            return None
        return self.moves[-1]

    def remove_last_move(self):
        """删除最后一步走法"""
        if len(self.moves) > 0:
            return self.moves.pop()
        return None


# 测试代码
if __name__ == "__main__":
    # 创建棋谱
    record = GameRecord(11)

    # 添加几步棋
    record.add_move((5, 5), 'red')
    record.add_move((4, 6), 'blue')
    record.add_move((6, 4), 'red')
    record.add_move((5, 6), 'blue')

    # 设置结果
    record.set_result('红色')

    # 打印棋谱
    print(record.get_moves_text())
    print("\n标准格式:")
    print(record.to_hex_format())

    # 保存和加载测试
    record.save_to_file('test_game.txt')

    new_record = GameRecord(11)
    new_record.load_from_file('test_game.txt')
    print("\n加载后的棋谱:")
    print(new_record.get_moves_text())
