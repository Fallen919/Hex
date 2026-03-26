"""
海克斯棋GUI界面
完全保留原有UI设计，仅更新模块导入
"""

from tkinter import *
from tkinter import ttk, messagebox, filedialog
import numpy as np
from copy import deepcopy
import socket
import pickle
import time
import subprocess
import sys
import threading
from pathlib import Path
import json

# 导入模块化的组件
from game_core import GameState as gamestate
from game_record import GameRecord
from opening_book import OpeningBook
from mcts_agent import MCTSAgent, RAVEAgent

# 为了向后兼容
mctsagent = MCTSAgent
rave_mctsagent = RAVEAgent

# 尝试导入CNN模块
try:
    # 首先尝试从cnn文件夹导入
    from cnn.Hex_cnn_integration import CNNAgent

    CNN_AVAILABLE = True
    print("[信息] CNN模块已加载（从cnn文件夹）")
except ImportError:
    try:
        # 如果失败，尝试直接导入
        from Hex_cnn_integration import CNNAgent

        CNN_AVAILABLE = True
        print("[信息] CNN模块已加载（从当前目录）")
    except ImportError as e:
        print(f"[警告] CNN模块不可用: {e}")
        CNN_AVAILABLE = False


        # 创建占位类
        class CNNAgent:
            def __init__(self, board_size=11, model_path=None):
                raise ImportError("CNN模块未安装")


class Gui:
    agent_type = {1: "UCT", 2: "RAVE", 3: "CNN"}
    AGENTS = {
        "UCT": mctsagent,
        "RAVE": rave_mctsagent,
        "CNN": CNNAgent
    }

    def __init__(self, root, agent_name='UCT'):
        self.root = root
        self.root.geometry('1300x650+100+20')
        self.agent_name = agent_name

        # 初始化游戏
        self.game = gamestate(11)

        # 为游戏添加棋谱记录
        self.game.game_record = GameRecord(11)

        # 初始化AI代理
        try:
            if agent_name == "CNN" and CNN_AVAILABLE:
                self.agent = CNNAgent(board_size=11)
            else:
                self.agent = self.AGENTS[agent_name]()
        except (KeyError, ImportError):
            print("未知代理或CNN不可用，默认使用UCT模式")
            self.agent_name = "UCT"
            self.agent = self.AGENTS["UCT"]()

        self.agent.set_gamestate(self.game)

        # 为代理添加开局库
        if not hasattr(self.agent, 'opening_book'):
            self.agent.opening_book = OpeningBook(11)
            self.agent.red_move_count = 0

        self.time = 1
        self.root.configure(bg='#363636')
        self.colors = {
            'red': '#ff0000',
            'blue': '#0000ff',
            'red_edge': '#ff6666',
            'blue_edge': '#6666ff',
            'bg': '#363636',
            'grid': '#999999',
            'text': '#ffffff',
            'button': '#555555',
            'button_active': '#777777'
        }
        global bg
        bg = self.colors['bg']
        self.last_move = None
        self.replay_mode = False
        self.replay_index = 0

        # 悔棋
        self.history_states = []
        self.hex_board = []
        self.max_history = 50

        # 初始化界面
        self.setup_ui()
        # 如果CNN可用，添加CNN控制面板
        if CNN_AVAILABLE:
            self.setup_cnn_controls()

    def setup_ui(self):
        self.frame_board = Frame(self.root)
        self.canvas = Canvas(self.frame_board, bg=bg)
        self.scroll_y = ttk.Scrollbar(self.frame_board, orient=VERTICAL)
        self.scroll_x = ttk.Scrollbar(self.frame_board, orient=HORIZONTAL)
        self.notebook = ttk.Notebook(self.frame_board, width=420)
        self.panel_game = Frame(self.notebook, bg=self.colors['bg'])
        self.panel_record = Frame(self.notebook, bg=self.colors['bg'])
        self.panel_opening = Frame(self.notebook, bg=self.colors['bg'])

        self.game_size_value = IntVar()
        self.game_time_value = IntVar()
        self.game_turn_value = IntVar()
        self.switch_agent_value = IntVar()
        self.switch_agent_value.set(1)
        self.game_turn_value.set(1)
        self.turn = {1: '红色', 2: '蓝色'}
        self.game_size = Scale(self.panel_game)
        self.game_time = Scale(self.panel_game)
        self.game_turn = Scale(self.panel_game)
        self.generate = Button(self.panel_game)
        self.reset_board = Button(self.panel_game)
        self.hex_board = []
        self.game_size_value.set(11)
        self.game_time_value.set(1)
        self.size = self.game_size_value.get()
        self.time = self.game_time_value.get()
        self.board = self.game.board
        self.board = np.int_(self.board).tolist()
        self.array_to_hex(self.board)
        self.blue_side()
        self.red_side()

        # 设置控件
        self.setup_game_controls()
        self.setup_record_controls()
        self.setup_opening_controls()

        # 布局
        self.frame_board.pack(fill=BOTH)
        self.notebook.add(self.panel_game, text='游戏选项')
        self.notebook.add(self.panel_record, text='棋谱管理')
        self.notebook.add(self.panel_opening, text='开局库')
        self.notebook.pack(side=RIGHT, fill=X)
        self.canvas.configure(width=800, height=550, bg=bg, cursor='hand2')
        self.canvas.pack(side=RIGHT, fill=Y)
        self.canvas.configure(yscrollcommand=self.scroll_y.set)
        self.scroll_y.configure(command=self.canvas.yview)
        self.scroll_x.configure(command=self.canvas.xview)

        # 绑定事件
        self.canvas.bind('<1>', self.mouse_click)

        # 添加CNN控制面板
        ##self.setup_cnn_controls()

    def pts(self):
        return [[85, 50], [105, 65], [105, 90], [85, 105], [65, 90], [65, 65]]

    def hexagon(self, points, color):
        if color == 0:
            hx = self.canvas.create_polygon(points[0], points[1], points[2],
                                            points[3], points[4], points[5],
                                            fill=self.colors['grid'], outline='black', width=2, activefill='cyan')
        elif color == 1:
            hx = self.canvas.create_polygon(points[0], points[1], points[2],
                                            points[3], points[4], points[5],
                                            fill=self.colors['red'], outline='black', width=2, activefill='cyan')
        elif color == 2:
            hx = self.canvas.create_polygon(points[0], points[1], points[2],
                                            points[3], points[4], points[5],
                                            fill=self.colors['blue'], outline='black', width=2, activefill='cyan')
        elif color == 3:
            hx = self.canvas.create_polygon(points[0], points[1], points[2],
                                            points[3], points[4], points[5],
                                            fill=self.colors['blue_edge'], outline='black', width=2)
        else:
            hx = self.canvas.create_polygon(points[0], points[1], points[2],
                                            points[3], points[4], points[5],
                                            fill=self.colors['red_edge'], outline='black', width=2)
        return hx

    def genrow(self, points, colors):
        x_offset = 40
        row = []
        temp_array = []
        for i in range(len(colors)):
            for point in points:
                temp_points_x = point[0] + x_offset * i
                temp_points_y = point[1]
                temp_array.append([temp_points_x, temp_points_y])
            if colors[i] == 0:
                hx = self.hexagon(temp_array, 0)
            elif colors[i] == 1:
                hx = self.hexagon(temp_array, 1)
            else:
                hx = self.hexagon(temp_array, 2)
            row.append(hx)
            temp_array = []
        return row

    def array_to_hex(self, array):
        """根据棋盘数组绘制六边形棋盘"""
        self.canvas.delete('all')
        if not hasattr(self, 'hex_board'):
            self.hex_board = []
        self.hex_board.clear()

        self.board = np.int_(array).tolist()

        initial_offset = 20
        y_offset = 40
        temp = []

        for i in range(len(self.board)):
            points = self.pts()
            for point in points:
                point[0] += initial_offset * i
                point[1] += y_offset * i
                temp.append([point[0], point[1]])
            row = self.genrow(temp, self.board[i])
            temp.clear()
            self.hex_board.append(row)

        # 绘制边界
        self.blue_side()
        self.red_side()

    def red_side(self):
        init_points = self.pts()
        label_x, label_y = 0, 0
        temp_list = []
        for pt in init_points:
            pt[0] -= 60
            pt[1] -= 40
        for t in range(len(init_points)):
            init_points[t][0] += 40
            label_x += init_points[t][0]
            label_y += init_points[t][1]
        label_x /= 6
        label_y /= 6
        for i in range(len(self.board)):
            self.hexagon(init_points, 4)
            label_x, label_y = 0, 0
            for pt in init_points:
                temp_list.append([pt[0] + (len(self.board) + 1) * 20, pt[1] + (len(self.board) + 1) * 40])
            self.hexagon(temp_list, 4)

            for pt in temp_list:
                label_x += pt[0]
                label_y += pt[1]
            label_x /= 6
            label_y /= 6
            self.canvas.create_text(label_x, label_y, fill=self.colors['text'], font="Times 20 bold",
                                    text=chr(ord('A') + i))
            temp_list.clear()
            for j in range(len(init_points)):
                init_points[j][0] += 40
                label_x += init_points[j][0]
                label_y += init_points[j][1]
            label_x /= 6
            label_y /= 6

    def blue_side(self):
        init_points = self.pts()
        for pt in init_points:
            pt[0] -= 40
        for pt in init_points:
            pt[0] -= 20
            pt[1] -= 40
        label_x, label_y = 0, 0
        init_offset = 20
        y_offset = 40
        temp_list = []
        for i in range(len(self.board)):
            for pt in range(len(init_points)):
                init_points[pt][0] += init_offset
                init_points[pt][1] += y_offset
                label_x += init_points[pt][0]
                label_y += init_points[pt][1]
            label_x /= 6
            label_y /= 6
            self.hexagon(init_points, 3)
            self.canvas.create_text(label_x, label_y, fill=self.colors['text'], font="Times 20 bold",
                                    text=str(len(self.board) - i))
            label_x, label_y = 0, 0
            for j in init_points:
                temp_list.append([j[0] + (len(self.board) + 1) * 40, j[1]])
            self.hexagon(temp_list, 3)
            temp_list.clear()

    def setup_opening_controls(self):
        """设置开局库控制面板"""
        self.panel_opening.configure(bg=bg)

        Label(
            self.panel_opening, text='开局库状态', font=('KaiTi', 14, 'bold'),
            foreground='white', bg=bg, pady=10
        ).pack(side=TOP, fill=X)

        self.opening_info = Label(
            self.panel_opening, font=('KaiTi', 12), fg='white', bg=bg,
            text="开局库已启用\n红棋前3步将使用高胜率位置\n蓝棋使用MCTS搜索", justify=LEFT
        )
        self.opening_info.pack(side=TOP, fill=X, pady=5)

        Label(
            self.panel_opening, text='开局库设置', font=('KaiTi', 14, 'bold'),
            foreground='white', bg=bg, pady=10
        ).pack(side=TOP, fill=X)

        opening_buttons = Frame(self.panel_opening, bg=bg)
        opening_buttons.pack(side=TOP, fill=X, pady=5)

        save_opening_btn = Button(
            opening_buttons, text='保存开局库', font=('KaiTi', 10),
            bg=self.colors['button'], fg='white', command=self.save_opening_book
        )
        save_opening_btn.pack(side=LEFT, padx=2, pady=5)

        load_opening_btn = Button(
            opening_buttons, text='加载开局库', font=('KaiTi', 10),
            bg=self.colors['button'], fg='white', command=self.load_opening_book
        )
        load_opening_btn.pack(side=LEFT, padx=2, pady=5)

        Label(
            self.panel_opening, text='推荐开局位置', font=('KaiTi', 14, 'bold'),
            foreground='white', bg=bg, pady=10
        ).pack(side=TOP, fill=X)

        opening_text_frame = Frame(self.panel_opening, bg=bg)
        opening_text_frame.pack(side=TOP, fill=BOTH, expand=True, padx=5, pady=5)

        self.opening_text = Text(opening_text_frame, height=15, width=35, font=('Courier', 9),
                                bg='white', fg='black', wrap=WORD)
        opening_scrollbar = Scrollbar(opening_text_frame, bg=self.colors['button'])
        opening_scrollbar.pack(side=RIGHT, fill=Y)
        self.opening_text.pack(side=LEFT, fill=BOTH, expand=True)
        self.opening_text.config(yscrollcommand=opening_scrollbar.set)
        opening_scrollbar.config(command=self.opening_text.yview)

        self.update_opening_display()

    def setup_cnn_controls(self):
        """设置CNN控制面板 - 完整版"""
        self.panel_cnn = Frame(self.notebook, bg=self.colors['bg'])
        self.notebook.add(self.panel_cnn, text='CNN模型')

        # 标题
        Label(
            self.panel_cnn, text='深度学习模型', font=('KaiTi', 16, 'bold'),
            foreground='white', bg=self.colors['bg'], pady=10
        ).pack(side=TOP, fill=X)

        # 模型状态
        status_frame = Frame(self.panel_cnn, bg=self.colors['bg'])
        status_frame.pack(side=TOP, fill=X, pady=5, padx=10)

        Label(
            status_frame, text='模型状态:', font=('KaiTi', 12, 'bold'),
            fg='white', bg=self.colors['bg']
        ).pack(side=TOP, anchor=W)

        self.cnn_status_label = Label(
            status_frame, font=('KaiTi', 11), fg='yellow',
            bg=self.colors['bg'], text="未加载模型\n请先加载或训练模型",
            justify=LEFT, wraplength=350
        )
        self.cnn_status_label.pack(side=TOP, fill=X, pady=5, anchor=W)

        # 分隔线
        ttk.Separator(self.panel_cnn, orient=HORIZONTAL).pack(fill=X, pady=10)

        # 模型操作按钮
        Label(
            self.panel_cnn, text='模型操作', font=('KaiTi', 14, 'bold'),
            foreground='white', bg=self.colors['bg'], pady=5
        ).pack(side=TOP, fill=X)

        button_frame = Frame(self.panel_cnn, bg=self.colors['bg'])
        button_frame.pack(side=TOP, fill=X, pady=5, padx=10)

        # 第一行按钮
        row1 = Frame(button_frame, bg=self.colors['bg'])
        row1.pack(side=TOP, fill=X, pady=2)

        self.load_model_btn = Button(
            row1, text='加载模型', font=('KaiTi', 10),
            bg=self.colors['button'], fg='white',
            command=self.load_cnn_model, width=15, cursor='hand2'
        )
        self.load_model_btn.pack(side=LEFT, padx=2, pady=3)

        self.save_model_btn = Button(
            row1, text='保存模型', font=('KaiTi', 10),
            bg=self.colors['button'], fg='white',
            command=self.save_cnn_model, width=15, cursor='hand2',
            state=DISABLED
        )
        self.save_model_btn.pack(side=LEFT, padx=2, pady=3)

        # 第二行按钮
        row2 = Frame(button_frame, bg=self.colors['bg'])
        row2.pack(side=TOP, fill=X, pady=2)

        self.train_model_btn = Button(
            row2, text='训练模型', font=('KaiTi', 10),
            bg=self.colors['button'], fg='white',
            command=self.train_cnn_model, width=15, cursor='hand2'
        )
        self.train_model_btn.pack(side=LEFT, padx=2, pady=3)

        self.eval_model_btn = Button(
            row2, text='评估模型', font=('KaiTi', 10),
            bg=self.colors['button'], fg='white',
            command=self.evaluate_cnn_model, width=15, cursor='hand2',
            state=DISABLED
        )
        self.eval_model_btn.pack(side=LEFT, padx=2, pady=3)

        # 分隔线
        ttk.Separator(self.panel_cnn, orient=HORIZONTAL).pack(fill=X, pady=10)

        # 训练设置
        Label(
            self.panel_cnn, text='训练设置', font=('KaiTi', 14, 'bold'),
            foreground='white', bg=self.colors['bg'], pady=5
        ).pack(side=TOP, fill=X)

        settings_frame = Frame(self.panel_cnn, bg=self.colors['bg'])
        settings_frame.pack(side=TOP, fill=X, pady=5, padx=10)

        # 游戏数量
        game_setting = Frame(settings_frame, bg=self.colors['bg'])
        game_setting.pack(side=TOP, fill=X, pady=3)

        Label(game_setting, text='训练局数:', fg='white',
              bg=self.colors['bg'], font=('KaiTi', 10), width=12, anchor=W).pack(side=LEFT)
        self.cnn_games_entry = Entry(game_setting, width=10, font=('Arial', 10))
        self.cnn_games_entry.insert(0, '100')
        self.cnn_games_entry.pack(side=LEFT, padx=5)

        Label(game_setting, text='局', fg='white',
              bg=self.colors['bg'], font=('KaiTi', 10)).pack(side=LEFT)

        # 学习率
        lr_setting = Frame(settings_frame, bg=self.colors['bg'])
        lr_setting.pack(side=TOP, fill=X, pady=3)

        Label(lr_setting, text='学习率:', fg='white',
              bg=self.colors['bg'], font=('KaiTi', 10), width=12, anchor=W).pack(side=LEFT)
        self.cnn_lr_entry = Entry(lr_setting, width=10, font=('Arial', 10))
        self.cnn_lr_entry.insert(0, '0.001')
        self.cnn_lr_entry.pack(side=LEFT, padx=5)

        # 批次大小
        batch_setting = Frame(settings_frame, bg=self.colors['bg'])
        batch_setting.pack(side=TOP, fill=X, pady=3)

        Label(batch_setting, text='批次大小:', fg='white',
              bg=self.colors['bg'], font=('KaiTi', 10), width=12, anchor=W).pack(side=LEFT)
        self.cnn_batch_entry = Entry(batch_setting, width=10, font=('Arial', 10))
        self.cnn_batch_entry.insert(0, '32')
        self.cnn_batch_entry.pack(side=LEFT, padx=5)

        # Epoch数量
        epoch_setting = Frame(settings_frame, bg=self.colors['bg'])
        epoch_setting.pack(side=TOP, fill=X, pady=3)

        Label(epoch_setting, text='训练轮数:', fg='white',
              bg=self.colors['bg'], font=('KaiTi', 10), width=12, anchor=W).pack(side=LEFT)
        self.cnn_epoch_entry = Entry(epoch_setting, width=10, font=('Arial', 10))
        self.cnn_epoch_entry.insert(0, '10')
        self.cnn_epoch_entry.pack(side=LEFT, padx=5)

        # 分隔线
        ttk.Separator(self.panel_cnn, orient=HORIZONTAL).pack(fill=X, pady=10)

        # 训练进度
        Label(
            self.panel_cnn, text='训练进度', font=('KaiTi', 14, 'bold'),
            foreground='white', bg=self.colors['bg'], pady=5
        ).pack(side=TOP, fill=X)

        progress_frame = Frame(self.panel_cnn, bg=self.colors['bg'])
        progress_frame.pack(side=TOP, fill=BOTH, expand=True, padx=10, pady=5)

        # 进度条
        self.cnn_progress_var = IntVar(value=0)
        self.cnn_progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.cnn_progress_var,
            maximum=100,
            mode='determinate'
        )
        self.cnn_progress_bar.pack(side=TOP, fill=X, pady=5)

        # 进度文本
        self.cnn_progress_label = Label(
            progress_frame, font=('KaiTi', 10), fg='white',
            bg=self.colors['bg'], text="等待开始训练...",
            justify=LEFT, wraplength=350
        )
        self.cnn_progress_label.pack(side=TOP, fill=X, pady=5)

        # 训练日志
        log_label = Label(
            progress_frame, text='训练日志:', font=('KaiTi', 11, 'bold'),
            fg='white', bg=self.colors['bg']
        )
        log_label.pack(side=TOP, anchor=W, pady=(10, 2))

        log_text_frame = Frame(progress_frame, bg=self.colors['bg'])
        log_text_frame.pack(side=TOP, fill=BOTH, expand=True)

        self.cnn_log_text = Text(
            log_text_frame, height=8, width=40, font=('Courier', 8),
            bg='#1e1e1e', fg='#00ff00', wrap=WORD
        )
        log_scrollbar = Scrollbar(log_text_frame, bg=self.colors['button'])
        log_scrollbar.pack(side=RIGHT, fill=Y)
        self.cnn_log_text.pack(side=LEFT, fill=BOTH, expand=True)
        self.cnn_log_text.config(yscrollcommand=log_scrollbar.set)
        log_scrollbar.config(command=self.cnn_log_text.yview)

        self.cnn_log_text.insert(1.0, "CNN模块已初始化\n等待操作...\n")
        self.cnn_log_text.config(state=DISABLED)

        # 初始化CNN代理为None
        self.cnn_agent = None

    def add_cnn_log(self, message):
        """添加CNN日志"""
        self.cnn_log_text.config(state=NORMAL)
        self.cnn_log_text.insert(END, f"{message}\n")
        self.cnn_log_text.see(END)
        self.cnn_log_text.config(state=DISABLED)
        self.root.update()

    def load_cnn_model(self):
        """加载CNN模型"""
        filename = filedialog.askopenfilename(
            filetypes=[
                ("PyTorch模型", "*.pth"),
                ("H5模型", "*.h5"),
                ("所有文件", "*.*")
            ],
            title="加载CNN模型"
        )

        if filename:
            try:
                self.add_cnn_log(f"正在加载模型: {filename}")

                cnn_agent = CNNAgent(board_size=self.game.size, model_path=filename)
                cnn_agent.set_gamestate(self.game)

                # 添加开局库
                if not hasattr(cnn_agent, 'opening_book'):
                    from opening_book import OpeningBook
                    cnn_agent.opening_book = OpeningBook(self.game.size)
                    cnn_agent.red_move_count = 0

                self.cnn_agent = cnn_agent

                # 更新状态
                model_name = filename.split('/')[-1]
                self.cnn_status_label.config(
                    text=f"模型已加载\n文件: {model_name}\n棋盘大小: {self.game.size}x{self.game.size}",
                    fg='green'
                )

                # 启用按钮
                self.save_model_btn.config(state=NORMAL)
                self.eval_model_btn.config(state=NORMAL)

                self.add_cnn_log("[完成] 模型加载成功")
                messagebox.showinfo("成功", f"CNN模型已从 {model_name} 加载")

            except Exception as e:
                self.add_cnn_log(f"[错误] 加载失败: {str(e)}")
                messagebox.showerror("错误", f"加载失败: {str(e)}")
                import traceback
                traceback.print_exc()

    def save_cnn_model(self):
        """保存CNN模型"""
        if self.cnn_agent is None:
            messagebox.showwarning("警告", "没有可保存的模型")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".pth",
            filetypes=[
                ("PyTorch模型", "*.pth"),
                ("H5模型", "*.h5"),
                ("所有文件", "*.*")
            ],
            title="保存训练后的模型"
        )

        if filename:
            try:
                self.add_cnn_log(f"正在保存模型: {filename}")

                # 假设CNNAgent有save_model方法
                if hasattr(self.cnn_agent, 'save_model'):
                    self.cnn_agent.save_model(filename)
                else:
                    # 如果没有，尝试保存模型参数
                    import torch
                    torch.save(self.cnn_agent.model.state_dict(), filename)

                self.add_cnn_log("[完成] 模型保存成功")
                messagebox.showinfo("成功", f"模型已保存到 {filename}")

            except Exception as e:
                self.add_cnn_log(f"[错误] 保存失败: {str(e)}")
                messagebox.showerror("错误", f"保存失败: {str(e)}")

    def train_cnn_model(self):
        """使用TCP分布式系统训练CNN模型"""
        try:
            num_games = int(self.cnn_games_entry.get())
        except ValueError:
            messagebox.showerror("错误", "请输入有效的训练局数")
            return

        if num_games <= 0:
            messagebox.showerror("错误", "训练局数必须大于0")
            return

        # 确认开始训练
        response = messagebox.askyesno(
            "开始TCP训练", 
            f"将启动TCP分布式训练系统:\n\n"
            f"• 训练局数: {num_games}\n"
            f"• 客户端数: 2\n"
            f"• 思考时间: 1.5秒/步\n\n"
            f"训练将在后台进行，可以在GUI中观战\n"
            f"数据保存至: training_data/\n"
            f"模型保存至: models/\n\n"
            f"确定开始训练？"
        )
        
        if not response:
            return

        # 禁用训练按钮
        self.train_model_btn.config(state=DISABLED)
        self.cnn_progress_var.set(0)

        self.add_cnn_log("=" * 60)
        self.add_cnn_log("[启动] 启动TCP分布式训练系统")
        self.add_cnn_log("=" * 60)
        self.add_cnn_log(f"训练局数: {num_games}")
        self.add_cnn_log(f"客户端数: 2 (共 {num_games * 2} 局)")
        self.add_cnn_log(f"模式: 后台训练 + GUI观战")
        self.add_cnn_log("=" * 60)

        # TCP训练进程管理
        self.tcp_processes = []
        self.tcp_bridge = None
        self.tcp_running = True
        self.current_observed_game_id = None
        self.current_observed_client = None
        self.current_observed_move_num = 0
        self.enable_winner_popup_test = True
        
        def start_tcp_training():
            try:
                # 0. 清理可能占用端口的旧进程
                self.add_cnn_log("[检查] 检查端口占用...")
                try:
                    import subprocess as sp
                    # 查找占用9999端口的进程
                    result = sp.run(
                        ['netstat', '-ano'],
                        capture_output=True,
                        text=True,
                        encoding='utf-8',
                        errors='replace'
                    )
                    
                    pids_to_kill = set()
                    for line in result.stdout.split('\n'):
                        if ':9999' in line and 'LISTENING' in line:
                            parts = line.split()
                            if parts:
                                try:
                                    pid = int(parts[-1])
                                    pids_to_kill.add(pid)
                                except:
                                    pass
                    
                    if pids_to_kill:
                        self.add_cnn_log(f"[警告] 发现{len(pids_to_kill)}个占用9999端口的进程，正在清理...")
                        for pid in pids_to_kill:
                            try:
                                sp.run(['taskkill', '/F', '/PID', str(pid)], 
                                      capture_output=True, timeout=2)
                            except:
                                pass
                        time.sleep(1)
                        self.add_cnn_log("[完成] 端口清理完成")
                    else:
                        self.add_cnn_log("[完成] 端口未被占用")
                except Exception as e:
                    self.add_cnn_log(f"[警告] 端口检查失败: {e}")
                
                # 1. 启动训练服务器
                self.add_cnn_log("📡 启动训练服务器...")
                server_cmd = [
                    sys.executable,
                    str(Path(__file__).parent / 'tcp_training_manager.py'),
                    '--mode', 'online',
                    '--port', '9999'
                ]
                
                server_process = subprocess.Popen(
                    server_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,  # 合并stderr到stdout
                    text=True,
                    encoding='utf-8',  # 指定UTF-8编码
                    errors='replace',  # 遇到无法解码的字符时替换
                    bufsize=1
                )
                self.tcp_processes.append(server_process)
                
                # 在后台读取服务器输出并检测就绪信号
                server_ready = threading.Event()
                
                def read_server_output():
                    for line in server_process.stdout:
                        output = line.rstrip()
                        print(f"[服务器] {output}")
                        # 检测就绪信号 - 等待Accept线程真正启动
                        if "Accept线程已启动" in output or "服务器就绪" in output:
                            server_ready.set()
                
                threading.Thread(target=read_server_output, daemon=True).start()
                
                # 等待服务器就绪信号或超时
                self.add_cnn_log("⏳ 等待服务器就绪...")
                if server_ready.wait(timeout=15):  # 增加到15秒
                    self.add_cnn_log("[完成] 服务器就绪信号已接收")
                    time.sleep(1)  # 额外等待1秒确保端口完全就绪
                else:
                    self.add_cnn_log("[警告] 未收到就绪信号，继续尝试连接...")
                
                # 等待服务器启动并监听端口
                self.add_cnn_log("⏳ 验证服务器端口...")
                max_wait = 10  # 最多等待10秒
                connected = False
                last_error = None
                
                for i in range(max_wait):
                    time.sleep(1)
                    
                    # 检查进程是否还在运行
                    if server_process.poll() is not None:
                        # 进程已退出
                        self.add_cnn_log(f"[错误] 服务器进程异常退出 (退出码: {server_process.returncode})")
                        raise Exception(f"服务器进程启动失败，退出码: {server_process.returncode}")
                    
                    # 尝试连接测试服务器是否就绪
                    for host in ['127.0.0.1', 'localhost']:
                        try:
                            test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                            test_sock.settimeout(3)  # 增加到3秒超时
                            test_sock.connect((host, 9999))
                            test_sock.close()
                            connected = True
                            self.add_cnn_log(f"[完成] 服务器端口验证成功")
                            break
                        except Exception as e:
                            last_error = f"{host}:9999 - {type(e).__name__}"
                    
                    if connected:
                        break
                
                if not connected:
                    self.add_cnn_log(f"[错误] 无法连接到服务器端口")
                    self.add_cnn_log(f"  最后错误: {last_error}")
                    self.add_cnn_log("  提示: 可能是防火墙或端口被占用")
                    raise Exception("服务器端口连接失败")
                
                # 2. 启动AI客户端
                self.add_cnn_log("🤖 启动2个AI客户端...")
                for i in range(2):
                    client_cmd = [
                        sys.executable,
                        str(Path(__file__).parent / 'tcp_selfplay_client.py'),
                        '--host', 'localhost',
                        '--port', '9999',
                        '--time', '1.5',
                        '--games', str(num_games)
                    ]
                    
                    client_process = subprocess.Popen(
                        client_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        encoding='utf-8',  # 指定UTF-8编码
                        errors='replace',  # 遇到无法解码的字符时替换
                        bufsize=1
                    )
                    
                    # 读取客户端输出
                    def read_client_output(process, client_id):
                        for line in process.stdout:
                            print(f"[客户端#{client_id}] {line.rstrip()}")
                    
                    threading.Thread(target=read_client_output, args=(client_process, i+1), daemon=True).start()
                    
                    self.tcp_processes.append(client_process)
                    self.add_cnn_log(f"  [完成] 客户端 #{i+1} 已启动")
                    time.sleep(0.5)
                
                # 3. 连接到服务器观战
                self.add_cnn_log("🎨 连接到训练服务器观战...")
                time.sleep(1)  # 让客户端先连接
                
                try:
                    self.tcp_bridge = self._create_tcp_bridge()
                    self.add_cnn_log("[完成] 观战连接已建立")
                    
                    self.add_cnn_log("=" * 60)
                    self.add_cnn_log("[完成] TCP训练系统已启动！")
                    self.add_cnn_log("=" * 60)
                    self.add_cnn_log("当前棋盘显示实时对弈")
                    self.add_cnn_log(f"[状态] {num_games * 2}局训练正在后台进行")
                    self.add_cnn_log(" 数据自动保存到 training_data/")
                    self.add_cnn_log("=" * 60)
                    
                    # 更新进度标签
                    self.cnn_progress_label.config(
                        text=f"TCP训练运行中...\n目标: {num_games * 2}局\n数据保存: training_data/"
                    )
                    
                    # 监控训练进度
                    self._monitor_tcp_training(num_games * 2)
                    
                except Exception as e:
                    self.add_cnn_log(f"[警告] 观战连接失败: {e}")
                    self.add_cnn_log("训练仍在后台继续...")
                
            except Exception as e:
                self.add_cnn_log(f"[错误] 启动失败: {e}")
                messagebox.showerror("错误", f"TCP训练启动失败:\n{e}")
                self.train_model_btn.config(state=NORMAL)
                self._stop_tcp_training()
        
        threading.Thread(target=start_tcp_training, daemon=True).start()
    
    def _create_tcp_bridge(self):
        """创建TCP观战桥接"""
        max_retries = 5
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                print(f"正在连接到训练服务器...（尝试 {attempt+1}/{max_retries}）")
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)  # 5秒超时
                sock.connect(('localhost', 9999))
                sock.settimeout(None)  # 连接后取消超时
                print("[连接] TCP连接已建立")
                
                # 注册为观察者
                msg = pickle.dumps({'type': 'register_observer'})
                sock.sendall(len(msg).to_bytes(4, 'big') + msg)
                print("[连接] 已注册为观察者")
                
                # 启动接收线程
                def receive_updates():
                    buffer = b''
                    print("观战接收线程已启动")
                    while self.tcp_running:
                        try:
                            chunk = sock.recv(4096)
                            if not chunk:
                                print("TCP连接已关闭")
                                break
                            buffer += chunk
                            
                            while len(buffer) >= 4:
                                msg_len = int.from_bytes(buffer[:4], 'big')
                                if len(buffer) < 4 + msg_len:
                                    break
                                
                                msg_data = buffer[4:4+msg_len]
                                buffer = buffer[4+msg_len:]
                                message = pickle.loads(msg_data)
                                print(f"收到消息类型: {message.get('type')}")
                                self._handle_tcp_message(message)
                        except Exception as e:
                            if self.tcp_running:
                                print(f"接收错误: {e}")
                                import traceback
                                traceback.print_exc()
                            break
                    print("观战接收线程已退出")
                
                threading.Thread(target=receive_updates, daemon=True).start()
                return sock
                
            except socket.timeout:
                print(f"[错误] 连接超时（尝试 {attempt+1}/{max_retries}）")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
            except Exception as e:
                print(f"[错误] 连接失败: {e}（尝试 {attempt+1}/{max_retries}）")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
        
        raise Exception(f"连接失败：已尝试 {max_retries} 次")
    
    def _handle_tcp_message(self, msg):
        """处理TCP训练消息"""
        msg_type = msg.get('type')

        if msg_type == 'observer_switched':
            client = msg.get('client')
            game_id = msg.get('game_id')
            reason = msg.get('reason', 'unknown')
            self.current_observed_client = client
            self.current_observed_game_id = game_id
            self.current_observed_move_num = 0

            switch_text = f"[切换] 已切换观战对局: {game_id}"
            print(f"观战切换: client={client}, game_id={game_id}, reason={reason}")
            self.add_cnn_log(switch_text)
            self.root.title(f"海克斯棋 - TCP训练中 | 当前对局 {game_id}")
            return
        
        if msg_type == 'game_update':
            board = msg.get('board')
            move = msg.get('move')
            move_num = msg.get('move_number', 0)
            player = msg.get('player')
            game_id = msg.get('game_id')
            client = msg.get('client')

            if game_id and game_id != self.current_observed_game_id:
                self.current_observed_game_id = game_id
                self.current_observed_client = client
                self.current_observed_move_num = 0
                self.add_cnn_log(f"[信息] 新对局开始: {game_id}")
                self.canvas.delete('last_move_marker')
            
            # 添加调试日志
            print(f"收到棋局更新: game_id={game_id}, 第{move_num}手, 玩家{player}, 落子{move}")
            self.current_observed_move_num = move_num
            
            if board is not None:
                def update_gui():
                    try:
                        # 更新棋盘
                        if isinstance(board, np.ndarray):
                            board_list = board.tolist()
                        else:
                            board_list = board
                        
                        self.game.board = board_list
                        self.board = board_list
                        self.array_to_hex(board_list)
                        
                        # 标记最后一步
                        if move:
                            x, y = move
                            if 0 <= x < len(self.hex_board) and 0 <= y < len(self.hex_board[x]):
                                hex_id = self.hex_board[x][y]
                                coords = self.canvas.coords(hex_id)
                                if coords:
                                    cx = sum(coords[::2]) / 6
                                    cy = sum(coords[1::2]) / 6
                                    self.canvas.delete('last_move_marker')
                                    self.canvas.create_oval(
                                        cx-8, cy-8, cx+8, cy+8,
                                        outline='yellow', width=3,
                                        tags='last_move_marker'
                                    )
                        
                        # 更新标题
                        player_str = '红方' if player == 1 else '蓝方'
                        title_game_id = game_id if game_id else '未知对局'
                        self.root.title(f"海克斯棋 - TCP训练中 | {title_game_id} | 第{move_num}手 - {player_str}")
                        print(f"GUI已更新: 第{move_num}手")
                    except Exception as e:
                        print(f"GUI更新失败: {e}")
                        import traceback
                        traceback.print_exc()
                
                self.root.after(0, update_gui)
        
        elif msg_type == 'game_end':
            winner = msg.get('winner')
            moves = msg.get('moves')
            game_id = msg.get('game_id')

            if self.current_observed_game_id and game_id and game_id != self.current_observed_game_id:
                print(f"忽略非当前对局结束消息: current={self.current_observed_game_id}, msg={game_id}")
                return

            winner_str = '红方' if winner == 1 else '蓝方'
            end_game_id = game_id if game_id else self.current_observed_game_id
            print(f"对局结束: game_id={end_game_id}, {winner_str}获胜 ({moves}手)")
            self.add_cnn_log(f"[结果] 对局结束 [{end_game_id}]: {winner_str}获胜 ({moves}手)")

            if self.enable_winner_popup_test:
                self.root.after(0, lambda: messagebox.showinfo(
                    "对局结束",
                    f"对局 [{end_game_id}] 已结束\n\n胜者: {winner_str}\n手数: {moves}"
                ))
    
    def _monitor_tcp_training(self, total_games):
        """监控TCP训练进度"""
        def monitor():
            stats_file = Path(__file__).parent / 'training_data' / 'training_stats.json'
            last_completed = -1
            initial_games = 0
            baseline_ready = False
            
            while self.tcp_running:
                time.sleep(5)  # 每5秒检查一次
                
                if stats_file.exists():
                    try:
                        with open(stats_file) as f:
                            stats = json.load(f)

                        current_games = stats.get('total_games', 0)
                        total_positions = stats.get('total_positions', 0)
                        batches = stats.get('batches_saved', 0)

                        if not baseline_ready:
                            initial_games = current_games
                            baseline_ready = True

                        completed_games = max(0, current_games - initial_games)
                        capped_completed = min(completed_games, total_games)
                        progress = int((capped_completed / total_games) * 100) if total_games > 0 else 0
                        
                        if capped_completed != last_completed:
                            self.cnn_progress_var.set(progress)
                            
                            self.cnn_progress_label.config(
                                text=f"训练进度: {capped_completed}/{total_games}局 ({progress}%)\n"
                                     f"训练位置: {total_positions} | 批次: {batches}"
                            )
                            
                            self.add_cnn_log(f"[进度] {capped_completed}/{total_games}局 | {total_positions}位置")
                            last_completed = capped_completed
                        
                        # 训练完成
                        if completed_games >= total_games:
                            self.cnn_progress_var.set(100)
                            self.add_cnn_log("=" * 60)
                            self.add_cnn_log("[完成] 训练完成！")
                            self.add_cnn_log(f"本次对局: {total_games}")
                            self.add_cnn_log(f"训练位置: {total_positions}")
                            self.add_cnn_log(f"数据批次: {batches}")
                            self.add_cnn_log("=" * 60)
                            
                            self._stop_tcp_training()
                            
                            self.root.after(0, lambda: messagebox.showinfo(
                                "训练完成",
                                f"TCP训练已完成！\n\n"
                                f"本次对局数: {total_games}\n"
                                f"训练位置: {total_positions}\n"
                                f"数据保存: training_data/\n"
                                f"模型保存: models/"
                            ))
                            
                            self.train_model_btn.config(state=NORMAL)
                            break
                            
                    except Exception as e:
                        print(f"读取统计失败: {e}")
        
        threading.Thread(target=monitor, daemon=True).start()
    
    def _stop_tcp_training(self):
        """停止TCP训练"""
        self.tcp_running = False
        
        if self.tcp_bridge:
            try:
                self.tcp_bridge.close()
            except:
                pass
        
        for p in self.tcp_processes:
            try:
                p.terminate()
                p.wait(timeout=3)
            except:
                try:
                    p.kill()
                except:
                    pass
        
        self.tcp_processes = []
        self.add_cnn_log("[停止] TCP训练系统已停止")

    def evaluate_cnn_model(self):
        """评估CNN模型"""
        if self.cnn_agent is None:
            messagebox.showwarning("警告", "请先加载或训练模型")
            return

        self.add_cnn_log("=" * 40)
        self.add_cnn_log("开始评估模型")

        try:
            # 简单评估：让CNN对当前局面给出预测
            if hasattr(self.cnn_agent, 'predict_move'):
                moves = self.cnn_agent.predict_move(self.game)

                self.add_cnn_log("预测的前5个走法:")
                for i, (move, prob) in enumerate(moves[:5]):
                    coord = chr(ord('A') + move[1]) + str(move[0] + 1)
                    self.add_cnn_log(f"  {i + 1}. {coord} - 概率: {prob:.2%}")

            # 或者让CNN和MCTS对战
            elif messagebox.askyesno("评估模型", "是否让CNN与MCTS对战进行评估？"):
                self.add_cnn_log("准备与MCTS对战...")
                # 这里可以添加对战逻辑
                self.add_cnn_log("评估功能开发中...")

            else:
                self.add_cnn_log("评估已取消")

        except Exception as e:
            self.add_cnn_log(f"[错误] 评估失败: {str(e)}")
            messagebox.showerror("错误", f"评估失败: {str(e)}")

        self.add_cnn_log("=" * 40)
    def save_opening_book(self):
        """保存开局库"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="保存开局库"
        )
        if filename:
            try:
                if hasattr(self.agent, 'opening_book'):
                    self.agent.opening_book.save_opening_book(filename)
                    messagebox.showinfo("成功", f"开局库已保存到 {filename}")
                else:
                    messagebox.showwarning("警告", "当前代理没有开局库")
            except Exception as e:
                messagebox.showerror("错误", f"保存失败: {str(e)}")

    def load_opening_book(self):
        """加载开局库"""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="加载开局库"
        )
        if filename:
            try:
                if hasattr(self.agent, 'opening_book'):
                    self.agent.opening_book.load_opening_book(filename)
                    self.update_opening_display()
                    messagebox.showinfo("成功", f"开局库已从 {filename} 加载")
                else:
                    messagebox.showwarning("警告", "当前代理没有开局库")
            except Exception as e:
                messagebox.showerror("错误", f"加载失败: {str(e)}")

    def update_opening_display(self):
        """更新开局库显示"""
        self.opening_text.delete(1.0, END)

        if not hasattr(self.agent, 'opening_book'):
            self.opening_text.insert(1.0, "当前代理没有开局库")
            return

        info_text = f"海克斯棋开局库 ({self.game.size}x{self.game.size})\n"
        info_text += "=" * 30 + "\n\n"

        opening_book = self.agent.opening_book

        for move_num in range(opening_book.max_opening_moves):
            info_text += f"红棋第 {move_num + 1} 步推荐位置:\n"

            if move_num in opening_book.opening_moves:
                move_data = opening_book.opening_moves[move_num]

                if 'best_moves' in move_data:
                    moves = move_data['best_moves']
                    weights = move_data['weights']

                    for i, (move, weight) in enumerate(zip(moves, weights)):
                        coord = chr(ord('A') + move[1]) + str(move[0] + 1)
                        info_text += f"  {i + 1}. {coord} (权重: {weight:.1f})\n"

                elif 'strategies' in move_data:
                    strategies = move_data['strategies']

                    for strategy_name, strategy in strategies.items():
                        if strategy_name == 'default':
                            continue

                        info_text += f"  策略: {strategy_name}\n"
                        if 'moves' in strategy:
                            moves = strategy['moves']
                            weights = strategy['weights']

                            for i, (move, weight) in enumerate(zip(moves[:3], weights[:3])):
                                coord = chr(ord('A') + move[1]) + str(move[0] + 1)
                                info_text += f"    - {coord} (权重: {weight:.1f})\n"

            info_text += "\n"

        info_text += "开局库说明:\n"
        info_text += "- 红棋第1步: 优先选择中心位置\n"
        info_text += "- 红棋第2步: 根据蓝棋位置调整策略\n"
        info_text += "- 红棋第3步: 形成基本结构\n"
        info_text += "- 红棋第4步开始: 使用MCTS搜索\n"

        self.opening_text.insert(1.0, info_text)

    def setup_game_controls(self):
        """设置游戏控制面板"""
        self.panel_game.configure(bg=bg)

        # 计时器
        self.timer_frame = Frame(self.panel_game, bg=bg)
        self.timer_frame.pack(side=TOP, fill=X, pady=5)
        Label(
            self.timer_frame, text='倒计时:', font=('KaiTi', 14, 'bold'),
            foreground='white', bg=bg
        ).pack(side=LEFT, padx=5)
        self.timer_label = Label(
            self.timer_frame, text='15:00', font=('KaiTi', 18, 'bold'),
            foreground='red', bg=bg
        )
        self.timer_label.pack(side=LEFT, padx=5)

        # 思考时间按钮
        Label(
            self.panel_game, text='思考时间(秒)', font=('KaiTi', 14, 'bold'),
            foreground='white', bg=bg, pady=10
        ).pack(side=TOP, fill=X)
        self.time_buttons = Frame(self.panel_game, bg=bg)
        time_values = [1, 5, 10, 15, 20]
        self.current_time = 1
        for time in time_values:
            btn = Button(
                self.time_buttons, text=f"{time}秒", font=('KaiTi', 12),
                bg=self.colors['button'], fg='white', command=lambda t=time: self.set_time_button(t)
            )
            btn.pack(side=LEFT, padx=5, pady=5)
        self.time_buttons.pack(side=TOP, fill=X)

        # 算法选择按钮
        Label(
            self.panel_game, text='选择算法', font=('KaiTi', 14, 'bold'),
            foreground='white', bg=bg, pady=10
        ).pack(side=TOP, fill=X)
        self.agent_buttons = Frame(self.panel_game, bg=bg)
        agent_names = ["UCT", "RAVE"]
        if CNN_AVAILABLE:
            agent_names.append("CNN")
        self.current_agent = agent_names[0]
        for agent in agent_names:
            btn = Button(
                self.agent_buttons, text=agent, font=('KaiTi', 12),
                bg=self.colors['button'], fg='white', command=lambda a=agent: self.set_agent_button(a)
            )
            btn.pack(side=LEFT, padx=5, pady=5)
        self.agent_buttons.pack(side=TOP, fill=X)

        # 当前玩家
        Label(
            self.panel_game, text='当前玩家', font=('KaiTi', 14, 'bold'),
            foreground='white', bg=bg, pady=10
        ).pack(side=TOP, fill=X)
        self.game_turn.configure(
            from_='1', to=2, tickinterval=1, bg=bg, fg='white',
            orient=HORIZONTAL, variable=self.game_turn_value
        )
        self.game_turn.pack(side=TOP)

        # 走法信息
        self.move_label = Label(
            self.panel_game, font=('KaiTi', 15, 'bold'), height=5, fg='white',
            justify=LEFT, bg=bg
        )
        self.move_label.pack(side=TOP, fill=X)

        # 游戏控制按钮
        self.control_buttons = Frame(self.panel_game, bg=bg)
        self.control_buttons.pack(side=TOP, fill=X, pady=5)

        self.reset_board.configure(
            text='重置棋盘', pady=10, cursor='hand2', width=12,
            font=('KaiTi', 12, 'bold'), bg=self.colors['button'], fg='white'
        )
        self.reset_board.pack(side=LEFT, padx=2)
        self.reset_board.bind('<ButtonRelease>', self.reset)

        self.generate.configure(
            text='生成', pady=10, cursor='hand2', width=12,
            font=('KaiTi', 12, 'bold'), bg=self.colors['button'], fg='white'
        )
        self.generate.pack(side=LEFT, padx=2)
        self.generate.bind('<ButtonRelease>', self.generate_move)

        # 悔棋按钮
        self.undo_button = Button(self.control_buttons)
        self.undo_button.configure(
            text='悔棋', pady=10, cursor='hand2', width=12,
            font=('KaiTi', 12, 'bold'), bg=self.colors['button'], fg='white',
            state=DISABLED
        )
        self.undo_button.pack(side=LEFT, padx=2)
        self.undo_button.bind('<ButtonRelease>', self.undo_move)

        # 计时器初始化
        self.total_seconds = 15 * 60
        self.timer_running = False
        self.update_timer()

    def setup_record_controls(self):
        """设置棋谱控制面板"""
        self.panel_record.configure(bg=bg)

        Label(
            self.panel_record, text='比赛信息设置', font=('KaiTi', 14, 'bold'),
            foreground='white', bg=bg, pady=10
        ).pack(side=TOP, fill=X)

        # 比赛信息输入
        info_frame = Frame(self.panel_record, bg=bg)
        info_frame.pack(side=TOP, fill=X, pady=5)

        Label(info_frame, text='比赛名称:', fg='white', bg=bg).grid(row=0, column=0, sticky=W, padx=5)
        self.event_entry = Entry(info_frame, width=20)
        self.event_entry.grid(row=0, column=1, padx=5, pady=2)

        Label(info_frame, text='红方:', fg='white', bg=bg).grid(row=1, column=0, sticky=W, padx=5)
        self.red_player_entry = Entry(info_frame, width=20)
        self.red_player_entry.grid(row=1, column=1, padx=5, pady=2)

        Label(info_frame, text='蓝方:', fg='white', bg=bg).grid(row=2, column=0, sticky=W, padx=5)
        self.blue_player_entry = Entry(info_frame, width=20)
        self.blue_player_entry.grid(row=2, column=1, padx=5, pady=2)

        # 棋谱操作按钮
        Label(
            self.panel_record, text='棋谱管理', font=('KaiTi', 14, 'bold'),
            foreground='white', bg=bg, pady=10
        ).pack(side=TOP, fill=X)

        record_buttons = Frame(self.panel_record, bg=bg)
        record_buttons.pack(side=TOP, fill=X, pady=5)

        save_sgf_btn = Button(
            record_buttons, text='保存SGF', font=('KaiTi', 10),
            bg=self.colors['button'], fg='white', command=self.save_sgf
        )
        save_sgf_btn.pack(side=LEFT, padx=2, pady=5)

        load_sgf_btn = Button(
            record_buttons, text='加载SGF', font=('KaiTi', 10),
            bg=self.colors['button'], fg='white', command=self.load_sgf
        )
        load_sgf_btn.pack(side=LEFT, padx=2, pady=5)

        export_txt_btn = Button(
            record_buttons, text='导出TXT', font=('KaiTi', 10),
            bg=self.colors['button'], fg='white', command=self.export_txt
        )
        export_txt_btn.pack(side=LEFT, padx=2, pady=5)

        # 棋谱回放控制
        Label(
            self.panel_record, text='棋谱回放', font=('KaiTi', 14, 'bold'),
            foreground='white', bg=bg, pady=10
        ).pack(side=TOP, fill=X)

        replay_buttons = Frame(self.panel_record, bg=bg)
        replay_buttons.pack(side=TOP, fill=X, pady=5)

        first_btn = Button(
            replay_buttons, text='|<<', font=('KaiTi', 10),
            bg=self.colors['button'], fg='white', command=self.replay_first
        )
        first_btn.pack(side=LEFT, padx=2, pady=5)

        prev_btn = Button(
            replay_buttons, text='<', font=('KaiTi', 10),
            bg=self.colors['button'], fg='white', command=self.replay_prev
        )
        prev_btn.pack(side=LEFT, padx=2, pady=5)

        next_btn = Button(
            replay_buttons, text='>', font=('KaiTi', 10),
            bg=self.colors['button'], fg='white', command=self.replay_next
        )
        next_btn.pack(side=LEFT, padx=2, pady=5)

        last_btn = Button(
            replay_buttons, text='>>|', font=('KaiTi', 10),
            bg=self.colors['button'], fg='white', command=self.replay_last
        )
        last_btn.pack(side=LEFT, padx=2, pady=5)

        exit_replay_btn = Button(
            replay_buttons, text='退出回放', font=('KaiTi', 10),
            bg=self.colors['button'], fg='white', command=self.exit_replay
        )
        exit_replay_btn.pack(side=LEFT, padx=2, pady=5)

        # 棋谱显示
        Label(
            self.panel_record, text='对局记录', font=('KaiTi', 14, 'bold'),
            foreground='white', bg=bg, pady=10
        ).pack(side=TOP, fill=X)

        text_frame = Frame(self.panel_record, bg=bg)
        text_frame.pack(side=TOP, fill=BOTH, expand=True, padx=5, pady=5)

        self.record_text = Text(text_frame, height=15, width=35, font=('Courier', 9),
                               bg='white', fg='black', wrap=WORD)
        record_scrollbar = Scrollbar(text_frame, bg=self.colors['button'])
        record_scrollbar.pack(side=RIGHT, fill=Y)
        self.record_text.pack(side=LEFT, fill=BOTH, expand=True)
        self.record_text.config(yscrollcommand=record_scrollbar.set)
        record_scrollbar.config(command=self.record_text.yview)

    def set_time_button(self, time_value):
        """设置思考时间"""
        self.current_time = time_value
        self.time = time_value
        self.game_time_value.set(time_value)

        # 更新所有时间按钮的颜色
        for widget in self.time_buttons.winfo_children():
            if isinstance(widget, Button):
                if widget['text'] == f"{time_value}秒":
                    widget.configure(bg=self.colors['button_active'])
                else:
                    widget.configure(bg=self.colors['button'])

    def set_agent_button(self, agent_name):
        """设置AI代理"""
        try:
            if agent_name == "CNN" and not CNN_AVAILABLE:
                messagebox.showerror("错误", "CNN模块不可用")
                return

            self.current_agent = agent_name
            self.agent_name = agent_name

            # 创建新的代理
            if agent_name == "CNN":
                if hasattr(self, 'cnn_agent'):
                    new_agent = self.cnn_agent
                else:
                    new_agent = CNNAgent(board_size=self.game.size)
            else:
                new_agent = self.AGENTS[agent_name]()

            new_agent.set_gamestate(self.game)

            # 添加开局库
            if not hasattr(new_agent, 'opening_book'):
                new_agent.opening_book = OpeningBook(self.game.size)
                new_agent.red_move_count = 0

            self.agent = new_agent

            # 更新所有代理按钮的颜色
            for widget in self.agent_buttons.winfo_children():
                if isinstance(widget, Button):
                    if widget['text'] == agent_name:
                        widget.configure(bg=self.colors['button_active'])
                    else:
                        widget.configure(bg=self.colors['button'])

        except Exception as e:
            messagebox.showerror("错误", f"切换代理失败: {str(e)}")
            import traceback
            traceback.print_exc()

    def save_sgf(self):
        """保存SGF格式棋谱"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".sgf",
            filetypes=[("SGF files", "*.sgf"), ("All files", "*.*")],
            title="保存SGF棋谱"
        )
        if filename:
            try:
                event = self.event_entry.get() or "友谊赛"
                red_player = self.red_player_entry.get() or "红方"
                blue_player = self.blue_player_entry.get() or "蓝方"

                self.game.game_record.set_game_info(event, red_player, blue_player)
                self.game.game_record.save_sgf(filename)

                messagebox.showinfo("成功", f"棋谱已保存到 {filename}")
            except Exception as e:
                messagebox.showerror("错误", f"保存失败: {str(e)}")

    def load_sgf(self):
        """加载SGF格式棋谱"""
        filename = filedialog.askopenfilename(
            filetypes=[("SGF files", "*.sgf"), ("All files", "*.*")],
            title="加载SGF棋谱"
        )
        if filename:
            try:
                self.game.game_record.load_sgf(filename)

                # 更新界面信息
                info = self.game.game_record.game_info
                if 'EV' in info:
                    self.event_entry.delete(0, END)
                    self.event_entry.insert(0, info['EV'])
                if 'PB' in info:
                    self.red_player_entry.delete(0, END)
                    self.red_player_entry.insert(0, info['PB'])
                if 'PW' in info:
                    self.blue_player_entry.delete(0, END)
                    self.blue_player_entry.insert(0, info['PW'])

                # 进入回放模式
                self.replay_mode = True
                self.replay_index = 0
                self.replay_first()

                self.update_record_display()

                messagebox.showinfo("成功", f"棋谱已从 {filename} 加载")
            except Exception as e:
                messagebox.showerror("错误", f"加载失败: {str(e)}")

    def export_txt(self):
        """导出TXT格式棋谱"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="导出TXT棋谱"
        )
        if filename:
            try:
                event = self.event_entry.get() or "友谊赛"
                red_player = self.red_player_entry.get() or "红方"
                blue_player = self.blue_player_entry.get() or "蓝方"

                self.game.game_record.set_game_info(event, red_player, blue_player)
                self.game.game_record.export_txt(filename)

                messagebox.showinfo("成功", f"棋谱已导出到 {filename}")
            except Exception as e:
                messagebox.showerror("错误", f"导出失败: {str(e)}")

    def replay_first(self):
        """回放到第一步"""
        if not self.replay_mode:
            return

        self.replay_index = 0
        self.game.board = np.zeros((self.game.size, self.game.size))
        self.game.toplay = 1
        self.array_to_hex(self.game.board)
        self.update_record_display()

    def replay_prev(self):
        """回放上一步"""
        if not self.replay_mode or self.replay_index <= 0:
            return

        self.replay_index -= 1
        self.replay_to_step(self.replay_index)

    def replay_next(self):
        """回放下一步"""
        if not self.replay_mode:
            return

        if self.replay_index < len(self.game.game_record.moves):
            self.replay_index += 1
            self.replay_to_step(self.replay_index)

    def replay_last(self):
        """回放到最后一步"""
        if not self.replay_mode:
            return

        self.replay_index = len(self.game.game_record.moves)
        self.replay_to_step(self.replay_index)

    def replay_to_step(self, step):
        """回放到指定步数"""
        self.game.board = np.zeros((self.game.size, self.game.size))
        self.game.toplay = 1

        for i in range(step):
            if i < len(self.game.game_record.moves):
                move = self.game.game_record.moves[i]
                self.game.board[move[0]][move[1]] = self.game.toplay
                self.game.toplay = 3 - self.game.toplay

        self.array_to_hex(self.game.board)
        self.update_record_display()

    def exit_replay(self):
        """退出回放模式"""
        self.replay_mode = False
        self.replay_index = 0
        messagebox.showinfo("提示", "已退出回放模式")

    def update_record_display(self):
        """更新棋谱显示"""
        self.record_text.delete(1.0, END)

        info = self.game.game_record.game_info
        info_text = f"比赛: {info.get('EV', '未知')}\n"
        info_text += f"红方: {info.get('PB', '未知')}\n"
        info_text += f"蓝方: {info.get('PW', '未知')}\n"
        info_text += f"日期: {info.get('DT', '未知')}\n"
        info_text += f"棋盘大小: {info.get('SZ', self.game.size)}\n"
        info_text += "=" * 30 + "\n\n"

        moves = self.game.game_record.moves
        for i, move in enumerate(moves):
            color = "红" if i % 2 == 0 else "蓝"
            coord = chr(ord('A') + move[1]) + str(move[0] + 1)

            if self.replay_mode and i == self.replay_index - 1:
                info_text += f">>> {i + 1}. {color}方 {coord} <<<\n"
            else:
                info_text += f"{i + 1}. {color}方 {coord}\n"

        self.record_text.insert(1.0, info_text)

    def update_timer(self):
        """更新计时器"""
        if self.timer_running and self.total_seconds > 0:
            self.total_seconds -= 1

        minutes = self.total_seconds // 60
        seconds = self.total_seconds % 60
        self.timer_label.config(text=f"{minutes:02d}:{seconds:02d}")

        if self.total_seconds == 0:
            self.timer_running = False
            messagebox.showinfo("时间到", "时间已用完！")

        self.root.after(1000, self.update_timer)

    def start_timer(self):
        """启动计时器"""
        self.timer_running = True

    def stop_timer(self):
        """停止计时器"""
        self.timer_running = False

    def reset_timer(self):
        """重置计时器"""
        self.total_seconds = 15 * 60
        self.timer_running = False

    def mouse_click(self, event):
        """处理鼠标点击事件"""
        if self.replay_mode:
            messagebox.showwarning("警告", "回放模式下不能下棋")
            return

        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)

        # 找到点击的六边形
        for i in range(len(self.hex_board)):
            for j in range(len(self.hex_board[i])):
                if self.canvas.type(self.hex_board[i][j]) == 'polygon':
                    coords = self.canvas.coords(self.hex_board[i][j])
                    if self.point_in_polygon(x, y, coords):
                        if self.game.board[i][j] == 0:
                            self.make_move(i, j)
                        return

    def point_in_polygon(self, x, y, coords):
        """判断点是否在多边形内"""
        n = len(coords) // 2
        inside = False

        p1x, p1y = coords[0], coords[1]
        for i in range(1, n + 1):
            p2x, p2y = coords[(i * 2) % len(coords)], coords[(i * 2 + 1) % len(coords)]
            if min(p1y, p2y) < y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def make_move(self, i, j):
        """执行落子"""
        if self.game.board[i][j] != 0:
            return

        # 保存当前状态用于悔棋
        self.save_state()

        # 记录当前玩家
        current_player = self.game.toplay

        # 执行落子
        self.game.make_move((i, j))

        # 记录到棋谱
        self.game.game_record.add_move((i, j), current_player)

        # 更新界面
        self.array_to_hex(self.game.board)
        self.last_move = (i, j)

        # 更新走法信息
        coord = chr(ord('A') + j) + str(i + 1)
        color = "红方" if current_player == 1 else "蓝方"
        self.move_label.config(text=f"上一步: {color}\n位置: {coord}")

        # 更新棋谱显示
        self.update_record_display()

        # 启用悔棋按钮
        self.undo_button.configure(state=NORMAL)

        # 检查胜负
        if self.game.winner() != 0:
            winner_color = "红方" if self.game.winner() == 1 else "蓝方"
            messagebox.showinfo("游戏结束", f"{winner_color}获胜！")
            self.stop_timer()

    def generate_move(self, event):
        """生成AI走法"""
        if self.replay_mode:
            messagebox.showwarning("警告", "回放模式下不能使用AI")
            return

        if self.game.winner() != 0:
            messagebox.showinfo("提示", "游戏已结束")
            return

        #  强制刷新代理状态
        if self.agent_name == "CNN":
            print("刷新CNN代理状态...")
            self.agent.set_gamestate(self.game)

            #  打印当前棋盘状态用于调试
            print(f"当前棋盘空位数: {np.sum(self.game.board == 0)}")
            print(f"红棋数: {np.sum(self.game.board == 1)}")
            print(f"蓝棋数: {np.sum(self.game.board == 2)}")

        # 保存当前状态
        self.save_state()

        # 启动计时器
        if not self.timer_running:
            self.start_timer()

        # 获取AI走法
        try:
            move = self.agent.best_move(self.time)

            if move is None:
                messagebox.showerror("错误", "AI无法生成有效走法")
                return

            # 执行前最后一次检查
            if self.game.board[move[0], move[1]] != 0:
                print(f"[错误] 致命错误: AI返回的位置 {move} 已被占用")
                print(f"   当前位置值: {self.game.board[move[0], move[1]]}")
                messagebox.showerror("错误", f"AI返回了已占用的位置 {move}")
                return

            # 记录当前玩家
            current_player = self.game.toplay

            # 执行走法
            self.game.make_move(move)
            self.game.game_record.add_move(move, current_player)

            #  同步代理状态
            if self.agent_name == "CNN":
                pass
            else:
                self.agent.move(move)

            # 更新界面
            self.array_to_hex(self.game.board)
            self.last_move = move

            # 更新走法信息
            coord = chr(ord('A') + move[1]) + str(move[0] + 1)
            color = "红方" if current_player == 1 else "蓝方"
            self.move_label.config(text=f"AI走法: {color}\n位置: {coord}")

            # 更新棋谱显示
            self.update_record_display()

            # 启用悔棋按钮
            self.undo_button.configure(state=NORMAL)

            # 检查胜负
            if self.game.winner() != 0:
                winner_color = "红方" if self.game.winner() == 1 else "蓝方"
                messagebox.showinfo("游戏结束", f"{winner_color}获胜！")
                self.stop_timer()

        except Exception as e:
            messagebox.showerror("错误", f"AI生成走法失败: {str(e)}")
            import traceback
            traceback.print_exc()

    def save_state(self):
        """保存当前状态用于悔棋"""
        state = {
            'board': deepcopy(self.game.board),
            'toplay': self.game.toplay,
            'moves': deepcopy(self.game.game_record.moves)
        }
        self.history_states.append(state)

        # 限制历史记录数量
        if len(self.history_states) > self.max_history:
            self.history_states.pop(0)

    def undo_move(self, event):
        """悔棋"""
        if len(self.history_states) == 0:
            messagebox.showwarning("警告", "没有可悔的棋")
            return

        # 恢复上一个状态
        state = self.history_states.pop()
        self.game.board = state['board']
        self.game.toplay = state['toplay']
        self.game.game_record.moves = state['moves']

        # 更新界面
        self.array_to_hex(self.game.board)
        self.update_record_display()

        # 如果没有历史了，禁用悔棋按钮
        if len(self.history_states) == 0:
            self.undo_button.configure(state=DISABLED)

    def reset(self, event):
        """重置棋盘"""
        self.game = gamestate(self.game_size_value.get())
        self.game.game_record = GameRecord(self.game_size_value.get())

        # 重新初始化AI
        try:
            if self.agent_name == "CNN" and CNN_AVAILABLE:
                if hasattr(self, 'cnn_agent'):
                    new_agent = self.cnn_agent
                else:
                    new_agent = CNNAgent(board_size=self.game.size)
            else:
                new_agent = self.AGENTS[self.agent_name]()

            new_agent.set_gamestate(self.game)

            if not hasattr(new_agent, 'opening_book'):
                new_agent.opening_book = OpeningBook(self.game.size)
                new_agent.red_move_count = 0

            self.agent = new_agent

        except Exception as e:
            print(f"重置AI失败: {str(e)}")

        self.size = self.game_size_value.get()
        self.board = self.game.board
        self.board = np.int_(self.board).tolist()
        self.array_to_hex(self.board)

        # 重置历史记录
        self.history_states.clear()
        self.undo_button.configure(state=DISABLED)

        # 重置计时器
        self.reset_timer()

        # 重置回放模式
        self.replay_mode = False
        self.replay_index = 0

        # 清空走法信息
        self.move_label.config(text="")

        # 更新棋谱显示
        self.update_record_display()


def main():
    """主函数"""
    root = Tk()
    root.title('海克斯棋 - Hex Game')

    # 默认使用UCT算法
    gui = Gui(root, agent_name='UCT')

    root.mainloop()


if __name__ == '__main__':
    main()


