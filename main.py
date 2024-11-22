# Made by cxrsedhappy with love and tears
# I have uni-code chars...
# 22.11.2024 RIP

import os
import enum
import time
import torch
import random
import keyboard

from colorama import Fore
from ascii_text import ShadedBlockyFont


WIDTH = 100
HEIGHT = 19
ASCII_NUM_BACKGROUND = ord('░')
ASCII_NUM_SNAKE = ord('▒')
ASCII_NUM_APPLE = ord('█')


class HeadDirection(enum.Enum):
    UP: tuple[int, int] = (0, 1)
    RIGHT: tuple[int, int] = (1, 0)
    DOWN: tuple[int, int] = (0, -0)
    LEFT: tuple[int, int] = (-1, 0)

    def __str__(self):
        return self.name


class GUI:
    def __init__(self):
        self.__title = ShadedBlockyFont().render('supremegod')
        self.__device_name: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.__torch_version = torch.__version__
        self.__style = {
            'border_h': '=',
            'border_v': '║',
            'corner': '╬',
            'padding': 1,
            'fc_on': Fore.GREEN,
            'fc_off': Fore.RED,
            'fc_reset': Fore.RESET
        }

    def __render_title(self):
        print(self.__title)

    def __render_params(self, ai_mode: bool, direction: tuple[int, int]):
        """
        Renders parameter information in a formatted table.

        Args:
            ai_mode (bool): Status of AI mode
            direction (tuple[int, int]): Current head direction
        """

        ai_option_color = self.__style["fc_on"] if ai_mode else self.__style["fc_off"]

        out = []
        labels = [' Torch version: ', ' AI mode: ', ' Direction: ']

        # Workaround to bypass shitty uni-code colours. I WANT COLOURS SO MUCH.
        # Contains (text, color uni-code offset)
        # p.s i hate uni-code colours
        values: list[tuple[str, int]] = [
            (
                self.__style["fc_on"] + self.__torch_version + self.__style["fc_reset"],
                len(self.__style["fc_on"] +  self.__style["fc_reset"]))
            ,
            (
                ai_option_color + str(ai_mode) + self.__style["fc_reset"],
                len(self.__style["fc_off"] +  self.__style["fc_reset"])
            ),
            (
                str(direction),
                0
            )
        ]

        total_widths = [50, 20, 25]
        value_widths = [total_widths[i] - len(labels[i]) + values[i][1] for i in range(len(labels))]

        content = []
        for i in range(len(labels)):
            content.append(f'{labels[i]}' + f'{values[i][0]:^{value_widths[i]}}')

        out.append(
            self.__style['corner'] +
            self.__style['corner'].join(self.__style["border_h"] * total_widths[i] for i in range(len(labels))) +
            self.__style['corner']
        )

        out.append(
            f"{self.__style['border_v']}" +
            self.__style['border_v'].join(content) +
            f"{self.__style['border_v']}"
        )

        out.append(
            self.__style['corner'] +
            self.__style['corner'].join(self.__style["border_h"] * total_widths[i] for i in range(len(labels))) +
            self.__style['corner']
        )

        print('\n'.join(out))

    def set_style(self, **kwargs):
        """Updates style configuration"""
        self.__style.update(kwargs)

    def render_frame(self, ai_mode: bool, board: list[str], head_direction: tuple[int, int]):
        """
        Render method of GUI.
        Applies rendering in order:
        - Title
        - Parameters
        - Board
        """
        self.clear_frame()
        self.__render_title()
        self.__render_params(ai_mode, head_direction)
        for row in board:
            print(row)

    @staticmethod
    def clear_frame():
        # cause blinking (?) epilepsy lmao?
        os.system('cls')


class Game:
    def __init__(self, gui: GUI):
        """
        Main game class. Process keyboard event, GUI frame render, in-game functionality
        """
        self.__gui = gui
        self.__ai_mode = False

        self.__board = [''.join([chr(ASCII_NUM_BACKGROUND) for _ in range(WIDTH)]) for _ in range(HEIGHT)]

        self.__has_apple_eaten = False
        self.__head_pos: tuple[int, int] = (random.randint(0, WIDTH), random.randint(0, HEIGHT))
        self.__head_direction: tuple[int, int] = (1, 0)

        self.__running = True
        self.__accept_input = True
        keyboard.hook(self.__on_btn_action)

    def __on_btn_action(self, e: keyboard.KeyboardEvent):
        if e.event_type == keyboard.KEY_DOWN:
            match e.name:
                case 's':
                    self.__head_direction = HeadDirection.DOWN
                case 'a':
                    self.__head_direction = HeadDirection.LEFT
                case 'w':
                    self.__head_direction = HeadDirection.UP
                case 'd':
                    self.__head_direction = HeadDirection.RIGHT
                case 'p':
                    self.__ai_mode = not self.__ai_mode
                case 'q':
                    self.__running = False

    def __unhook_on_btn_action(self):
        keyboard.unhook(self.__on_btn_action)

    def render(self):
        """Process render to the screen"""
        while self.__running:
            self.update()
            self.__gui.render_frame(self.__ai_mode, self.__board, self.__head_direction)
            time.sleep(0.1)

        self.__unhook_on_btn_action()


    def is_border(self, x: int, y: int):
        """Checks if x, y coords at the border"""
        ...

    def is_snake_body(self, x: int, y: int):
        """Checks if x, y coords at snake body"""
        ...

    @staticmethod
    def gather_empty_space() -> list[tuple[int, int]]:
        empty_spaces = []
        for i in range(WIDTH):
            for j in range(0, HEIGHT):
                empty_spaces.append((i, j))
        return empty_spaces


    def generate_apple(self):
        """Generate apple at random possible position"""
        empty_spaces = self.gather_empty_space()
        x, y = random.choice(empty_spaces)
        self.__board[y][x] = chr(ASCII_NUM_APPLE)

    def update(self):
        """Process pre-render functionality"""
        if not self.__ai_mode:
            ...


if __name__ == '__main__':
    gui = GUI()
    game = Game(gui)
    game.render()
