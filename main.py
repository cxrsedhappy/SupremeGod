import enum
import os
import random
import time
import torch

from colorama import Fore
from sympy.core.random import randint
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

class GUI:
    def __init__(self):
        self.title = ShadedBlockyFont().render('supremegod')
        self.__device_name: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.__torch_version = torch.__version__

    def render_title(self):
        print(self.title)

    def render_params(self, ai_mode: bool, direction: tuple[int, int]):
        """TODO: Complete parameters text"""
        print(
            '+' + '-' * 45 + '+' + '-' * 15 + "+" + '-' * 27 + "+\n" +
            f'| Torch version {Fore.GREEN + self.__torch_version + Fore.RESET:>8} using device {Fore.GREEN + self.__device_name + Fore.RESET:>4} | AI Mode {str(ai_mode):<5} | Head direction {str(direction):<10} |\n' +
            '+' + '-' * 45 + '+' + '-' * 15 + '+' + '-' * 27 + "+\n"
        )

    def render_frame(self, ai_mode: bool, board: list[str], head_direction: tuple[int, int]):
        """
        Main method of GUI.
        Applies rendering in order:
        - Title
        - Parameters
        - Board
        """
        self.clear_frame()
        self.render_title()
        self.render_params(ai_mode, head_direction)
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
        self.gui = gui
        self.ai_mode = False
        self.__running = False
        self.__board = [
            ''.join([chr(ASCII_NUM_BACKGROUND) for _ in range(WIDTH)]) for _ in range(HEIGHT)
        ]
        self.__has_apple_eaten = False
        self.__head_pos: tuple[int, int] = (randint(0, WIDTH), randint(0, HEIGHT))
        self.__head_direction: tuple[int, int] = (1, 0)
        # keyboard.hook(lambda e: self.__on_btn_action(e))

    # def __on_btn_action(self, e: keyboard):
    #     if e.event_type == keyboard.KEY_DOWN:
    #         if e.name == 's':
    #             self.__head_direction = HeadDirection.DOWN
    #         if e.name == 'a':
    #             self.__head_direction = HeadDirection.LEFT
    #         if e.name == 'w':
    #             self.__head_direction = HeadDirection.UP
    #         if e.name == 'd':
    #             self.__head_direction = HeadDirection.RIGHT

    def __change_ai_mode(self):
        self.ai_mode = not self.ai_mode

    def render(self):
        """Process render to the screen"""
        while True:
            self.gui.render_frame(self.ai_mode, self.__board, self.__head_direction)
            time.sleep(0.1)

    def is_border(self, x: int, y: int):
        """Checks if x, y coords at the border"""
        ...

    def is_snake_body(self, x: int, y: int):
        """Checks if x, y coords at snake body"""
        ...

    @staticmethod
    def gather_empty_space(self) -> list[tuple[int, int]]:
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
        if not self.ai_mode:
            ...


if __name__ == '__main__':
    gui = GUI()
    game = Game(gui)
    game.render()
