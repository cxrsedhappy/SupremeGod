import enum
import time
import torch
import curses
import random

from ascii_text import ShadedBlockyFont

WIDTH = 100
HEIGHT = 20
ASCII_NUM_BACKGROUND = ord('░')
ASCII_NUM_SNAKE = ord('▒')
ASCII_NUM_APPLE = ord('█')


class HeadDirection(enum.Enum):
    UP: tuple[int, int] = (0, -1)
    RIGHT: tuple[int, int] = (1, 0)
    DOWN: tuple[int, int] = (0, 1)
    LEFT: tuple[int, int] = (-1, 0)

    def __str__(self):
        return self.name


class GUI:
    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.__title = ShadedBlockyFont().render('supremegod')
        self.__device_name: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.__torch_version = torch.__version__
        self.__style = {
            'border_h': '=',
            'border_v': '║',
            'corner': '╬',
            'padding': 1
        }
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)

    def __render_title(self):
        for i, line in enumerate(self.__title.split('\n')):
            self.stdscr.addstr(i, 0, line)

    def __render_params(self, ai_mode: bool, direction: tuple[int, int]):
        y_offset = len(self.__title.split('\n'))

        labels = [' Torch version: ', ' AI mode: ', ' Direction: ', ' AI move: ']
        values = [self.__torch_version, str(ai_mode), str(direction), '[0, 0, 1]']

        total_widths = [35, 21, 19, 20]
        value_widths = [total_widths[i] - len(labels[i]) for i in range(len(labels))]

        content = []
        for i in range(len(labels)):
            content.append(f'{labels[i]}' + f'{values[i]:^{value_widths[i]}}')

        self.stdscr.addstr(y_offset, 0, self.__style['corner'] +
                           self.__style['corner'].join(self.__style["border_h"] * w for w in total_widths) +
                           self.__style['corner'])

        # F*ck colours, all my homies use gray monitors
        self.stdscr.addstr(y_offset + 1, 0,
            f"{self.__style['border_v']}" +
            self.__style['border_v'].join(content) +
            f"{self.__style['border_v']}"
        )

        self.stdscr.addstr(y_offset + 2, 0, self.__style['corner'] +
                           self.__style['corner'].join(self.__style["border_h"] * w for w in total_widths) +
                           self.__style['corner'])

    def render_frame(self, ai_mode: bool, board: list[str], head_direction: tuple[int, int]):
        self.stdscr.clear()
        self.__render_title()
        self.__render_params(ai_mode, head_direction)

        y_offset = len(self.__title.split('\n')) + 3
        for y, row in enumerate(board):
            self.stdscr.addstr(y + y_offset, 0, ''.join(row))

        self.stdscr.refresh()


class Game:
    def __init__(self, gui: GUI):
        self.__gui = gui
        self.__ai_mode = False
        self.__board = [''.join([chr(ASCII_NUM_BACKGROUND) for _ in range(WIDTH)]) for _ in range(HEIGHT)]
        self.__head_pos = (random.randint(0, WIDTH - 1), random.randint(0, HEIGHT - 1))
        self.__head_direction = HeadDirection.RIGHT.value
        self.__running = True

    def handle_input(self, key):
        if key == ord('q'):
            self.__running = False
        elif key == ord('p'):
            self.__ai_mode = not self.__ai_mode
        elif key == curses.KEY_UP:
            self.__head_direction = HeadDirection.UP.value
        elif key == curses.KEY_DOWN:
            self.__head_direction = HeadDirection.DOWN.value
        elif key == curses.KEY_LEFT:
            self.__head_direction = HeadDirection.LEFT.value
        elif key == curses.KEY_RIGHT:
            self.__head_direction = HeadDirection.RIGHT.value

    def render(self):
        while self.__running:
            key = self.__gui.stdscr.getch()
            if key != -1:
                self.handle_input(key)

            self.update()
            self.__gui.render_frame(self.__ai_mode, self.__board, self.__head_direction)
            time.sleep(0.1)

    def update(self):
        if not self.__ai_mode:
            ...


def main(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(1)
    gui = GUI(stdscr)
    game = Game(gui)
    game.render()


if __name__ == '__main__':
    curses.wrapper(main)