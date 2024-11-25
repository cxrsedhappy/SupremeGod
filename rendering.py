import torch
import curses

from ascii_text import ShadedBlockyFont


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
        """Renders title"""
        for i, line in enumerate(self.__title.split('\n')):
            self.stdscr.addstr(i, 0, line)

    def __render_params(self, ai_mode: bool, direction: tuple[int, int], time_scale: float):
        """Renders params section on the GUI"""
        y_offset = len(self.__title.split('\n'))

        labels = [' Torch version: ', ' AI mode: ', ' Direction: ', ' Time scale: ']
        values = [self.__torch_version, str(ai_mode), str(direction), time_scale]

        total_widths = [35, 21, 19, 20]
        value_widths = [total_widths[i] - len(labels[i]) for i in range(len(labels))]

        content = []
        for i in range(len(labels)):
            content.append(f'{labels[i]}' + f'{values[i]:^{value_widths[i]}}')

        self.stdscr.addstr(y_offset, 0, self.__style['corner'] +
                           self.__style['corner'].join(self.__style["border_h"] * w for w in total_widths) +
                           self.__style['corner'])

        # F*ck uni-code colours all my homies use gray monitors
        self.stdscr.addstr(y_offset + 1, 0,
            f"{self.__style['border_v']}" +
            self.__style['border_v'].join(content) +
            f"{self.__style['border_v']}"
        )

        self.stdscr.addstr(y_offset + 2, 0, self.__style['corner'] +
                           self.__style['corner'].join(self.__style["border_h"] * w for w in total_widths) +
                           self.__style['corner'])

    def render_frame(self, ai_mode: bool, board: list[str], head_direction: tuple[int, int], time_scale: float):
        """
        Renders all GUI interface
        You have to call next_frame() function in the game loop to display rendered text
        TODO: too many params, possible solution to use kwargs or idk **
        """
        self.stdscr.clear()
        self.__render_title()
        self.__render_params(ai_mode, head_direction, time_scale)

        y_offset = len(self.__title.split('\n')) + 3
        for y, row in enumerate(board):
            self.stdscr.addstr(y + y_offset, 0, ''.join(row))

    def next_frame(self):
        self.stdscr.refresh()