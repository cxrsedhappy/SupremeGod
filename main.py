import enum
import time
import curses
import random
import matplotlib.pyplot as plt

from collections import deque

import numpy as np

from agent import SnakeAgent
from rendering import GUI

WIDTH = 100
HEIGHT = 20
ASCII_NUM_BACKGROUND = ord('░')
ASCII_NUM_SNAKE = ord('█')
ASCII_NUM_APPLE = ord('▒')


class HeadDirection(enum.Enum):
    UP: tuple[int, int] = (0, -1)
    RIGHT: tuple[int, int] = (1, 0)
    DOWN: tuple[int, int] = (0, 1)
    LEFT: tuple[int, int] = (-1, 0)

    def __str__(self):
        return self.name


class Game:
    def __init__(self, gui: GUI):
        self.__gui = gui

        self.agent = SnakeAgent()
        self.iteration: int = 0
        self.time_scale: int = 0
        self.__board = [''.join([chr(ASCII_NUM_BACKGROUND) for _ in range(WIDTH)]) for _ in range(HEIGHT)]
        self.__apple_pos: tuple[int, int] = (random.randint(0, WIDTH - 1), random.randint(0, HEIGHT - 1))
        self.__head_pos: tuple[int, int] = (random.randint(0, WIDTH - 1), random.randint(0, HEIGHT - 1))
        self.__head_direction = HeadDirection.RIGHT.value
        self.__snake_body = deque([self.__head_pos])
        self.__running: bool = True      # Is game running
        self.__ai_mode: bool = True      # learning mode
        self.__ai_move: int  = 2000
        self.__score:   int  = 0         # Player's score
        self.__bx_scr:  int  = 0         # x pos where board starts
        self.__by_scr:  int  = 8         # y pos where board starts
        self.scores = []

    def _get_relative_directions(self):
        """Получение относительных направлений (лево, перед, право) относительно текущего направления"""
        if self.__head_direction == HeadDirection.UP.value:
            return [HeadDirection.LEFT.value, HeadDirection.UP.value, HeadDirection.RIGHT.value]
        elif self.__head_direction == HeadDirection.RIGHT.value:
            return [HeadDirection.UP.value, HeadDirection.RIGHT.value, HeadDirection.DOWN.value]
        elif self.__head_direction == HeadDirection.DOWN.value:
            return [HeadDirection.RIGHT.value, HeadDirection.DOWN.value, HeadDirection.LEFT.value]
        else:  # LEFT
            return [HeadDirection.DOWN.value, HeadDirection.LEFT.value, HeadDirection.UP.value]

    def __check_head_on_apple(self, next_pos: tuple[int, int]):
        if next_pos == self.__apple_pos:
            self.__apple_pos = (random.randint(0, WIDTH - 1), random.randint(0, HEIGHT - 1))
            self.__score += 1
            self.__ai_move = 2000
        else:
            self.__snake_body.popleft()

    def __render_entities(self):
        """Render snake body and apple on the board and info about game"""
        for pos in self.__snake_body:
            self.__gui.stdscr.addstr(self.__by_scr + pos[1], self.__bx_scr + pos[0], chr(ASCII_NUM_SNAKE))

        self.__gui.stdscr.addstr(
            self.__by_scr + self.__apple_pos[1], self.__bx_scr + self.__apple_pos[0], chr(ASCII_NUM_APPLE)
        )

        footer_text = f'Score: {self.__score}'
        if self.__ai_mode:
            footer_text += f' Iteration: {self.iteration}'
        self.__gui.stdscr.addstr(29, 0, f'{footer_text:^{WIDTH}}')


    def __handle_input(self, key):
        """
        Handling keyboard inputs
        TODO: find match key solution
        """
        current_move = self.__head_direction

        if key == ord('q'):
            self.__running = False
            plt.plot(np.arange(0, self.iteration), self.scores)
            plt.xlabel('Iteration')
            plt.ylabel('Score')
            plt.title('Learning Progress')
            plt.grid(True)
            plt.show()
            self.agent.save_model()
        elif key == ord('p'):
            self.__ai_mode = not self.__ai_mode
        elif key == ord('+'):
            self.time_scale += 0.01
        elif key == ord('-'):
            self.time_scale -= 0.01
        elif key == curses.KEY_UP:
            self.__head_direction = HeadDirection.UP.value
        elif key == curses.KEY_DOWN:
            self.__head_direction = HeadDirection.DOWN.value
        elif key == curses.KEY_LEFT:
            self.__head_direction = HeadDirection.LEFT.value
        elif key == curses.KEY_RIGHT:
            self.__head_direction = HeadDirection.RIGHT.value

        # Checks if head direction is apposite
        if (-1 * self.__head_direction[0], -1 * self.__head_direction[1]) == current_move:
            self.__head_direction = current_move

    def new_iteration(self):
        """Starts new iteration of learning"""
        self.__apple_pos = (random.randint(0, WIDTH - 1), random.randint(0, HEIGHT - 1))
        self.__head_pos = (random.randint(0, WIDTH - 1), random.randint(0, HEIGHT - 1))
        self.__head_direction = HeadDirection.RIGHT.value
        self.__snake_body = deque([self.__head_pos])
        self.__score = 0
        self.__ai_move = 2000
        self.iteration += 1

    def render(self):
        """Renders frame. Also handle keyboard inputs"""
        while self.__running:
            key = self.__gui.stdscr.getch()
            if key != -1:
                self.__handle_input(key)

            self.__gui.render_frame(self.__ai_mode, self.__board, self.__head_direction, self.time_scale)
            self.update()
            self.__gui.next_frame()
            time.sleep(self.time_scale)

    def update(self):
        """Process game logic"""
        match self.__ai_mode:
            case 0:
                next_pos = (
                    self.__head_pos[0] + self.__head_direction[0], self.__head_pos[1] + self.__head_direction[1]
                )

                if not self.is_boarder(next_pos) and next_pos not in self.__snake_body:
                    self.__head_pos = next_pos
                    self.__snake_body.append(next_pos)
                    self.__check_head_on_apple(next_pos)
                else:
                    self.__running = False

            case 1:
                obstacles, apple_pos = self.get_state_info()
                state = self.agent.get_state(obstacles, apple_pos)

                action = self.agent.choose_action(state, self.__head_direction)
                self.__head_direction = action

                next_pos = (
                    self.__head_pos[0] + self.__head_direction[0], self.__head_pos[1] + self.__head_direction[1]
                )

                if not self.is_boarder(next_pos) and next_pos not in self.__snake_body and self.__ai_move >= 0:
                    self.__head_pos = next_pos
                    self.__snake_body.append(next_pos)
                    self.__check_head_on_apple(next_pos)
                else:
                    if self.__score >= 15:
                        self.agent.load_model(
                            f'C:\\Users\\stasp\\PycharmProjects\\snake-reinforcement\\best\\model_iteration{self.iteration}S{self.__score}.pkl'
                        )
                    self.scores.append(self.__score)
                    self.new_iteration()

                self.__ai_move -= 1

                next_obstacles, next_apple_pos = self.get_state_info()
                next_state = self.agent.get_state(next_obstacles, next_apple_pos)

                self.agent.learn(state, action, self.get_reward(next_pos), next_state)

        self.__render_entities()

    @staticmethod
    def is_boarder(position: tuple[int, int]) -> bool:
        if 0 <= position[1] <= HEIGHT - 1 and 0 <= position[0] <= WIDTH - 1:
            return False
        return True

    def get_state_info(self):
        """Get state info"""
        obstacles = []
        apple_pos = []

        directions = self._get_relative_directions()

        for d in directions[:3]:
            next_pos = (self.__head_pos[0] + d[0], self.__head_pos[1] + d[1])
            is_obstacle = (self.is_boarder(next_pos) or next_pos in self.__snake_body)
            obstacles.append(1 if is_obstacle else 0)

        head_x, head_y = self.__head_pos
        apple_x, apple_y = self.__apple_pos

        # Left
        apple_pos.append(1 if apple_x < head_x else 0)
        # Forward
        apple_pos.append(1 if (
                (self.__head_direction == HeadDirection.UP.value and apple_y < head_y) or
                (self.__head_direction == HeadDirection.DOWN.value and apple_y > head_y) or
                (self.__head_direction == HeadDirection.RIGHT.value and apple_x > head_x) or
                (self.__head_direction == HeadDirection.LEFT.value and apple_x < head_x)
        ) else 0)
        # Right
        apple_pos.append(1 if apple_x > head_x else 0)
        # Down
        apple_pos.append(1 if apple_y > head_y else 0)

        return obstacles, apple_pos

    def get_reward(self, next_pos):
        """Reward based on move"""
        if self.is_boarder(next_pos) or next_pos in self.__snake_body:
            return -10  # Увеличим штраф за столкновение
        elif next_pos == self.__apple_pos:
            return 10  # Увеличим награду за яблоко
        else:
            current_distance = self.manhattan_distance(self.__head_pos, self.__apple_pos)
            new_distance = self.manhattan_distance(next_pos, self.__apple_pos)
            if new_distance < current_distance:
                return 1  # Увеличим награду за приближение к яблоку
            return -0.1  # Увеличим штраф за отдаление от яблока

    def manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def main(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(1)
    gui = GUI(stdscr)
    game = Game(gui)
    game.render()


if __name__ == '__main__':
    curses.wrapper(main)