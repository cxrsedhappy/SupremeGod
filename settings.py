import enum

WIDTH = 100
HEIGHT = 20

ASCII_NUM_BACKGROUND = ord('░')
ASCII_NUM_APPLE = ord('▒')
ASCII_NUM_SNAKE = ord('█')
MOVES_PER_ITER = 2000


class Modes(enum.Enum):
    PLAYER: int = 0
    QL: int = 1
    DQL: int = 2


class HeadDirection(enum.Enum):
    UP: tuple[int, int] = (0, -1)
    RIGHT: tuple[int, int] = (1, 0)
    DOWN: tuple[int, int] = (0, 1)
    LEFT: tuple[int, int] = (-1, 0)

    def __str__(self):
        return self.name