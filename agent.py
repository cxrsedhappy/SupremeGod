import os

import numpy as np
import random
from collections import defaultdict
import pickle


class SnakeAgent:
    def __init__(self, learning_rate=0.01, discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995):
        self.q_table = defaultdict(lambda: np.zeros(4))  # 4 possible states
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.actions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # UP, RIGHT, DOWN, LEFT

    def get_state(self, obstacles, apple_pos):
        """
        Converts the input data to a state
        obstacles: [left, front, right]
        apple_pos: [left, front, right, bottom]
        """
        return tuple(list(obstacles) + list(apple_pos))

    def choose_action(self, state, current_direction):
        """Choosing an action based on the epsilon-greedy strategy"""
        if random.random() < self.epsilon:
            valid_actions = self._get_valid_actions(current_direction)
            return random.choice(valid_actions)

        state_values = self.q_table[state]
        valid_actions = self._get_valid_actions(current_direction)
        valid_indices = [self.actions.index(action) for action in valid_actions]
        valid_values = [state_values[i] for i in valid_indices]

        max_value_index = valid_indices[np.argmax(valid_values)]
        return self.actions[max_value_index]

    def _get_valid_actions(self, current_direction):
        """Get list of valid actions"""
        opposite = (-current_direction[0], -current_direction[1])
        return [action for action in self.actions if action != opposite]

    def learn(self, state, action, reward, next_state):
        """Update Q-table"""
        action_idx = self.actions.index(action)
        old_value = self.q_table[state][action_idx]
        next_max = np.max(self.q_table[next_state])

        new_value = (1 - self.lr) * old_value + self.lr * (reward + self.gamma * next_max)
        self.q_table[state][action_idx] = new_value

        # Уменьшаем epsilon
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)

    def save_model(self, filename='best\\snake_agent.pkl'):
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

        try:
            with open(filename, 'wb') as f:
                pickle.dump(dict(self.q_table), f)
        except Exception as e:
            print(f"Error occurred when trying to save the model: {e}")

    def load_model(self, filename='best\\snake_agent.pkl'):
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

        try:
            if os.path.exists(filename):
                with open(filename, 'rb') as f:
                    self.q_table = defaultdict(lambda: np.zeros(4), pickle.load(f))
            else:
                print(f"File {filename} not found. Creating new Q-table.")
                self.q_table = defaultdict(lambda: np.zeros(4))
        except Exception as e:
            print(f"Error occurred: {e}\nCreating new Q-table.")
            self.q_table = defaultdict(lambda: np.zeros(4))