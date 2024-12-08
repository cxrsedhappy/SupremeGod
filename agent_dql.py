import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random


class DQNetwork(nn.Module):
    def __init__(self, input_size=81, hidden_size=256, output_size=4):
        super(DQNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(64 * 9 * 9, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Преобразуем входные данные в формат (batch_size, channels, height, width)
        x = x.view(-1, 1, 9, 9)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class DQLAgent:
    def __init__(self, state_size=81, action_size=4, hidden_size=256):
        self.state_size = state_size  # 9x9 = 81
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        self.target_update_frequency = 100  # Частота обновления целевой сети
        self.steps = 0  # Счетчик шагов

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQNetwork().to(self.device)
        self.target_model = DQNetwork().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.actions = [
            (-1, 0),  # LEFT
            (1, 0),  # RIGHT
            (0, -1),  # UP
            (0, 1)  # DOWN
        ]

        self.update_target_network()

    def update_target_network(self):
        """Copy weights from target network to main network."""
        self.target_model.load_state_dict(self.model.state_dict())

    def choose_action(self, state, valid_actions):
        opposite_actions = {
            (1, 0): (-1, 0),  # RIGHT -> LEFT
            (-1, 0): (1, 0),  # LEFT -> RIGHT
            (0, 1): (0, -1),  # DOWN -> UP
            (0, -1): (0, 1)  # UP -> DOWN
        }

        if len(self.memory) > 0:
            last_action = self.memory[-1][1]
            opposite = opposite_actions.get(last_action)
            if opposite in valid_actions:
                valid_actions.remove(opposite)

        if not valid_actions:  # if there is no valid actions then go random
            return random.choice(self.actions)

        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        state = state.unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)

        # invalid actions mask
        action_mask = torch.ones(self.action_size) * float('-inf')
        for action in valid_actions:
            idx = self.actions.index(action)
            action_mask[idx] = 0

        q_values = q_values + action_mask.to(self.device)
        action_idx = q_values.argmax().item()

        return self.actions[action_idx]

    def get_state(self, board):
        # Board matrix into Tensor
        state = torch.FloatTensor(board)
        return state

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        actions = torch.tensor([self.actions.index(a) for a in actions], dtype=torch.long).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q_values = self.model(states)
        next_q_values = self.target_model(next_states)

        max_next_q_values = next_q_values.max(1)[0]
        expected_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        loss = nn.MSELoss()(current_q_values.gather(1, actions.unsqueeze(1)).squeeze(),
                            expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.steps += 1
        if self.steps % self.target_update_frequency == 0:
            self.update_target_network()

    def save_model(self, filename='models\\dql_model.pth'):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filename)

    def load_model(self, filename='models\\dql_model.pth'):
        if os.path.exists(filename):
            checkpoint = torch.load(filename)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.update_target_network()