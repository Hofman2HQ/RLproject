import numpy as np

class Environment:
    def __init__(self, size, terminal_state, slippery_state):
        self.size = size
        self.terminal_state = terminal_state
        self.slippery_state = slippery_state
        self.observation_space = size * size
        self.action_space = 4  # Up, Right, Down, Left
        self.police_position = None
        self.reset()

    def reset(self):
        self.agent_position = (0, 0)
        return self._get_state()

    def update_police_position(self, position):
        self.police_position = position

    def step(self, action):
        if self.agent_position == self.slippery_state and np.random.random() < 0.5:
            action = np.random.randint(4)

        x, y = self.agent_position
        if action == 0:  # Up
            y = max(0, y - 1)
        elif action == 1:  # Right
            x = min(self.size - 1, x + 1)
        elif action == 2:  # Down
            y = min(self.size - 1, y + 1)
        elif action == 3:  # Left
            x = max(0, x - 1)

        self.agent_position = (x, y)

        done = False
        reward = -1  # Small negative reward for each step

        if self.agent_position == self.terminal_state:
            reward = 100
            done = True
        elif self.agent_position == self.police_position:
            reward = -100
            done = True

        return self._get_state(), reward, done

    def _get_state(self):
        return self.agent_position[0] * self.size + self.agent_position[1]