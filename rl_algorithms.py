import numpy as np

class RLAlgorithm:
    def __init__(self, observation_space, action_space, gamma, alpha, epsilon):
        self.observation_space = observation_space
        self.action_space = action_space
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.min_epsilon = 0.01
        self.epsilon_decay = 0.995
        self.q_table = np.zeros((observation_space, action_space))

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_space)
        else:
            return np.argmax(self.q_table[state])

    def choose_best_action(self, state):
        return np.argmax(self.q_table[state])

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

class SARSA(RLAlgorithm):
    def update(self, state, action, reward, next_state, next_action):
        current_q = self.q_table[state, action]
        next_q = self.q_table[next_state, next_action]
        new_q = current_q + self.alpha * (reward + self.gamma * next_q - current_q)
        self.q_table[state, action] = new_q

class QLearning(RLAlgorithm):
    def update(self, state, action, reward, next_state):
        current_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state])
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state, action] = new_q