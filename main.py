import pygame
from environment import Environment
from agent import Agent
from police import Police
from rl_algorithms import SARSA, QLearning
from visualization import visualize_game, plot_rewards, explain_visualization

def get_user_input():
    print("Select algorithm: (1) SARSA, (2) Q-Learning")
    algorithm = int(input())
    
    print("Enter grid size (max 10, recommended: 10):")
    grid_size = min(10, int(input() or "10"))
    
    print("Enter number of training episodes (recommended: 1000):")
    train_episodes = int(input() or "1000")
    
    print("Enter number of playing episodes (recommended: 10):")
    play_episodes = int(input() or "10")
    
    print("Enter gamma (discount factor) (recommended: 0.95):")
    gamma = float(input() or "0.95")
    
    print("Enter alpha (learning rate) (recommended: 0.1):")
    alpha = float(input() or "0.1")
    
    print("Enter epsilon (exploration rate) (recommended: 0.1):")
    epsilon = float(input() or "0.1")
    
    print("Enter max steps per episode (recommended: 100):")
    max_steps = int(input() or "100")
    
    print(f"Enter terminal state (x y) (recommended: {grid_size-1} {grid_size-1}):")
    terminal_state = tuple(map(int, (input() or f"{grid_size-1} {grid_size-1}").split()))
    
    print(f"Enter slippery state (x y) (recommended: {grid_size//2} {grid_size//2}):")
    slippery_state = tuple(map(int, (input() or f"{grid_size//2} {grid_size//2}").split()))
    
    return algorithm, grid_size, train_episodes, play_episodes, gamma, alpha, epsilon, max_steps, terminal_state, slippery_state

def main():
    algorithm, grid_size, train_episodes, play_episodes, gamma, alpha, epsilon, max_steps, terminal_state, slippery_state = get_user_input()
    
    env = Environment(grid_size, terminal_state, slippery_state)
    agent = Agent(env.observation_space, env.action_space)
    police = Police(grid_size)
    
    if algorithm == 1:
        rl_algorithm = SARSA(env.observation_space, env.action_space, gamma, alpha, epsilon)
    else:
        rl_algorithm = QLearning(env.observation_space, env.action_space, gamma, alpha, epsilon)
    
    # Training phase
    rewards_history = []
    for episode in range(train_episodes):
        state = env.reset()
        police.reset()
        env.update_police_position(police.position)
        total_reward = 0
        
        for step in range(max_steps):
            action = rl_algorithm.choose_action(state)
            next_state, reward, done = env.step(action)
            next_action = rl_algorithm.choose_action(next_state)
            
            if algorithm == 1:
                rl_algorithm.update(state, action, reward, next_state, next_action)
            else:
                rl_algorithm.update(state, action, reward, next_state)
            
            state = next_state
            total_reward += reward
            police.move()
            env.update_police_position(police.position)
            
            if done:
                break
        
        rewards_history.append(total_reward)
        rl_algorithm.decay_epsilon()
        print(f"Episode {episode + 1}/{train_episodes}, Total Reward: {total_reward}")
    
    # Plot rewards and explain visualization
    plot_rewards(rewards_history)
    explain_visualization()
    
    # Playing phase
    for episode in range(play_episodes):
        state = env.reset()
        police.reset()
        env.update_police_position(police.position)
        total_reward = 0
        
        for step in range(max_steps):
            action = rl_algorithm.choose_best_action(state)
            next_state, reward, done = env.step(action)
            
            state = next_state
            total_reward += reward
            police.move()
            env.update_police_position(police.position)
            
            if visualize_game(env, agent, police, episode, step, total_reward, action):
                return  # Exit the game if the window is closed
            
            if done:
                break
        
        print(f"Play Episode {episode + 1}/{play_episodes}, Total Reward: {total_reward}")

if __name__ == "__main__":
    main()