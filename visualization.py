import pygame
import matplotlib.pyplot as plt

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
GRAY = (200, 200, 200)

def visualize_game(env, agent, police, episode, step, total_reward, action):
    CELL_SIZE = 50
    GRID_SIZE = env.size
    WINDOW_SIZE = (GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE + 150)  # Increased height for logs
    
    # Initialize Pygame and create the screen if it doesn't exist
    if not pygame.get_init():
        pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Reinforcement Learning Game")
    
    screen.fill(WHITE)
    
    # Draw grid
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            pygame.draw.rect(screen, BLACK, (i * CELL_SIZE, j * CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)
    
    # Draw agent
    agent_x, agent_y = env.agent_position
    pygame.draw.circle(screen, BLUE, (agent_x * CELL_SIZE + CELL_SIZE // 2, agent_y * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 3)
    
    # Draw police
    police_x, police_y = police.position
    pygame.draw.rect(screen, RED, (police_x * CELL_SIZE + CELL_SIZE // 4, police_y * CELL_SIZE + CELL_SIZE // 4, CELL_SIZE // 2, CELL_SIZE // 2))
    
    # Draw terminal state
    term_x, term_y = env.terminal_state
    pygame.draw.rect(screen, GREEN, (term_x * CELL_SIZE, term_y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    
    # Draw slippery state
    slip_x, slip_y = env.slippery_state
    pygame.draw.circle(screen, GRAY, (slip_x * CELL_SIZE + CELL_SIZE // 2, slip_y * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 2)
    
    # Draw logs
    font = pygame.font.Font(None, 24)
    log_text = [
        f"Episode: {episode + 1}",
        f"Step: {step + 1}",
        f"Total Reward: {total_reward}",
        f"Action: {['Up', 'Right', 'Down', 'Left'][action]}",
        "Blue: Agent, Red: Police, Green: Goal, Gray: Slippery"
    ]
    for i, text in enumerate(log_text):
        text_surface = font.render(text, True, BLACK)
        screen.blit(text_surface, (10, GRID_SIZE * CELL_SIZE + 10 + i * 30))
    
    pygame.display.flip()
    pygame.time.wait(100)  # Adjust speed of visualization

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            return True  # Signal to stop the game loop

    return False  # Continue the game loop

def plot_rewards(rewards):
    plt.figure(figsize=(12, 8))
    plt.plot(rewards)
    plt.title('Rewards per Episode during Training')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    
    explanation = (
        "This graph shows the total reward obtained in each training episode.\n"
        "As training progresses, we expect to see an upward trend in rewards,\n"
        "indicating that the agent is learning to perform better over time.\n"
        "Fluctuations are normal due to the exploration-exploitation trade-off\n"
        "and the stochastic nature of the environment."
    )
    plt.figtext(0.5, 0.01, explanation, wrap=True, horizontalalignment='center', fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.show()

def explain_visualization():
    print("\nVisualization Explanation:")
    print("- Blue circle: Agent")
    print("- Red square: Police officer")
    print("- Green square: Terminal state (goal)")
    print("- Gray circle: Slippery state")
    print("- Grid: Game environment")
    print("\nThe agent (blue) tries to reach the terminal state (green) while avoiding the police officer (red).")
    print("The slippery state (gray) has a 50% chance of moving the agent in a random direction.")
    print("\nPress Enter to start the game visualization...")
    input()