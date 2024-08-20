import random

class Police:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.center_x = grid_size // 2
        self.reset()

    def reset(self):
        self.position = (self.center_x, random.randint(0, self.grid_size - 1))
        self.direction = random.choice([-1, 1])

    def move(self):
        x, y = self.position
        y += self.direction
        if y < 0 or y >= self.grid_size:
            self.direction *= -1
            y += 2 * self.direction
        self.position = (self.center_x, y)