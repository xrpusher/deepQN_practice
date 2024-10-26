import numpy as np
import imageio
import cv2
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from gymnasium.spaces import Discrete, Box

# Environment parameters
GRID_SIZE = 5
SCALE_FACTOR = 48  # Увеличен масштаб для улучшения качества изображения
NUM_EPISODES = 200  # Количество эпизодов
INITIAL_CHARGE = 100  # Начальный уровень заряда для каждого трактора
FRAME_SKIP = 5  # Сохраняем каждый 5-й кадр

class PotatoFieldEnv(AECEnv):
    metadata = {"render_modes": ["rgb_array"], "is_parallelizable": True}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.grid = np.ones((GRID_SIZE, GRID_SIZE))
        self.possible_agents = ["tractor_red", "tractor_blue"]
        self.agents = self.possible_agents[:]
        self.positions = {"tractor_red": (0, 0), "tractor_blue": (GRID_SIZE - 1, GRID_SIZE - 1)}
        self.rewards = {agent: 0 for agent in self.agents}
        self.charges = {agent: INITIAL_CHARGE for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.central_buffer = {}
        self.max_rewards = {agent: float('-inf') for agent in self.possible_agents}  # Track max rewards

    def observation_space(self, agent):
        return Box(low=0, high=1, shape=(GRID_SIZE, GRID_SIZE), dtype=np.float32)

    def action_space(self, agent):
        return Discrete(6)

    def reset(self, seed=None, options=None):
        self.grid = np.ones((GRID_SIZE, GRID_SIZE))
        self.positions = {"tractor_red": (0, 0), "tractor_blue": (GRID_SIZE - 1, GRID_SIZE - 1)}
        self.rewards = {agent: 0 for agent in self.agents}
        self.charges = {agent: INITIAL_CHARGE for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.central_buffer = {agent: None for agent in self.agents}
        return {agent: self.observe(agent) for agent in self.agents}

    def step(self, actions):
        for agent, action in actions.items():
            if self.charges[agent] > 0 and not self.terminations[agent]:
                self.charges[agent] -= 1
                self.central_buffer[agent] = self._propose_move(agent, action)

        self._apply_actions_from_buffer()
        
        if all(charge <= 0 for charge in self.charges.values()) or np.sum(self.grid == 1) == 0:
            for agent in self.agents:
                self.terminations[agent] = True

    def _propose_move(self, agent, action):
        x, y = self.positions[agent]
        new_x, new_y = x, y

        if action == 0 and y > 0:
            new_y -= 1
        elif action == 1 and y < GRID_SIZE - 1:
            new_y += 1
        elif action == 2 and x > 0:
            new_x -= 1
        elif action == 3 and x < GRID_SIZE - 1:
            new_x += 1
        elif action == 4:
            if self.grid[y][x] == 1:
                self.grid[y][x] = -1 if agent == "tractor_red" else -2
                self.rewards[agent] += 1
            else:
                self.rewards[agent] -= 0.5
        elif action == 5:
            self.rewards[agent] -= 0.1

        return (new_x, new_y) if action in [0, 1, 2, 3] else self.positions[agent]

    def _apply_actions_from_buffer(self):
        positions = list(self.central_buffer.values())
        if len(positions) == len(set(positions)):
            for agent, pos in self.central_buffer.items():
                self.positions[agent] = pos
        else:
            for agent, pos in self.central_buffer.items():
                if positions.count(pos) == 1:
                    self.positions[agent] = pos
                else:
                    self.rewards[agent] -= 0.1

    def observe(self, agent):
        obs = np.ones((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        for i, pos in enumerate(self.positions.values()):
            obs[pos[1], pos[0]] = -1 if i == 0 else -2
        return obs

    def render(self):
        display_grid = np.ones((GRID_SIZE * SCALE_FACTOR, GRID_SIZE * SCALE_FACTOR + 50, 3), dtype=np.uint8) * 255

        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if self.grid[y][x] == 1:
                    center = (x * SCALE_FACTOR + SCALE_FACTOR // 2, y * SCALE_FACTOR + SCALE_FACTOR // 2)
                    axes = (SCALE_FACTOR // 4, SCALE_FACTOR // 6)
                    cv2.ellipse(display_grid, center, axes, 0, 0, 360, (0, 255, 255), -1)
                elif self.grid[y][x] == -1:
                    self._draw_cross(display_grid, x, y, (255, 0, 0))
                elif self.grid[y][x] == -2:
                    self._draw_cross(display_grid, x, y, (0, 0, 255))

        for agent, pos in self.positions.items():
            color = (255, 0, 0) if agent == "tractor_red" else (0, 0, 255)
            cv2.rectangle(display_grid,
                          (pos[0] * SCALE_FACTOR, pos[1] * SCALE_FACTOR),
                          ((pos[0] + 1) * SCALE_FACTOR, (pos[1] + 1) * SCALE_FACTOR),
                          color, -1)

        for i, agent in enumerate(self.possible_agents):
            reward_text = f"{agent}: {self.rewards[agent]:.1f} | Charge: {self.charges[agent]}"
            cv2.putText(display_grid, reward_text, (10, 15 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)

        return display_grid

    def _draw_cross(self, grid, x, y, color):
        cv2.line(grid,
                 (x * SCALE_FACTOR + SCALE_FACTOR // 4, y * SCALE_FACTOR + SCALE_FACTOR // 4),
                 (x * SCALE_FACTOR + 3 * SCALE_FACTOR // 4, y * SCALE_FACTOR + 3 * SCALE_FACTOR // 4),
                 color, 2)
        cv2.line(grid,
                 (x * SCALE_FACTOR + SCALE_FACTOR // 4, y * SCALE_FACTOR + 3 * SCALE_FACTOR // 4),
                 (x * SCALE_FACTOR + 3 * SCALE_FACTOR // 4, y * SCALE_FACTOR + SCALE_FACTOR // 4),
                 color, 2)

    def close(self):
        pass

# Run the simulation and track max rewards
env = PotatoFieldEnv(render_mode="rgb_array")
frames = []

for episode in range(NUM_EPISODES):
    env.reset()
    for step in range(INITIAL_CHARGE):
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        env.step(actions)
        
        if step % FRAME_SKIP == 0:
            frames.append(env.render())
        
        if all(env.terminations.values()):
            break

    # Update max rewards after each episode
    for agent in env.possible_agents:
        env.max_rewards[agent] = max(env.max_rewards[agent], env.rewards[agent])

# Save video
with imageio.get_writer("potato_field_simulation_optimized.mp4", fps=10) as writer:
    for frame in frames:
        writer.append_data(frame)

# Print max rewards for each agent across episodes
print("Max rewards across all episodes:")
for agent, reward in env.max_rewards.items():
    print(f"{agent}: {reward:.1f}")

print("Video saved as potato_field_simulation_optimized.mp4")
