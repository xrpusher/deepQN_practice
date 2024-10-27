import numpy as np
import cv2
import imageio  # Для записи видео
from pettingzoo import AECEnv
from gymnasium.spaces import Discrete, Box

# Параметры среды
GRID_SIZE = 5
SCALE_FACTOR = 48
NUM_EPISODES = 300
INITIAL_CHARGE = 100
FRAME_SKIP = 5
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EPSILON = 0.3  # Вероятность исследования

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
        self.q_tables = {agent: np.zeros((GRID_SIZE, GRID_SIZE, 6)) for agent in self.agents}  # Q-таблицы

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

        self._apply_actions_from_buffer()  # Применяем действия

        # Проверка завершения эпизода
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
        elif action == 4:  # Сбор картошки
            if self.grid[y][x] == 1:
                self.grid[y][x] = -1 if agent == "tractor_red" else -2
                self.rewards[agent] += 1
            else:
                self.rewards[agent] -= 0.5
        elif action == 5:  # Ожидание
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

    def choose_action(self, agent):
        """Выбирает действие на основе ε-greedy стратегии."""
        x, y = self.positions[agent]
        if np.random.rand() < EPSILON:
            return np.random.randint(0, 6)  # Исследуем случайное действие
        return np.argmax(self.q_tables[agent][y, x])  # Выбираем лучшее действие

    def centralized_critic_update(self, agent, action, reward, new_position):
        """Обновляет значение Q для заданного агента на основе централизованного критика."""
        x, y = self.positions[agent]
        new_x, new_y = new_position
        best_future_q = np.max(self.q_tables[agent][new_y, new_x])
        # Q-learning обновление
        self.q_tables[agent][y, x, action] = (1 - LEARNING_RATE) * self.q_tables[agent][y, x, action] + \
                                             LEARNING_RATE * (reward + DISCOUNT_FACTOR * best_future_q)

    def render(self):
        display_grid = np.ones((GRID_SIZE * SCALE_FACTOR, GRID_SIZE * SCALE_FACTOR, 3), dtype=np.uint8) * 255
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

        return display_grid

    def _draw_cross(self, grid, x, y, color):
        cv2.line(grid,
                 (x * SCALE_FACTOR + SCALE_FACTOR // 4, y * SCALE_FACTOR + SCALE_FACTOR // 4),
                 (x * SCALE_FACTOR + 3 * SCALE_FACTOR // 4, y * SCALE_FACTOR + 3 * SCALE_FACTOR // 4),
                 color, 2)
        cv2.line(grid,
                 (x * SCALE_FACTOR + SCALE_FACTOR // 4, y * SCALE_FACTOR + 3 * SCALE_FACTOR // 4),
                 (x * SCALE_FACTOR + 3 * SCALE_FACTOR // 4, y * SCALE_FACTOR // 4),
                 color, 2)

    def close(self):
        pass

# Основной цикл обучения и запись видео
env = PotatoFieldEnv(render_mode="rgb_array")
frames = []

for episode in range(NUM_EPISODES):
    env.reset()
    for step in range(INITIAL_CHARGE):
        actions = {agent: env.choose_action(agent) for agent in env.agents}
        env.step(actions)

        # Обновляем Q-таблицы с помощью централизованного критика
        for agent, action in actions.items():
            reward = env.rewards[agent]
            new_position = env.central_buffer[agent]
            env.centralized_critic_update(agent, action, reward, new_position)

        if step % FRAME_SKIP == 0:
            frames.append(env.render())  # Сохраняем кадры

        if all(env.terminations.values()):
            break



# Вывод максимальных наград для каждого агента
print("Max rewards for each agent:")
for agent, reward in env.rewards.items():
    print(f"{agent}: {reward:.1f}")

# Запись видео из сохранённых кадров
with imageio.get_writer("potato_field_simulation_maddpg.mp4", fps=10) as writer:
    for frame in frames:
        writer.append_data(frame)

print("Видео сохранено как potato_field_simulation_maddpg.mp4")
