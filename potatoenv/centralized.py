import numpy as np
import imageio
import cv2
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from gymnasium.spaces import Discrete, Box

# Environment parameters
GRID_SIZE = 5
SCALE_FACTOR = 32  # Масштабирование ячейки для визуализации
NUM_EPISODES = 500  # Количество эпизодов
INITIAL_CHARGE = 100  # Начальный уровень заряда для каждого трактора

class PotatoFieldEnv(AECEnv):
    metadata = {"render_modes": ["rgb_array"], "is_parallelizable": True}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.grid = np.ones((GRID_SIZE, GRID_SIZE))  # 1 - картошка, 0 - пусто
        self.possible_agents = ["tractor_red", "tractor_blue"]
        self.agents = self.possible_agents[:]
        self.positions = {"tractor_red": (0, 0), "tractor_blue": (GRID_SIZE - 1, GRID_SIZE - 1)}
        self.rewards = {agent: 0 for agent in self.agents}
        self.charges = {agent: INITIAL_CHARGE for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.central_buffer = {}  # Централизованный буфер для координации действий агентов

    def observation_space(self, agent):
        return Box(low=0, high=1, shape=(GRID_SIZE, GRID_SIZE), dtype=np.float32)

    def action_space(self, agent):
        return Discrete(6)  # Вверх, вниз, влево, вправо, собрать картошку, ждать

    def reset(self, seed=None, options=None):
        self.grid = np.ones((GRID_SIZE, GRID_SIZE))
        self.positions = {"tractor_red": (0, 0), "tractor_blue": (GRID_SIZE - 1, GRID_SIZE - 1)}
        self.rewards = {agent: 0 for agent in self.agents}
        self.charges = {agent: INITIAL_CHARGE for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.central_buffer = {agent: None for agent in self.agents}  # Сброс буфера
        return {agent: self.observe(agent) for agent in self.agents}

    def step(self, actions):
        """Метод step принимает централизованные действия для всех агентов и координирует их."""
        for agent, action in actions.items():
            if self.charges[agent] > 0 and not self.terminations[agent]:
                # Уменьшаем заряд и выполняем действие
                self.charges[agent] -= 1
                self.central_buffer[agent] = self._propose_move(agent, action)

        # Применение действий из буфера с учетом коллизий
        self._apply_actions_from_buffer()
        
        # Проверка завершения эпизода
        if all(charge <= 0 for charge in self.charges.values()) or np.sum(self.grid == 1) == 0:
            for agent in self.agents:
                self.terminations[agent] = True

    def _propose_move(self, agent, action):
        """Предлагает новое положение или действие для агента."""
        x, y = self.positions[agent]
        new_x, new_y = x, y

        if action == 0 and y > 0:  # Вверх
            new_y -= 1
        elif action == 1 and y < GRID_SIZE - 1:  # Вниз
            new_y += 1
        elif action == 2 and x > 0:  # Влево
            new_x -= 1
        elif action == 3 and x < GRID_SIZE - 1:  # Вправо
            new_x += 1
        elif action == 4:  # Собрать картошку
            if self.grid[y][x] == 1:
                self.grid[y][x] = -1 if agent == "tractor_red" else -2
                self.rewards[agent] += 1
            else:
                self.rewards[agent] -= 0.5
        elif action == 5:  # Ожидание
            self.rewards[agent] -= 0.1

        return (new_x, new_y) if action in [0, 1, 2, 3] else self.positions[agent]

    def _apply_actions_from_buffer(self):
        """Применяет действия из буфера с проверкой на коллизии."""
        positions = list(self.central_buffer.values())
        if len(positions) == len(set(positions)):  # Отсутствие коллизий
            for agent, pos in self.central_buffer.items():
                self.positions[agent] = pos
        else:  # Обработка конфликтов
            for agent, pos in self.central_buffer.items():
                if positions.count(pos) == 1:
                    self.positions[agent] = pos
                else:
                    self.rewards[agent] -= 0.1  # Штраф за конфликт

    def observe(self, agent):
        """Возвращает наблюдение для агента с учетом его текущего положения и состояния поля."""
        obs = np.ones((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        for i, pos in enumerate(self.positions.values()):
            obs[pos[1], pos[0]] = -1 if i == 0 else -2
        return obs

    def render(self):
        """Отрисовывает текущее состояние поля и агентов."""
        display_grid = np.ones((GRID_SIZE * SCALE_FACTOR, GRID_SIZE * SCALE_FACTOR + 100, 3), dtype=np.uint8) * 255

        # Отображение картошки и пометок
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

        # Отображение тракторов
        for agent, pos in self.positions.items():
            color = (255, 0, 0) if agent == "tractor_red" else (0, 0, 255)
            cv2.rectangle(display_grid,
                          (pos[0] * SCALE_FACTOR, pos[1] * SCALE_FACTOR),
                          ((pos[0] + 1) * SCALE_FACTOR, (pos[1] + 1) * SCALE_FACTOR),
                          color, -1)

        # Отображение награды и заряда
        for i, agent in enumerate(self.possible_agents):
            reward_text = f"{agent}: {self.rewards[agent]:.1f} | Charge: {self.charges[agent]}"
            cv2.putText(display_grid, reward_text, (10, 20 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        return display_grid

    def _draw_cross(self, grid, x, y, color):
        """Рисует крест на позиции (x, y) с заданным цветом."""
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

# Сохранение видео и расчет максимальных наград
env = PotatoFieldEnv(render_mode="rgb_array")
frames = []
max_rewards = {agent: float('-inf') for agent in env.possible_agents}  # Инициализация для максимальных наград

for episode in range(NUM_EPISODES):
    env.reset()
    print(f"Эпизод {episode + 1}/{NUM_EPISODES}")
    for step in range(INITIAL_CHARGE):
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        env.step(actions)
        frames.append(env.render())
        if all(env.terminations.values()):
            break

    # Обновляем максимальные награды после каждого эпизода
    for agent in env.agents:
        max_rewards[agent] = max(max_rewards[agent], env.rewards[agent])

# Сохраняем видео
with imageio.get_writer("potato_field_simulation_manual.mp4", fps=10) as writer:
    for frame in frames:
        writer.append_data(frame)

# Выводим максимальные награды после всех эпизодов
print("Максимальные награды за все эпизоды:")
for agent, reward in max_rewards.items():
    print(f"{agent}: {reward:.1f}")

print("Видео сохранено как potato_field_simulation_manual.mp4")
