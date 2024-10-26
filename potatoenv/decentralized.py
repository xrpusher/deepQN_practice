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
        self.pos = {"tractor_red": [0, 0], "tractor_blue": [GRID_SIZE - 1, GRID_SIZE - 1]}
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.rewards = {agent: 0 for agent in self.agents}
        self.charges = {agent: INITIAL_CHARGE for agent in self.agents}  # Заряд каждого трактора
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.steps = 0
        self.pending_moves = {}

    def observation_space(self, agent):
        return Box(low=0, high=1, shape=(GRID_SIZE, GRID_SIZE), dtype=np.float32)

    def action_space(self, agent):
        return Discrete(6)  # Вверх, вниз, влево, вправо, собрать картошку, ждать

    def reset(self, seed=None, options=None):
        self.grid = np.ones((GRID_SIZE, GRID_SIZE))
        self.pos = {"tractor_red": [0, 0], "tractor_blue": [GRID_SIZE - 1, GRID_SIZE - 1]}
        self.rewards = {agent: 0 for agent in self.agents}
        self.charges = {agent: INITIAL_CHARGE for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.agents = self.possible_agents[:]
        self.agent_selection = self._agent_selector.next()
        self.steps = 0
        self.pending_moves = {}
        return {agent: self.observe(agent) for agent in self.agents}

    def step(self, action):
        agent = self.agent_selection
        x, y = self.pos[agent]
        new_x, new_y = x, y  # Координаты по умолчанию

        # Проверка уровня заряда
        if self.charges[agent] > 0:
            # Определяем новое положение в зависимости от действия
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
                self.rewards[agent] -= 0.1  # Небольшой штраф за ожидание

            # Уменьшение заряда на единицу за каждый шаг
            self.charges[agent] -= 1

        # Обработка ожидания из-за потенциального столкновения
        if (new_x, new_y) in self.pending_moves.values():
            # Ожидание, если другой трактор нацелился на ту же клетку
            self.rewards[agent] -= 0.1
        else:
            self.pending_moves[agent] = (new_x, new_y)

        # Переход к следующему агенту
        if self._agent_selector.is_last():
            # Применение движений после проверки всех агентов
            positions = list(self.pending_moves.values())
            if len(positions) == len(set(positions)):  # Проверка отсутствия столкновений
                for ag, pos in self.pending_moves.items():
                    self.pos[ag] = pos
            self.pending_moves.clear()
            self.agent_selection = self._agent_selector.next()
            self._end_step_processing()
        else:
            self.agent_selection = self._agent_selector.next()

    def _end_step_processing(self):
        """Проверка на завершение по сбору картошки или исчерпанию заряда."""
        self.steps += 1

        # Завершение, если весь заряд израсходован или все картошки собраны
        if all(charge <= 0 for charge in self.charges.values()) or np.sum(self.grid == 1) == 0:
            for agent in self.agents:
                self.terminations[agent] = True

    def observe(self, agent):
        obs = np.ones((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        for i, pos in enumerate(self.pos.values()):
            obs[pos[1], pos[0]] = -1 if i == 0 else -2  # Маркировка агентов
        return obs

    def render(self):
        display_grid = np.ones((GRID_SIZE * SCALE_FACTOR, GRID_SIZE * SCALE_FACTOR + 100, 3), dtype=np.uint8) * 255

        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if self.grid[y][x] == 1:
                    center = (x * SCALE_FACTOR + SCALE_FACTOR // 2, y * SCALE_FACTOR + SCALE_FACTOR // 2)
                    axes = (SCALE_FACTOR // 4, SCALE_FACTOR // 6)
                    cv2.ellipse(display_grid, center, axes, 0, 0, 360, (0, 255, 255), -1)
                elif self.grid[y][x] == -1:
                    cv2.line(display_grid,
                             (x * SCALE_FACTOR + SCALE_FACTOR // 4, y * SCALE_FACTOR + SCALE_FACTOR // 4),
                             (x * SCALE_FACTOR + 3 * SCALE_FACTOR // 4, y * SCALE_FACTOR + 3 * SCALE_FACTOR // 4),
                             (255, 0, 0), 2)
                    cv2.line(display_grid,
                             (x * SCALE_FACTOR + SCALE_FACTOR // 4, y * SCALE_FACTOR + 3 * SCALE_FACTOR // 4),
                             (x * SCALE_FACTOR + 3 * SCALE_FACTOR // 4, y * SCALE_FACTOR + SCALE_FACTOR // 4),
                             (255, 0, 0), 2)
                elif self.grid[y][x] == -2:
                    cv2.line(display_grid,
                             (x * SCALE_FACTOR + SCALE_FACTOR // 4, y * SCALE_FACTOR + SCALE_FACTOR // 4),
                             (x * SCALE_FACTOR + 3 * SCALE_FACTOR // 4, y * SCALE_FACTOR + 3 * SCALE_FACTOR // 4),
                             (0, 0, 255), 2)
                    cv2.line(display_grid,
                             (x * SCALE_FACTOR + SCALE_FACTOR // 4, y * SCALE_FACTOR + 3 * SCALE_FACTOR // 4),
                             (x * SCALE_FACTOR + 3 * SCALE_FACTOR // 4, y * SCALE_FACTOR + SCALE_FACTOR // 4),
                             (0, 0, 255), 2)

        for agent, pos in self.pos.items():
            color = (255, 0, 0) if agent == "tractor_red" else (0, 0, 255)
            cv2.rectangle(display_grid,
                          (pos[0] * SCALE_FACTOR, pos[1] * SCALE_FACTOR),
                          ((pos[0] + 1) * SCALE_FACTOR, (pos[1] + 1) * SCALE_FACTOR),
                          color, -1)

        for i, agent in enumerate(self.possible_agents):
            reward_text = f"{agent}: {self.rewards[agent]:.1f} | Charge: {self.charges[agent]}"
            cv2.putText(display_grid, reward_text, (10, 20 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        return display_grid

    def close(self):
        pass

# Сохранение видео
env = PotatoFieldEnv(render_mode="rgb_array")
frames = []
max_rewards = {agent: float('-inf') for agent in env.possible_agents}  # Для отслеживания максимальных наград

for episode in range(NUM_EPISODES):
    env.reset()
    print(f"Эпизод {episode + 1}/{NUM_EPISODES}")
    for step in range(INITIAL_CHARGE):
        for agent in env.agents:
            action = env.action_space(agent).sample()
            env.step(action)
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
