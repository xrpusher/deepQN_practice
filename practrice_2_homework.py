import numpy as np
import random
import imageio
import cv2
from pettingzoo.utils import ParallelEnv
from gymnasium.spaces import Discrete, Box

# Параметры игрового поля
GRID_SIZE = 10  # Размер игрового поля 10x10
NUM_AGENTS = 3  # Количество агентов (танков)
CELL_SIZE = 32  # Размер ячейки для визуализации
BULLET_SIZE = CELL_SIZE // 4  # Размер пули как четверть клетки
MAX_STEPS = 500  # Максимальное количество шагов в эпизоде
NUM_EPISODES = 5  # Количество эпизодов для обучения

class TankEnv(ParallelEnv):
    metadata = {'render_modes': ['rgb_array'], 'name': 'tank_v0'}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.possible_agents = [f"agent_{i}" for i in range(NUM_AGENTS)]
        self.agent_name_mapping = {name: i for i, name in enumerate(self.possible_agents)}
        
        # Определяем пространство действий и наблюдений для каждого агента
        self.action_spaces = {agent: Discrete(4) for agent in self.possible_agents}  # 4 действия
        self.observation_spaces = {
            agent: Box(low=0, high=1, shape=(GRID_SIZE, GRID_SIZE, 3), dtype=np.float32) 
            for agent in self.possible_agents
        }
        
        self.grid = None
        self.agent_positions = {}
        self.agent_directions = {}
        self.bullets = []
        self.steps = 0

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.grid = np.ones((GRID_SIZE, GRID_SIZE, 3), dtype=np.float32)  # Белый фон
        self.agent_positions = {agent: [random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)] for agent in self.agents}
        self.agent_directions = {agent: random.choice(['up', 'down', 'left', 'right']) for agent in self.agents}
        self.bullets = []
        self.steps = 0  # Счетчик шагов
        
        # Начальные награды, завершения и трюкации
        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        return self._observe_all(), self.infos

    def step(self, actions):
        for agent in self.agents:
            self.rewards[agent] = -0.1  # Штраф за шаг
            
        # Обработка действий агентов
        for agent, action in actions.items():
            if action == 0:  # движение вперед
                self._move_forward(agent)
            elif action == 1:  # поворот влево
                self._turn_left(agent)
            elif action == 2:  # поворот вправо
                self._turn_right(agent)
            elif action == 3:  # выстрел
                self._shoot(agent)
        
        # Обновление позиций пуль и проверка столкновений
        self._update_bullets()
        self.steps += 1
        
        # Проверка завершения игры по достижении максимального количества шагов
        if self.steps >= MAX_STEPS:
            for agent in self.agents:
                self.truncations[agent] = True
        
        self.agents = [agent for agent in self.agents if not self.terminations[agent]]
        return self._observe_all(), self.rewards, self.terminations, self.truncations, self.infos

    def _move_forward(self, agent):
        x, y = self.agent_positions[agent]
        direction = self.agent_directions[agent]
        if direction == 'up' and y > 0:
            y -= 1
        elif direction == 'down' and y < GRID_SIZE - 1:
            y += 1
        elif direction == 'left' and x > 0:
            x -= 1
        elif direction == 'right' and x < GRID_SIZE - 1:
            x += 1
        self.agent_positions[agent] = [x, y]

    def _turn_left(self, agent):
        directions = ['up', 'left', 'down', 'right']
        idx = directions.index(self.agent_directions[agent])
        self.agent_directions[agent] = directions[(idx + 1) % 4]

    def _turn_right(self, agent):
        directions = ['up', 'left', 'down', 'right']
        idx = directions.index(self.agent_directions[agent])
        self.agent_directions[agent] = directions[(idx - 1) % 4]

    def _shoot(self, agent):
        x, y = self.agent_positions[agent]
        direction = self.agent_directions[agent]
        self.bullets.append({'position': [x, y], 'direction': direction, 'agent': agent})

    def _update_bullets(self):
        new_bullets = []
        for bullet in self.bullets:
            x, y = bullet['position']
            direction = bullet['direction']
            
            # Обновление позиции пули
            if direction == 'up' and y > 0:
                y -= 1
            elif direction == 'down' and y < GRID_SIZE - 1:
                y += 1
            elif direction == 'left' and x > 0:
                x -= 1
            elif direction == 'right' and x < GRID_SIZE - 1:
                x += 1
            bullet['position'] = [x, y]
            
            # Проверка попадания в танк
            hit = False
            for agent, pos in self.agent_positions.items():
                if pos == [x, y] and agent != bullet['agent']:
                    self.rewards[bullet['agent']] += 1  # награда за попадание
                    self.rewards[agent] -= 1  # штраф за получение попадания
                    self.terminations[agent] = True  # завершение игры для пораженного агента
                    hit = True
                    break
            
            # Пуля исчезает, если она попала в танк или вышла за границы поля
            if not hit and 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                new_bullets.append(bullet)
        
        self.bullets = new_bullets

    def _observe_all(self):
        obs = {}
        for agent in self.agents:
            obs[agent] = self._observe(agent)
        return obs

    def _observe(self, agent):
        obs = np.ones((GRID_SIZE, GRID_SIZE, 3), dtype=np.float32)  # Белый фон
        for i, (ag, pos) in enumerate(self.agent_positions.items()):
            color = [0, 1, 0] if i == 0 else [1, 0, 0] if i == 1 else [0, 0, 1]
            obs[pos[1], pos[0]] = color

        for bullet in self.bullets:
            x, y = bullet['position']
            obs[y, x] = [0, 0, 0]  # Черный для пуль
        return obs

    def render(self):
        display_grid = np.ones((GRID_SIZE * CELL_SIZE + 20, GRID_SIZE * CELL_SIZE, 3), dtype=np.float32)
        
        # Отображаем танки и награды
        for i, (agent, pos) in enumerate(self.agent_positions.items()):
            color = [0, 1, 0] if i == 0 else [1, 0, 0] if i == 1 else [0, 0, 1]
            x, y = pos
            display_grid[y * CELL_SIZE: (y + 1) * CELL_SIZE, x * CELL_SIZE: (x + 1) * CELL_SIZE] = color
            
            # Отметка крестиком для побежденных танков
            if self.terminations[agent]:
                center_x, center_y = (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2)
                cv2.line(display_grid, (center_x - CELL_SIZE // 4, center_y - CELL_SIZE // 4),
                         (center_x + CELL_SIZE // 4, center_y + CELL_SIZE // 4), (0, 0, 0), 2)
                cv2.line(display_grid, (center_x - CELL_SIZE // 4, center_y + CELL_SIZE // 4),
                         (center_x + CELL_SIZE // 4, center_y - CELL_SIZE // 4), (0, 0, 0), 2)

        # Отображаем пули
        for bullet in self.bullets:
            x, y = bullet['position']
            bx, by = x * CELL_SIZE + CELL_SIZE // 2 - BULLET_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2 - BULLET_SIZE // 2
            display_grid[by:by + BULLET_SIZE, bx:bx + BULLET_SIZE] = [0, 0, 0]  # Черный цвет пуль

        # Отображение наград внизу экрана с уменьшенным шрифтом
        for i, agent in enumerate(self.possible_agents):
            reward_text = f"{agent}: {self.rewards[agent]:.1f}" + (" (dead)" if self.terminations[agent] else "")
            cv2.putText(display_grid, reward_text, (i * CELL_SIZE * 4, GRID_SIZE * CELL_SIZE + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.1, (0, 0, 0), 1, cv2.LINE_AA)
        
        return (display_grid * 255).astype(np.uint8)

    def close(self):
        pass

# Запуск среды и запись видео с несколькими эпизодами
env = TankEnv(render_mode="rgb_array")
frames = []

for episode in range(NUM_EPISODES):
    observations, infos = env.reset()
    print(f"Эпизод {episode + 1}/{NUM_EPISODES}")
    for step in range(MAX_STEPS):
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        frame = env.render()
        frames.append(frame)
        print(f"Шаг {step + 1}, Награды: {rewards}")
        
        # Останавливаем эпизод, если все агенты завершили игру
        if all(terminations.values()):
            break

env.close()

# Сохранение видео
with imageio.get_writer('tank_simulation_marl.mp4', fps=10) as writer:
    for frame in frames:
        writer.append_data(frame)

print("Видео сохранено как tank_simulation_marl.mp4")
