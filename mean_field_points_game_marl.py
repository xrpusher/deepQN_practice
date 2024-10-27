import numpy as np
import imageio
import cv2
from pettingzoo.utils import AECEnv
from gymnasium.spaces import Discrete, Box

# Параметры среды
GRID_SIZE = 5
INITIAL_SCORE = 0
MAX_STEPS = 50
VIDEO_PATH = "meanfield_capture_game.mp4"

class MeanFieldCaptureGame(AECEnv):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)  # Поле игры
        self.possible_agents = ["agent_1", "agent_2", "agent_3", "agent_4"]
        self.agents = self.possible_agents[:]
        self.teams = {
            "team_red": ["agent_1", "agent_2"],
            "team_blue": ["agent_3", "agent_4"]
        }
        self.agent_positions = {agent: (np.random.randint(0, GRID_SIZE), np.random.randint(0, GRID_SIZE)) for agent in self.agents}
        self.scores = {agent: INITIAL_SCORE for agent in self.agents}
        self.current_step = 0
        self.action_space = Discrete(5)  # Вверх, вниз, влево, вправо, захват
        self.observation_space = Box(low=0, high=1, shape=(GRID_SIZE, GRID_SIZE), dtype=np.float32)

    def reset(self):
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        self.agent_positions = {agent: (np.random.randint(0, GRID_SIZE), np.random.randint(0, GRID_SIZE)) for agent in self.agents}
        self.scores = {agent: INITIAL_SCORE for agent in self.agents}
        self.current_step = 0
        return {agent: self.observe(agent) for agent in self.agents}

    def observe(self, agent):
        obs = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        x, y = self.agent_positions[agent]
        obs[x, y] = 1
        return obs

    def step(self, actions):
        for agent, action in actions.items():
            if self.agent_positions[agent]:  # Если агент не удалён
                self.move(agent, action)
                self.capture_cell(agent)
        
        # Завершение эпизода
        self.current_step += 1
        if self.current_step >= MAX_STEPS or all(self.scores[agent] > 10 for agent in self.agents):
            self.agents = []

    def move(self, agent, action):
        x, y = self.agent_positions[agent]
        if action == 0 and y > 0:  # Вверх
            self.agent_positions[agent] = (x, y - 1)
        elif action == 1 and y < GRID_SIZE - 1:  # Вниз
            self.agent_positions[agent] = (x, y + 1)
        elif action == 2 and x > 0:  # Влево
            self.agent_positions[agent] = (x - 1, y)
        elif action == 3 and x < GRID_SIZE - 1:  # Вправо
            self.agent_positions[agent] = (x + 1, y)

    def capture_cell(self, agent):
        x, y = self.agent_positions[agent]
        team = "team_red" if agent in self.teams["team_red"] else "team_blue"
        other_team = "team_blue" if team == "team_red" else "team_red"
        
        # Проверка состояния окружающих клеток (Mean Field)
        mean_field = np.mean([self.scores[other_agent] for other_agent in self.teams[other_team]])
        if mean_field < 5:  # Условие для победы в клетке
            self.grid[x, y] = 1 if team == "team_red" else 2  # Захват клетки командой
            self.scores[agent] += 1  # Добавление очков агенту

    def render_frame(self):
        # Создаём изображение для каждого кадра
        display_grid = np.ones((GRID_SIZE * 50, GRID_SIZE * 50, 3), dtype=np.uint8) * 255
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                # Отображение захваченных клеток
                color = (0, 0, 255) if self.grid[y, x] == 1 else (255, 0, 0) if self.grid[y, x] == 2 else (200, 200, 200)
                cv2.rectangle(display_grid, (x * 50, y * 50), ((x + 1) * 50, (y + 1) * 50), color, -1)

        # Отображаем агентов в форме кружков и добавляем текст с текущими значениями счёта
        for agent, (x, y) in self.agent_positions.items():
            color = (0, 0, 255) if agent in self.teams["team_red"] else (255, 0, 0)
            cv2.circle(display_grid, (x * 50 + 25, y * 50 + 25), 10, color, -1)
            cv2.putText(display_grid, agent[-1], (x * 50 + 15, y * 50 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Добавляем текущие результаты
        text_position = 20
        for agent, score in self.scores.items():
            cv2.putText(display_grid, f"{agent} Score: {score}", (10, text_position), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            text_position += 20

        return display_grid


# Пример использования и запись видео
env = MeanFieldCaptureGame()
env.reset()

frames = []
max_scores = {agent: 0 for agent in env.agents}

for step in range(MAX_STEPS):
    actions = {agent: np.random.randint(0, 5) for agent in env.agents}
    env.step(actions)
    frame = env.render_frame()
    frames.append(frame)
    
    # Обновляем максимальные значения счёта для логирования
    for agent in env.agents:
        max_scores[agent] = max(max_scores[agent], env.scores[agent])

# Сохранение видео
with imageio.get_writer(VIDEO_PATH, fps=5) as writer:
    for frame in frames:
        writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# Вывод максимальных результатов по каждому агенту
print("Максимальные результаты по каждому агенту:")
for agent, max_score in max_scores.items():
    print(f"{agent}: {max_score}")

print(f"Видео сохранено как {VIDEO_PATH}")
