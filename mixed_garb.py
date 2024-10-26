import numpy as np
import imageio
import cv2
import matplotlib.pyplot as plt
import os
import pickle
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from gymnasium.spaces import Discrete, Box

# Параметры игры
GRID_SIZE = 6
SCALE_FACTOR = 32
INITIAL_CHARGE = 100
NUM_EPISODES = 1000
SAVE_PATH = "training_state_teams.pkl"
VIDEO_PATH = "garbage_collection_training.mp4"

class GarbageCollectionEnv(AECEnv):
    metadata = {"render_modes": ["rgb_array"], "is_parallelizable": True}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE))
        self.possible_agents = ["robot_sber_1", "robot_sber_2", "robot_yandex_1", "robot_yandex_2"]
        self.agents = self.possible_agents[:]
        self.teams = {
            "sber": ["robot_sber_1", "robot_sber_2"],
            "yandex": ["robot_yandex_1", "robot_yandex_2"]
        }
        self.positions = {
            "robot_sber_1": (0, 0),
            "robot_sber_2": (0, GRID_SIZE - 1),
            "robot_yandex_1": (GRID_SIZE - 1, 0),
            "robot_yandex_2": (GRID_SIZE - 1, GRID_SIZE - 1)
        }
        self.rewards = {agent: 0 for agent in self.agents}
        self.charges = {agent: INITIAL_CHARGE for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.episode_rewards = []
        self.elo_scores = {"sber": 1200, "yandex": 1200}  # ELO-рейтинг для каждой компании
        self.load_training_state()

    def observation_space(self, agent):
        return Box(low=0, high=1, shape=(GRID_SIZE, GRID_SIZE), dtype=np.float32)

    def action_space(self, agent):
        return Discrete(6)  # Вверх, вниз, влево, вправо, собрать мусор, ждать

    def reset(self, seed=None, options=None):
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE))
        self.positions = {
            "robot_sber_1": (0, 0),
            "robot_sber_2": (0, GRID_SIZE - 1),
            "robot_yandex_1": (GRID_SIZE - 1, 0),
            "robot_yandex_2": (GRID_SIZE - 1, GRID_SIZE - 1)
        }
        self.rewards = {agent: 0 for agent in self.agents}
        self.charges = {agent: INITIAL_CHARGE for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.generate_garbage()
        return {agent: self.observe(agent) for agent in self.agents}

    def generate_garbage(self):
        for _ in range(GRID_SIZE):
            x, y = np.random.randint(0, GRID_SIZE, size=2)
            self.grid[x, y] = 1  # Метка для мусора

    def step(self, actions):
        for agent, action in actions.items():
            if self.charges[agent] > 0 and not self.terminations[agent]:
                self.charges[agent] -= 1
                self._move(agent, action)

        if all(self.terminations.values()) or all(self.grid.flatten() == 0):
            self.episode_rewards.append(sum(self.rewards.values()))

    def _move(self, agent, action):
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
        elif action == 4 and self.grid[x, y] == 1:  # Сбор мусора
            self.grid[x, y] = 0
            self.rewards[agent] += 1
        elif action == 5:  # Ожидание
            self.rewards[agent] -= 0.1

        self.positions[agent] = (new_x, new_y)

    def observe(self, agent):
        obs = np.ones((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        for i, pos in enumerate(self.positions.values()):
            obs[pos[1], pos[0]] = -1 if i == 0 else -2
        return obs

    def render(self):
        """Отрисовка состояния, включающая ELO и награды компаний."""
        display_grid = np.ones((GRID_SIZE * SCALE_FACTOR, GRID_SIZE * SCALE_FACTOR + 100, 3), dtype=np.uint8) * 255
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if self.grid[y][x] == 1:
                    center = (x * SCALE_FACTOR + SCALE_FACTOR // 2, y * SCALE_FACTOR + SCALE_FACTOR // 2)
                    axes = (SCALE_FACTOR // 4, SCALE_FACTOR // 6)
                    cv2.ellipse(display_grid, center, axes, 0, 0, 360, (0, 255, 0), -1)
        
        for agent, pos in self.positions.items():
            color = (255, 0, 0) if "sber" in agent else (0, 0, 255)
            cv2.rectangle(display_grid, (pos[0] * SCALE_FACTOR, pos[1] * SCALE_FACTOR),
                          ((pos[0] + 1) * SCALE_FACTOR, (pos[1] + 1) * SCALE_FACTOR), color, -1)

        # Отображение среднего ELO и наград команд (уменьшенный шрифт)
        sber_reward = sum(self.rewards[agent] for agent in self.teams["sber"])
        yandex_reward = sum(self.rewards[agent] for agent in self.teams["yandex"])
        
        text_sber = f"Sber: Reward={sber_reward}, ELO={self.elo_scores['sber']:.1f}"
        text_yandex = f"Yandex: Reward={yandex_reward}, ELO={self.elo_scores['yandex']:.1f}"
        
        cv2.putText(display_grid, text_sber, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(display_grid, text_yandex, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)

        return display_grid

    def update_elo(self):
        """Обновление ELO рейтинга на основе командных результатов."""
        sber_reward = sum(self.rewards[agent] for agent in self.teams["sber"])
        yandex_reward = sum(self.rewards[agent] for agent in self.teams["yandex"])

        ea = 1 / (1 + 10 ** ((self.elo_scores["yandex"] - self.elo_scores["sber"]) / 400))
        eb = 1 - ea
        k = 32  # Коэффициент K для рейтинга ELO
        self.elo_scores["sber"] += k * ((1 if sber_reward > yandex_reward else 0 if sber_reward == yandex_reward else -1) - ea)
        self.elo_scores["yandex"] += k * ((1 if yandex_reward > sber_reward else 0 if yandex_reward == sber_reward else -1) - eb)

    def save_training_state(self, episode):
        with open(SAVE_PATH, "wb") as f:
            pickle.dump({
                "episode_rewards": self.episode_rewards,
                "elo_scores": self.elo_scores,
                "last_episode": episode
            }, f)

    def load_training_state(self):
        if os.path.exists(SAVE_PATH):
            with open(SAVE_PATH, "rb") as f:
                data = pickle.load(f)
                self.episode_rewards = data["episode_rewards"]
                self.elo_scores = data["elo_scores"]
                return data["last_episode"]
        return 0  # Начать с нуля, если данных нет

def main():
    env = GarbageCollectionEnv()
    last_episode = env.load_training_state()
    print(f"Продолжение обучения с эпизода {last_episode + 1}")

    rewards_over_time = []
    elo_over_time = {"sber": [], "yandex": []}
    frames = []

    for episode in range(last_episode + 1, NUM_EPISODES + 1):
        env.reset()
        for _ in range(100):  # Число шагов на эпизод
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            env.step(actions)
            frames.append(env.render())
        
        env.update_elo()  # Обновляем ELO по результатам эпизода
        rewards_over_time.append(sum(env.rewards.values()))
        elo_over_time["sber"].append(env.elo_scores["sber"])
        elo_over_time["yandex"].append(env.elo_scores["yandex"])

        if episode % 10 == 0:
            env.save_training_state(episode)
            print(f"Сохранено состояние на эпизоде {episode}")

    with imageio.get_writer(VIDEO_PATH, fps=10) as writer:
        for frame in frames:
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(rewards_over_time, label="Total Rewards per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Rewards Over Time")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(elo_over_time["sber"], label="ELO Sber")
    plt.plot(elo_over_time["yandex"], label="ELO Yandex")
    plt.xlabel("Episode")
    plt.ylabel("ELO Score")
    plt.title("ELO Scores Over Time")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
