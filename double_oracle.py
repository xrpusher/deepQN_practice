import numpy as np
import imageio
import cv2
from pettingzoo.utils import AECEnv
from gymnasium.spaces import Discrete
import matplotlib.pyplot as plt

# Параметры видео и графика
VIDEO_PATH = "double_oracle_game.mp4"
NUM_EPISODES = 100
FPS = 2
ADAPTATION_INTERVAL = 5  # Частота проверки равновесия

class DoubleOracleGame(AECEnv):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.possible_agents = ["agent_1", "agent_2"]
        self.agents = self.possible_agents[:]
        
        # Действия: 0 - мир, 1 - атака, 2 - отступление
        self.action_space = Discrete(3)  
        self.observation_space = Discrete(3)
        
        # Выплаты
        self.payoffs = {
            (0, 0): (3, 3),
            (0, 1): (0, 5),
            (1, 0): (5, 0),
            (1, 1): (1, 1),
            (2, 0): (2, 1),
            (2, 1): (1, 2),
            (0, 2): (1, 2),
            (1, 2): (4, 0),
            (2, 2): (2, 2)
        }
        
        # Начальные стратегии
        self.strategy_profiles = {"agent_1": [0, 1], "agent_2": [0, 2]}
        self.rewards = {agent: 0 for agent in self.agents}
        
        # Счетчик эпизодов
        self.current_episode = 0

    def reset(self):
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        # Не сбрасываем strategy_profiles, чтобы стратегии накапливались
        # self.strategy_profiles = {"agent_1": [0, 1], "agent_2": [0, 2]}
        return {agent: 0 for agent in self.agents}

    def observe(self, agent):
        return np.random.choice(self.strategy_profiles[agent])

    def step(self, actions):
        # Вычисляем результат текущего шага
        outcome = self.payoffs.get((actions["agent_1"], actions["agent_2"]), (0, 0))
        self.rewards["agent_1"] += outcome[0]
        self.rewards["agent_2"] += outcome[1]

        # Увеличиваем счетчик эпизодов
        self.current_episode += 1

        # Обновляем стратегии через ADAPTATION_INTERVAL эпизодов
        if self.current_episode % ADAPTATION_INTERVAL == 0:
            self.update_strategies()
        
        self.agents = []
        # Можно добавить информацию о завершении эпизода, если требуется

    def update_strategies(self):
        for agent in self.possible_agents:
            other_agent = "agent_2" if agent == "agent_1" else "agent_1"
            current_strategies = self.strategy_profiles[agent]
            other_strategies = self.strategy_profiles[other_agent]
            
            # Находим лучший ответ для текущего агента против всех стратегий другого агента
            best_response = None
            best_payoff = -np.inf
            for action in range(3):
                expected_payoff = 0
                for other_action in other_strategies:
                    payoff = self.payoffs.get((action, other_action), (0, 0))[self.possible_agents.index(agent)]
                    expected_payoff += payoff / len(other_strategies)
                if expected_payoff > best_payoff:
                    best_payoff = expected_payoff
                    best_response = action
            
            # Добавляем лучший ответ, если его еще нет в стратегиях
            if best_response not in current_strategies:
                self.strategy_profiles[agent].append(best_response)

    def render_frame(self, episode, actions, final=False):
        # Генерация изображения текущего состояния
        display_frame = np.ones((300, 400, 3), dtype=np.uint8) * 255
        y_position = 50
        
        # Отображаем текущий эпизод и выбранные действия
        cv2.putText(display_frame, f"Episode {episode}", (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        y_position += 40
        for agent, action in actions.items():
            action_text = "Peace" if action == 0 else "Attack" if action == 1 else "Retreat"
            cv2.putText(display_frame, f"{agent}: {action_text}", (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            y_position += 30
        
        # Итоговые награды и стратегии
        if final:
            for agent, score in self.rewards.items():
                strategies = ", ".join(["Peace" if s == 0 else "Attack" if s == 1 else "Retreat" for s in self.strategy_profiles[agent]])
                cv2.putText(display_frame, f"{agent} Final Score: {score}", (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                y_position += 30
                cv2.putText(display_frame, f"Strategies: {strategies}", (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                y_position += 40

        return display_frame

# Запуск игры и запись видео
env = DoubleOracleGame()
frames = []

# Для хранения наград по эпизодам
episode_rewards = {agent: [] for agent in env.possible_agents}

for episode in range(1, NUM_EPISODES + 1):
    env.reset()
    actions = {agent: env.observe(agent) for agent in env.possible_agents}
    env.step(actions)
    
    # Запись наград
    for agent, reward in env.rewards.items():
        episode_rewards[agent].append(reward)

    # Запись кадров игры
    frame = env.render_frame(episode, actions)
    frames.append(frame)
    frame_final = env.render_frame(episode, actions, final=True)
    frames.append(frame_final)

    # Печать текущих результатов в консоль
    print(f"Эпизод {episode}")
    print("Текущие награды:", env.rewards)
    print("Текущие стратегии:", env.strategy_profiles)
    print("\n")

# Сохранение кадров в видео
with imageio.get_writer(VIDEO_PATH, fps=FPS) as writer:
    for frame in frames:
        writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# Построение графика наград по эпизодам
plt.figure(figsize=(10, 6))
for agent, rewards in episode_rewards.items():
    plt.plot(rewards, label=f"{agent} Rewards")
plt.xlabel("Episodes")
plt.ylabel("Total Rewards")
plt.title("Reward Progression per Episode in Double Oracle Game")
plt.legend()
plt.show()

print(f"Видео сохранено как {VIDEO_PATH}")
