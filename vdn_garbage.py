import numpy as np
import imageio
import cv2
import matplotlib.pyplot as plt
import os
import pickle
from collections import deque, namedtuple
import random


from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from gymnasium.spaces import Discrete, Box


# Параметры среды и обучения
GRID_SIZE = 5
SCALE_FACTOR = 32
INITIAL_CHARGE = 100
NUM_EPISODES = 500
SAVE_PATH = "training_state_vdn251113.pkl"
VIDEO_PATH = "garbage_collection_training_vdn.mp4"
MEMORY_SIZE = 10000
BATCH_SIZE = 32
GAMMA = 0.99
LEARNING_RATE = 1e-3
HIDDEN_DIM = 64


# Определение среды GarbageCollectionEnv
class GarbageCollectionEnv(AECEnv):
    metadata = {"render_modes": ["rgb_array"], "is_parallelizable": True}


    def __init__(self, render_mode=None):
        super().__init__()
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
        for _ in range(GRID_SIZE * 2):  # Увеличено количество мусора
            x, y = np.random.randint(0, GRID_SIZE, size=2)
            self.grid[x, y] = 1  # Метка для мусора


    def step(self, actions):
        for agent, action in actions.items():
            if self.charges[agent] > 0 and not self.terminations[agent]:
                self.charges[agent] -= 1
                self._move(agent, action)


        # Проверка завершения эпизода
        if all(self.terminations.values()) or np.all(self.grid == 0):
            self.episode_rewards.append(sum(self.rewards.values()))
            self.update_elo()
            self.save_training_state(len(self.episode_rewards))
            # Завершение эпизода для всех агентов
            for agent in self.agents:
                self.terminations[agent] = True


    def _move(self, agent, action):
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
        elif action == 4 and self.grid[x, y] == 1:  # Сбор мусора
            self.grid[x, y] = 0
            self.rewards[agent] += 1
        elif action == 5:  # Ожидание
            self.rewards[agent] -= 0.1


        self.positions[agent] = (new_x, new_y)


    def observe(self, agent):
        obs = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        obs[self.positions[agent]] = 1  # Обозначение позиции агента
        return obs


    def render(self):
        """Отрисовка состояния, включающая ELO и награды компаний."""
        display_grid = np.ones((GRID_SIZE * SCALE_FACTOR, GRID_SIZE * SCALE_FACTOR + 200, 3), dtype=np.uint8) * 255
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


        cv2.putText(display_grid, text_sber, (GRID_SIZE * SCALE_FACTOR + 10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(display_grid, text_yandex, (GRID_SIZE * SCALE_FACTOR + 10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


        return display_grid


   
    def update_elo(self):
        """Обновление ELO рейтинга на основе командных результатов."""
        sber_reward = sum(self.rewards[agent] for agent in self.teams["sber"])
        yandex_reward = sum(self.rewards[agent] for agent in self.teams["yandex"])

        ea = 1 / (1 + 10 ** ((self.elo_scores["yandex"] - self.elo_scores["sber"]) / 400))
        eb = 1 - ea
        k = 32  # Коэффициент K для рейтинга ELO

        if sber_reward > yandex_reward:
            sa, sb = 1, 0
        elif sber_reward < yandex_reward:
            sa, sb = 0, 1
        else:
            sa, sb = 0.5, 0.5  # Если награды равны

        old_sber_elo = self.elo_scores["sber"]
        old_yandex_elo = self.elo_scores["yandex"]

        # Обновляем ELO
        self.elo_scores["sber"] += k * (sa - ea)
        self.elo_scores["yandex"] += k * (sb - eb)

        # Вывод обновленных значений ELO для каждой команды
        print(f"Обновление ELO: Sber ({old_sber_elo} -> {self.elo_scores['sber']}), "
            f"Yandex ({old_yandex_elo} -> {self.elo_scores['yandex']})")



        self.elo_scores["sber"] += k * (sa - ea)
        self.elo_scores["yandex"] += k * (sb - eb)


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
        self.episode_rewards = []
        self.elo_scores = {"sber": 1200, "yandex": 1200}
        return 0  # Начать с нуля, если данных нет


# Реализация Нейронной Сети на NumPy
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=1e-3):
        # Инициализация весов и смещений
        self.W1 = np.random.randn(hidden_size, input_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(output_size, hidden_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros((output_size, 1))
        self.learning_rate = learning_rate


    def relu(self, z):
        return np.maximum(0, z)


    def relu_derivative(self, z):
        return (z > 0).astype(float)


    def softmax(self, z):
        e_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return e_z / e_z.sum(axis=0, keepdims=True)


    def forward(self, x):
        """
        Прямой проход.
        x: входные данные (input_size x batch_size)
        """
        self.x = x
        self.z1 = np.dot(self.W1, x) + self.b1  # (hidden_size x batch_size)
        self.a1 = self.relu(self.z1)            # (hidden_size x batch_size)
        self.z2 = np.dot(self.W2, self.a1) + self.b2  # (output_size x batch_size)
        self.a2 = self.softmax(self.z2)         # (output_size x batch_size)
        return self.a2


    def backward(self, grad_a2):
        """
        Обратный проход.
        grad_a2: градиент функции потерь по выходу (output_size x batch_size)
        """
        m = self.x.shape[1]


        # Градиент по W2 и b2
        grad_z2 = grad_a2  # Для softmax с кросс-энтропией
        grad_W2 = np.dot(grad_z2, self.a1.T) / m  # (output_size x hidden_size)
        grad_b2 = np.sum(grad_z2, axis=1, keepdims=True) / m  # (output_size x 1)


        # Градиент по a1
        grad_a1 = np.dot(self.W2.T, grad_z2)  # (hidden_size x batch_size)


        # Градиент по z1
        grad_z1 = grad_a1 * self.relu_derivative(self.z1)  # (hidden_size x batch_size)


        # Градиент по W1 и b1
        grad_W1 = np.dot(grad_z1, self.x.T) / m  # (hidden_size x input_size)
        grad_b1 = np.sum(grad_z1, axis=1, keepdims=True) / m  # (hidden_size x 1)


        # Обновление весов и смещений
        self.W2 -= self.learning_rate * grad_W2
        self.b2 -= self.learning_rate * grad_b2
        self.W1 -= self.learning_rate * grad_W1
        self.b1 -= self.learning_rate * grad_b1


# Replay Buffer
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)


    def push(self, state, actions, rewards, next_state, dones):
        self.buffer.append(Transition(state, actions, rewards, next_state, dones))


    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)


    def __len__(self):
        return len(self.buffer)


# VDN Агент
class VDNAgent:
    def __init__(self, env):
        self.env = env
        self.n_agents = len(env.agents)
        self.state_dim = GRID_SIZE * GRID_SIZE  # Размер состояния
        self.action_dim = 6  # Количество действий


        # Создание Q-сетей для каждого агента
        self.q_networks = {}
        for agent in env.agents:
            self.q_networks[agent] = NeuralNetwork(input_size=self.state_dim, hidden_size=HIDDEN_DIM, output_size=self.action_dim, learning_rate=LEARNING_RATE)


        # Replay Buffer
        self.replay_buffer = ReplayBuffer(MEMORY_SIZE)


    def select_actions(self, observations, epsilon=0.1):
        actions = {}
        for agent in self.env.agents:
            if np.random.rand() < epsilon:
                action = np.random.randint(0, self.action_dim)
            else:
                state = observations[agent].flatten().reshape(-1, 1)
                q_values = self.q_networks[agent].forward(state)
                action = np.argmax(q_values, axis=0)[0]
            actions[agent] = action
        return actions



    def store_transition(self, observations, actions, rewards, next_observations, dones):
        self.replay_buffer.push(observations, actions, rewards, next_observations, dones)


    def update(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return


        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)


        # Перебор каждого агента для обновления его Q-сети
        total_loss = 0  # Для отслеживания общей потери


        for agent in self.env.agents:
            # Подготовка данных для агента
            state_batch = np.array([s[agent].flatten() for s in states]).T  # (state_dim x batch_size)
            action_batch = np.array([a[agent] for a in actions]).reshape(1, BATCH_SIZE)  # (1 x batch_size)
            reward_batch = np.array([r[agent] for r in rewards]).reshape(1, BATCH_SIZE)  # (1 x batch_size)
            next_state_batch = np.array([s[agent].flatten() for s in next_states]).T  # (state_dim x batch_size)
            done_batch = np.array([d[agent] for d in dones]).reshape(1, BATCH_SIZE)  # (1 x batch_size)


            # Получение текущих Q-значений
            q_values = self.q_networks[agent].forward(state_batch)  # (action_dim x batch_size)
            # Получаем Q-значения выбранных действий
            q_selected = q_values[action_batch.flatten(), np.arange(BATCH_SIZE)]  # (batch_size,)


            # Получение максимальных Q-значений для next_states
            q_next = self.q_networks[agent].forward(next_state_batch)  # (action_dim x batch_size)
            q_next_max = np.max(q_next, axis=0)  # (batch_size,)


            # Целевая Q-функция
            target_q = reward_batch.flatten() + GAMMA * q_next_max * (1 - done_batch.flatten())  # (batch_size,)


            # Потери (MSE)
            loss = (q_selected.flatten() - target_q) ** 2
            total_loss += np.mean(loss)


            # Градиенты и обновление весов
            grad_output = 2 * (q_selected.flatten() - target_q) / BATCH_SIZE  # (batch_size,)


            # Создание grad_z2_full (action_dim x batch_size)
            grad_z2_full = np.zeros((self.action_dim, BATCH_SIZE))
            grad_z2_full[action_batch.flatten(), np.arange(BATCH_SIZE)] = grad_output  # (action_dim x batch_size)


            # Обратный проход
            self.q_networks[agent].backward(grad_z2_full)


        # Вывод средней потери за обновление
        print(f"Средняя потеря за обновление: {total_loss / self.n_agents:.4f}")



# Убедитесь, что только одна функция main() определена

def main():
    env = GarbageCollectionEnv()
    last_episode = env.load_training_state()
    print(f"Продолжение обучения с эпизода {last_episode + 1}")

    agent = VDNAgent(env)

    frames = []
    rewards_over_time = []
    elo_over_time = {"sber": [], "yandex": []}

    # Переменные для отслеживания максимальных значений ELO и Reward
    max_elo_sber = env.elo_scores["sber"]
    max_elo_yandex = env.elo_scores["yandex"]
    max_reward_sber = 0
    max_reward_yandex = 0

    for episode in range(last_episode + 1, NUM_EPISODES + 1):
        observations = env.reset()
        episode_reward = 0
        done = {agent_id: False for agent_id in env.agents}

        for step in range(INITIAL_CHARGE):
            actions = agent.select_actions(observations)
            env.step(actions)
            next_observations = {agent_id: env.observe(agent_id) for agent_id in env.agents}
            rewards = {agent_id: env.rewards[agent_id] for agent_id in env.agents}
            dones = {agent_id: env.terminations[agent_id] for agent_id in env.agents}

            agent.store_transition(observations, actions, rewards, next_observations, dones)
            agent.update()

            # Обновляем ELO после каждого шага
            env.update_elo()

            observations = next_observations
            episode_reward += sum(rewards.values())

            if all(dones.values()):
                break

        # Добавляем расчет командных наград за эпизод для вывода
        reward_sber = sum(env.rewards[agent] for agent in env.teams["sber"])
        reward_yandex = sum(env.rewards[agent] for agent in env.teams["yandex"])

        print(f"Эпизод {episode}: Награда Sber={reward_sber}, Yandex={reward_yandex}")
        print(f"ELO Sber={env.elo_scores['sber']:.2f}, ELO Yandex={env.elo_scores['yandex']:.2f}\n")

        # Обновляем максимальные значения ELO и Reward
        max_elo_sber = max(max_elo_sber, env.elo_scores["sber"])
        max_elo_yandex = max(max_elo_yandex, env.elo_scores["yandex"])
        max_reward_sber = max(max_reward_sber, reward_sber)
        max_reward_yandex = max(max_reward_yandex, reward_yandex)

        rewards_over_time.append(episode_reward)
        elo_over_time["sber"].append(env.elo_scores["sber"])
        elo_over_time["yandex"].append(env.elo_scores["yandex"])

        # Сохранение состояния каждые 100 эпизодов
        if episode % 100 == 0:
            env.save_training_state(episode)
            print(f"Сохранено состояние на эпизоде {episode}")

        # Вывод прогресса каждые 10 эпизодов
        if episode % 10 == 0:
            mean_reward = np.mean(rewards_over_time[-10:])
            mean_elo_sber = np.mean(elo_over_time["sber"][-10:])
            mean_elo_yandex = np.mean(elo_over_time["yandex"][-10:])
            print(f"Эпизод {episode}/{NUM_EPISODES}, Средняя награда: {mean_reward:.2f}, "
                  f"ELO Sber: {mean_elo_sber:.1f}, ELO Yandex: {mean_elo_yandex:.1f}")

    # Вывод максимальных значений ELO и Reward за все эпизоды
    print("\nМаксимальные значения за все эпизоды:")
    print(f"Максимальный ELO Sber: {max_elo_sber}")
    print(f"Максимальный ELO Yandex: {max_elo_yandex}")
    print(f"Максимальный Reward Sber: {max_reward_sber}")
    print(f"Максимальный Reward Yandex: {max_reward_yandex}")

    # Сохранение видео
    try:
        with imageio.get_writer(VIDEO_PATH, fps=10) as writer:
            for frame in frames:
                writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        print(f"Видео сохранено как {VIDEO_PATH}")
    except Exception as e:
        print(f"Не удалось сохранить видео: {e}")



# Запуск программы
if __name__ == "__main__":
    main()
