!pip install gymnasium[atari]
!pip install gymnasium[accept-rom-license]
!pip install torch torchvision torchaudio
!pip install imageio
!pip install tqdm

# 2. Импорт необходимых библиотек
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from gymnasium.wrappers import FrameStack
import imageio
from tqdm import tqdm

# 3. Определение обёрток для Atari
class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)
        self._skip = skip
        self._obs_buffer = deque(maxlen=2)

    def reset(self, **kwargs):
        self._obs_buffer.clear()
        obs, info = self.env.reset(**kwargs)
        self._obs_buffer.append(obs)
        return obs, info

    def step(self, action):
        total_reward = 0.0
        done = False
        truncated = False
        info = {}
        for _ in range(self._skip):
            obs, reward, done, truncated, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done or truncated:
                break
        max_frame = np.maximum(self._obs_buffer[0], self._obs_buffer[1])
        return max_frame, total_reward, done or truncated, truncated or done, info

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        super(EpisodicLifeEnv, self).__init__(env)
        self.lives = 0
        self.was_real_done = True

    def reset(self, **kwargs):
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            obs, info = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.was_real_done = done
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            done = True
        self.lives = lives
        return obs, reward, done, truncated, info

class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        super(FireResetEnv, self).__init__(env)
        self.fire_action = 1  # Обычно действие FIRE имеет индекс 1

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        action_meanings = self.env.unwrapped.get_action_meanings()
        if 'FIRE' in action_meanings:
            fire_idx = action_meanings.index('FIRE')
            obs, reward, done, truncated, info = self.env.step(fire_idx)
        return obs, info

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84):
        super(WarpFrame, self).__init__(env)
        from gymnasium.spaces import Box
        self.width = width
        self.height = height
        self.observation_space = Box(low=0, high=255, shape=(self.height, self.width), dtype=np.uint8)

    def observation(self, frame):
        import cv2
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame

class ClipRewardEnv(gym.RewardWrapper):
    def reward(self, reward):
        return np.sign(reward)

def wrap_atari(env):
    env = MaxAndSkipEnv(env, skip=4)                # Пропуск кадров
    env = EpisodicLifeEnv(env)                      # Завершение эпизода при потере жизни
    env = FireResetEnv(env)                         # Действие FIRE при сбросе
    env = WarpFrame(env, width=84, height=84)       # Приведение кадра к 84x84, черно-белый
    env = ClipRewardEnv(env)                        # Ограничение вознаграждения до {-1, 0, 1}
    env = FrameStack(env, 4)                        # Стек из 4 кадров
    return env

# 4. Создание и оборачивание среды
env = gym.make("ALE/ElevatorAction-v5", render_mode="rgb_array")
env = wrap_atari(env)
print("Форма наблюдений:", env.observation_space.shape)
print("Количество действий:", env.action_space.n)

# 5. Определение нейронной сети DQN
class DQN(nn.Module):
    def __init__(self, input_channels, n_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),  # (4,84,84) -> (32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),              # (32,20,20) -> (64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),              # (64,9,9) -> (64,7,7)
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out((input_channels, 84, 84))
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size(0), -1)
        return self.fc(conv_out)

# 6. Реализация Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)  # Ограничение размера буфера

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def __len__(self):
        return len(self.buffer)

# 7. Определение функции для вычисления TD Loss (Double DQN)
def compute_td_loss(batch_size, replay_buffer, model, target_model, gamma, optimizer, device):
    if len(replay_buffer) < batch_size:
        return None

    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    # Преобразуем данные в тензоры и перемещаем на устройство (CPU/GPU)
    state = torch.tensor(state, dtype=torch.float32).to(device)        # (batch, 4,84,84)
    next_state = torch.tensor(next_state, dtype=torch.float32).to(device)
    action = torch.tensor(action, dtype=torch.long).to(device)
    reward = torch.tensor(reward, dtype=torch.float32).to(device)
    done = torch.tensor(done, dtype=torch.float32).to(device)

    # Вычисляем Q-значения текущих состояний
    q_values = model(state).gather(1, action.unsqueeze(-1)).squeeze(-1)

    # Вычисляем целевые Q-значения
    with torch.no_grad():
        next_q_values = target_model(next_state).max(1)[0]
        target_q_values = reward + gamma * next_q_values * (1 - done)

    # Вычисляем потерю (MSE)
    loss = nn.MSELoss()(q_values, target_q_values)

    # Обратное распространение ошибки и обновление весов
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

# 8. Функция обучения агента
def train_dqn(env, model, target_model, replay_buffer, n_episodes, batch_size, gamma, epsilon, epsilon_decay, min_epsilon, sync_freq, optimizer, device):
    episode_rewards = []
    for episode in tqdm(range(1, n_episodes + 1)):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                state_tensor = torch.tensor(np.array([state]), dtype=torch.float32).to(device)  # Без permute
                with torch.no_grad():
                    q_values = model(state_tensor)
                    action = torch.argmax(q_values).item()

            next_state, reward, done, truncated, info = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            loss = compute_td_loss(batch_size, replay_buffer, model, target_model, gamma, optimizer, device)

        # Обновляем целевую модель каждые sync_freq эпизодов
        if episode % sync_freq == 0:
            target_model.load_state_dict(model.state_dict())

        # Декрементируем epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)

        # Печатаем результаты каждые 100 эпизодов
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode}, Average Reward (last 100): {avg_reward:.2f}, Epsilon: {epsilon:.4f}")

    return episode_rewards

# 9. Инициализация модели, целевой модели, оптимизатора и Replay Buffer
# Проверка наличия GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")

# Параметры среды
input_channels = env.observation_space.shape[0]  # 4
n_actions = env.action_space.n  # 18

# Инициализация модели и целевой модели с правильным количеством каналов
model = DQN(input_channels, n_actions).to(device)
target_model = DQN(input_channels, n_actions).to(device)
target_model.load_state_dict(model.state_dict())

# Оптимизатор и Replay Buffer
optimizer = optim.Adam(model.parameters(), lr=0.0001)
replay_buffer = ReplayBuffer(10000)

# Проверка Replay Buffer
for i in range(15000):
    replay_buffer.push(np.zeros((4, 84, 84)), 0, 0.0, np.zeros((4, 84, 84)), False)
print(f"Текущий размер Replay Buffer: {len(replay_buffer)}")  # Ожидаемый вывод: 10000

# 10. Параметры обучения и запуск процесса обучения
# Параметры обучения
n_episodes = 1000          # Общее количество эпизодов
batch_size = 32            # Размер батча
gamma = 0.99               # Коэффициент дисконтирования
epsilon = 1.0              # Начальное значение epsilon для epsilon-greedy
epsilon_decay = 0.995      # Коэффициент декремента epsilon
min_epsilon = 0.01         # Минимальное значение epsilon
sync_freq = 100            # Частота обновления целевой модели

# Запуск процесса обучения
episode_rewards = train_dqn(env, model, target_model, replay_buffer, n_episodes, batch_size, gamma, epsilon, epsilon_decay, min_epsilon, sync_freq, optimizer, device)

# 11. Запись Видео Игры После Обучения
def record_video(env, model, filename, device):
    state, _ = env.reset()
    done = False
    frames = []
    while not done:
        frames.append(env.render())
        state_tensor = torch.tensor(np.array([state]), dtype=torch.float32).to(device)
        with torch.no_grad():
            q_values = model(state_tensor)
            action = torch.argmax(q_values).item()
        next_state, reward, done, truncated, info = env.step(action)
        state = next_state
    imageio.mimsave(filename, frames, fps=30)
    print(f"Видео сохранено как {filename}")

# Запись видео
record_video(env, model, "elevator_action_dqn.mp4", device)

# 12. Сохранение и Загрузка Модели (Дополнительно)
# Сохранение модели
torch.save(model.state_dict(), "dqn_elevator_action.pth")
print("Модель сохранена как dqn_elevator_action.pth")

# Загрузка модели
model = DQN(input_channels, n_actions).to(device)
model.load_state_dict(torch.load("dqn_elevator_action.pth"))
model.eval()
print("Модель загружена из dqn_elevator_action.pth")
