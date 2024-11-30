import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium.wrappers import FrameStack, GrayScaleObservation, ResizeObservation
import imageio

# Определение класса DQN
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(self._calculate_conv_output(input_shape), 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
        
    def _calculate_conv_output(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        x = x / 255.0  # Нормализация пиксельных значений
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Создание среды с теми же обертками
env = gym.make("ALE/Riverraid-v5", render_mode="rgb_array")
env = GrayScaleObservation(env)
env = ResizeObservation(env, 84)
env = FrameStack(env, num_stack=4)

# Получение формы входных данных и количества действий
input_shape = env.observation_space.shape  # Должно быть (4, 84, 84)
num_actions = env.action_space.n

# Инициализация сети и загрузка сохраненных весов
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN(input_shape, num_actions).to(device)
policy_net.load_state_dict(torch.load('dqn_riverraid_final.pth', map_location=device))
policy_net.eval()
print("Модель загружена из файла dqn_riverraid.pth")

# Функция для выбора действия с использованием обученной модели
def select_action(state):
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = policy_net(state)
    return q_values.max(1)[1].item()

# Функция для записи видео
def record_video(env, out_path, fps=30):
    frames = []
    state, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        frame = env.render()
        frames.append(frame)
        action = select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        total_reward += reward
    # Сохранение видео
    imageio.mimsave(out_path, frames, fps=fps)
    print(f"Видео сохранено в {out_path}. Общая награда: {total_reward}")

# Запись видео с использованием обученной модели
record_video(env, 'dqn_riverraid_play.mp4')
