import os
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
import imageio

# Функция для создания обернутой среды
def create_env():
    env = gym.make("ALE/Riverraid-v5", render_mode="rgb_array")
    env = GrayScaleObservation(env)  # Преобразуем в черно-белый
    env = ResizeObservation(env, 84)  # Изменяем размер на 84x84
    env = FrameStack(env, num_stack=4)  # Стек 4-х кадров
    return env

# Создаем векторизованную среду с 4 окружениями
env = DummyVecEnv([create_env for _ in range(4)])
env = VecFrameStack(env, n_stack=4)  # Объединяем кадры по каналам

# Загружаем обученную модель
model_path = 'ppo_riverraid_final.zip'  # Убедитесь, что путь к модели корректен
model = PPO.load(model_path, env=env, device='cuda')

# Функция для записи видео
def record_video(env, model, out_path, fps=30):
    frames = []
    obs = env.reset()
    dones = [False] * env.num_envs
    while not all(dones):
        # Получаем кадр из первого окружения
        frame = env.envs[0].render()
        frames.append(frame)
        # Получаем действие от модели
        action, _ = model.predict(obs)
        # Шаг в среде
        obs, rewards, dones, infos = env.step(action)
    # Сохраняем видео
    imageio.mimsave(out_path, frames, fps=fps)
    print(f"Видео сохранено по пути: {out_path}")

# Путь для сохранения видео
output_video_path = 'riverraid_ppo_play.mp4'

# Запускаем запись видео
record_video(env, model, output_video_path)
