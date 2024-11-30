import os
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
import imageio

# Функция для создания обёрнутой среды
def create_env():
    env = gym.make("ALE/Riverraid-v5", render_mode="rgb_array")
    env = GrayScaleObservation(env)  # Преобразуем в черно-белый
    env = ResizeObservation(env, 84)  # Изменяем размер на 84x84
    env = FrameStack(env, num_stack=4)  # Стек 4-х кадров
    return env

# Создаём векторизованную среду
env = DummyVecEnv([create_env])
env = VecFrameStack(env, n_stack=4)  # Объединяем кадры по каналам

# Загружаем модель и веса
model = PPO("CnnPolicy", env, verbose=1, device="cuda")
weights_path = "ppo_riverraid_weights.pth"
model.policy.load_state_dict(torch.load(weights_path, map_location="cuda" if torch.cuda.is_available() else "cpu"))
print(f"Веса модели загружены из {weights_path}.")

# Быстрое тестовое "обучение" (по сути, тренировочный цикл для проверки)
# Обучение на нескольких шагах для подтверждения работоспособности
model.learn(total_timesteps=1000, reset_num_timesteps=False)
print("Быстрое обучение завершено.")

# Функция для записи видео
def record_video(env, model, out_path, fps=30):
    frames = []
    obs = env.reset()
    done = False
    while not done:
        # Получаем кадр
        frame = env.render()
        frames.append(frame)

        # Получаем действие от модели
        action, _ = model.predict(obs)
        # Делаем шаг в среде
        obs, rewards, dones, infos = env.step(action)
        done = dones

    # Сохраняем видео
    imageio.mimsave(out_path, frames, fps=fps)
    print(f"Видео сохранено по пути: {out_path}")

# Путь для сохранения видео
output_video_path = "riverraid_ppo_play.mp4"

# Запуск модели и запись видео
record_video(create_env(), model, output_video_path)
