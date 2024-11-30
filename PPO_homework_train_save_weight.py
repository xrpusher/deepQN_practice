import os
import time
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
import torch
import imageio

# Кастомный callback для логирования эпизодов и вычисления оставшегося времени
class EpisodeLoggingCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=1):
        super(EpisodeLoggingCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.start_time = None
        self.num_episodes = 0

    def _on_training_start(self):
        self.start_time = time.time()
        print("Обучение началось...")

    def _on_step(self):
        for info in self.locals['infos']:
            maybe_ep_info = info.get('episode')
            if maybe_ep_info is not None:
                self.num_episodes += 1
                reward = maybe_ep_info['r']
                length = maybe_ep_info['l']

                elapsed_time = time.time() - self.start_time
                timesteps_done = self.num_timesteps
                timesteps_remaining = self.total_timesteps - timesteps_done
                time_per_timestep = elapsed_time / timesteps_done if timesteps_done > 0 else 0
                eta = timesteps_remaining * time_per_timestep

                print(f"Эпизод {self.num_episodes}: вознаграждение = {reward}, длина = {length}")
                print(f"Выполнено {timesteps_done}/{self.total_timesteps} шагов. "
                      f"Осталось примерно {eta/60:.2f} минут.")
        return True

# Создание обёрнутой среды
def create_env():
    env = gym.make("ALE/Riverraid-v5", render_mode="rgb_array")
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, 84)
    env = FrameStack(env, num_stack=4)
    return env

# Функция для записи видео
def record_video(env, model, out_path, fps=30):
    frames = []
    obs, _ = env.reset()
    done = False
    while not done:
        frame = env.render()
        frames.append(frame)

        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    imageio.mimsave(out_path, frames, fps=fps)
    print(f"Видео сохранено по пути: {out_path}")

# Векторизованная среда с 4 окружениями
vec_env = make_vec_env(create_env, n_envs=4)
vec_env = VecFrameStack(vec_env, n_stack=4)

# Логирование с использованием TensorBoard
new_logger = configure('./ppo_riverraid_tensorboard/', ["stdout", "tensorboard"])

# Параметры сохранения весов
weights_path = "ppo_riverraid_weights.pth"

# Callback для сохранения контрольных точек модели
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./checkpoints', name_prefix='ppo_riverraid')

# Кастомный callback для логирования эпизодов
episode_logging_callback = EpisodeLoggingCallback(total_timesteps=1000000)

# Создание модели
model = PPO("CnnPolicy", vec_env, verbose=1, tensorboard_log="./ppo_riverraid_tensorboard/", device="cuda")

# Обучение модели
model.learn(total_timesteps=1000000, callback=[checkpoint_callback, episode_logging_callback], reset_num_timesteps=False)

# Сохранение только весов
torch.save(model.policy.state_dict(), weights_path)
print(f"Веса модели сохранены в {weights_path}.")

# Создание новой среды для записи видео
env = create_env()

# Загрузка весов в модель
model.policy.load_state_dict(torch.load(weights_path, map_location="cuda" if torch.cuda.is_available() else "cpu"))
print(f"Веса модели загружены из {weights_path}.")

# Запись видео
output_video_path = "riverraid_ppo_play.mp4"
record_video(env, model, output_video_path)
