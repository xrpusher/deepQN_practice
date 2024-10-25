import time
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
import imageio
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure

# Кастомный callback для логирования оставшегося времени
class TimeLoggingCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=1):
        super(TimeLoggingCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.start_time = None

    def _on_training_start(self):
        self.start_time = time.time()
        print("Обучение началось...")

    def _on_step(self):
        elapsed_time = time.time() - self.start_time
        completed_timesteps = self.num_timesteps
        remaining_timesteps = self.total_timesteps - completed_timesteps
        fps = completed_timesteps / elapsed_time
        remaining_time = remaining_timesteps / fps

        if self.verbose > 0:
            print(f"Выполнено шагов: {completed_timesteps}/{self.total_timesteps}, "
                  f"Оставшееся время: {remaining_time:.2f} секунд, "
                  f"FPS: {fps:.2f}")

        return True

# Создаем обернутую среду для Riverraid
def create_env():
    env = gym.make("ALE/Riverraid-v5", render_mode="rgb_array")
    env = GrayScaleObservation(env)  # Преобразуем в черно-белый
    env = ResizeObservation(env, 84)  # Изменяем размер на 84x84
    env = FrameStack(env, num_stack=4)  # Стек 4-х кадров
    return env

# Функция для записи видео
def record_video(env, model, out_path, fps=30):
    frames = []
    obs = env.reset()
    done = False
    while not done:
        frame = env.render()  # Получаем кадр
        frames.append(frame)  # Сохраняем кадр
        
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)

    # Сохраняем видео
    imageio.mimsave(out_path, frames, fps=fps)
    print(f"Видео сохранено по пути: {out_path}")

# Векторизованная среда с 4 окружениями
vec_env = make_vec_env(create_env, n_envs=4)
vec_env = VecFrameStack(vec_env, n_stack=4)  # Объединяем 4 кадра в один по каналу

# Логирование с использованием TensorBoard
new_logger = configure('./ppo_riverraid_tensorboard/', ["stdout", "tensorboard"])

# Модель PPO с использованием CnnPolicy
model = PPO("CnnPolicy", vec_env, verbose=1, tensorboard_log="./ppo_riverraid_tensorboard/", device="cuda")
model.set_logger(new_logger)

# Callback для сохранения контрольных точек модели и логирования
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./checkpoints/', name_prefix='ppo_riverraid')
eval_callback = EvalCallback(vec_env, best_model_save_path='./logs/best_model',
                             log_path='./logs/', eval_freq=5000, deterministic=True, render=False)

# Кастомный callback для логирования оставшегося времени
time_logging_callback = TimeLoggingCallback(total_timesteps=1000000)

# Обучение модели с логированием времени и сохранением контрольных точек
model.learn(total_timesteps=1000000, callback=[checkpoint_callback, eval_callback, time_logging_callback])

# Сохранение модели после обучения
model.save("ppo_riverraid")

# Запуск обученной модели для записи видео
env = create_env()  # Создаем новую среду для записи видео
output_video_path = "riverraid_ppo_play.mp4"
record_video(env, model, output_video_path)
