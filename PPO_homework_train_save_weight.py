import os
import time
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
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
        # Проверяем, завершился ли эпизод
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

# Создаем обернутую среду для Riverraid
def create_env():
    env = gym.make("ALE/Riverraid-v5", render_mode="rgb_array")
    env = GrayScaleObservation(env)  # Преобразуем в черно-белый
    env = ResizeObservation(env, 84)  # Изменяем размер на 84x84
    env = FrameStack(env, num_stack=4)  # Стек 4-х кадров
    return env

# Функция для загрузки или создания новой модели
def load_or_create_model(checkpoint_dir, env, logger):
    # Проверяем, есть ли чекпоинты для возобновления
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".zip")]
    if checkpoint_files:
        latest_checkpoint = os.path.join(checkpoint_dir, max(checkpoint_files, key=os.path.getctime))
        print(f"Загрузка последнего чекпоинта: {latest_checkpoint}")
        model = PPO.load(latest_checkpoint, env=env, device="cuda")
    else:
        print("Создание новой модели.")
        model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./ppo_riverraid_tensorboard/", device="cuda")
    model.set_logger(logger)
    return model

# Функция для записи видео
def record_video(env, model, out_path, fps=30):
    frames = []
    obs, _ = env.reset()
    done = False
    while not done:
        frame = env.render()  # Получаем кадр
        frames.append(frame)  # Сохраняем кадр

        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    # Сохраняем видео
    imageio.mimsave(out_path, frames, fps=fps)
    print(f"Видео сохранено по пути: {out_path}")

# Векторизованная среда с 4 окружениями
vec_env = make_vec_env(create_env, n_envs=4)
vec_env = VecFrameStack(vec_env, n_stack=4)  # Объединяем 4 кадра в один по каналу

# Логирование с использованием TensorBoard
new_logger = configure('./ppo_riverraid_tensorboard/', ["stdout", "tensorboard"])

# Параметры сохранения чекпоинтов
checkpoint_dir = './checkpoints/'
os.makedirs(checkpoint_dir, exist_ok=True)

# Callback для сохранения контрольных точек модели
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=checkpoint_dir, name_prefix='ppo_riverraid')

# Кастомный callback для логирования эпизодов
episode_logging_callback = EpisodeLoggingCallback(total_timesteps=1000000)

# Загружаем последнюю модель или создаем новую
model = load_or_create_model(checkpoint_dir, vec_env, new_logger)

# Определяем, сколько шагов уже было выполнено
if model.num_timesteps > 0:
    timesteps_already_done = model.num_timesteps
else:
    timesteps_already_done = 0

# Обучение модели с поддержкой паузы и сохранением чекпоинтов
model.learn(total_timesteps=1000000, callback=[checkpoint_callback, episode_logging_callback],
            reset_num_timesteps=False)

# Сохранение модели после завершения обучения
model.save("ppo_riverraid_final")
print("Обучение завершено и модель сохранена.")

# Запуск обученной модели для записи видео
env = create_env()  # Создаем новую среду для записи видео
output_video_path = "riverraid_ppo_play.mp4"
record_video(env, model, output_video_path)
