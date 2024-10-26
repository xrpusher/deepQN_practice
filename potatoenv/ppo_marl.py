import numpy as np
import imageio
import cv2
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from gymnasium.spaces import Discrete, Box

# Environment parameters
GRID_SIZE = 5
SCALE_FACTOR = 32  # Масштабирование ячейки для визуализации
NUM_EPISODES = 500  # Количество эпизодов
INITIAL_CHARGE = 100  # Начальный уровень заряда для каждого трактора
FRAME_SKIP = 5  # Сохранение каждого 5-го кадра

class PotatoFieldEnv(AECEnv):
    metadata = {"render_modes": ["rgb_array"], "is_parallelizable": True}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.grid = np.ones((GRID_SIZE, GRID_SIZE))  # 1 - картошка, 0 - пусто
        self.possible_agents = ["tractor_red", "tractor_blue"]
        self.agents = self.possible_agents[:]
        self.positions = {"tractor_red": (0, 0), "tractor_blue": (GRID_SIZE - 1, GRID_SIZE - 1)}
        self.rewards = {agent: 0 for agent in self.agents}
        self.charges = {agent: INITIAL_CHARGE for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.central_buffer = {}  # Централизованный буфер для координации действий агентов
        self.max_rewards = {agent: float('-inf') for agent in self.possible_agents}  # Отслеживание максимальных наград

    def observation_space(self, agent):
        return Box(low=0, high=1, shape=(GRID_SIZE, GRID_SIZE), dtype=np.float32)

    def action_space(self, agent):
        return Discrete(6)  # Вверх, вниз, влево, вправо, собрать картошку, ждать

    def reset(self, seed=None, options=None):
        self.grid = np.ones((GRID_SIZE, GRID_SIZE))
        self.positions = {"tractor_red": (0, 0), "tractor_blue": (GRID_SIZE - 1, GRID_SIZE - 1)}
        self.rewards = {agent: 0 for agent in self.agents}
        self.charges = {agent: INITIAL_CHARGE for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.central_buffer = {agent: None for agent in self.agents}  # Сброс буфера
        self.agents = self.possible_agents[:]
        self._agent_selector = agent_selector(self.agents)
        self._selected_agent = self._agent_selector.next()
        observations = {agent: self.observe(agent) for agent in self.agents}
        return observations

    def step(self, actions):
        """Метод step принимает централизованные действия для всех агентов и координирует их."""
        for agent, action in actions.items():
            if self.charges[agent] > 0 and not self.terminations[agent]:
                # Уменьшаем заряд и выполняем действие
                self.charges[agent] -= 1
                self.central_buffer[agent] = self._propose_move(agent, action)

        # Применение действий из буфера с учетом коллизий
        self._apply_actions_from_buffer()
        
        # Проверка завершения эпизода
        if all(charge <= 0 for charge in self.charges.values()) or np.sum(self.grid == 1) == 0:
            for agent in self.agents:
                self.terminations[agent] = True

    def _propose_move(self, agent, action):
        """Предлагает новое положение или действие для агента."""
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
        elif action == 4:  # Собрать картошку
            if self.grid[y][x] == 1:
                self.grid[y][x] = -1 if agent == "tractor_red" else -2
                self.rewards[agent] += 1
            else:
                self.rewards[agent] -= 0.5
        elif action == 5:  # Ожидание
            self.rewards[agent] -= 0.1

        return (new_x, new_y) if action in [0, 1, 2, 3] else self.positions[agent]

    def _apply_actions_from_buffer(self):
        """Применяет действия из буфера с проверкой на коллизии."""
        positions = list(self.central_buffer.values())
        if len(positions) == len(set(positions)):  # Отсутствие коллизий
            for agent, pos in self.central_buffer.items():
                self.positions[agent] = pos
        else:  # Обработка конфликтов
            for agent, pos in self.central_buffer.items():
                if positions.count(pos) == 1:
                    self.positions[agent] = pos
                else:
                    self.rewards[agent] -= 0.1  # Штраф за конфликт

    def observe(self, agent):
        """Возвращает наблюдение для агента с учетом его текущего положения и состояния поля."""
        obs = np.ones((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        for i, pos in enumerate(self.positions.values()):
            obs[pos[1], pos[0]] = -1 if i == 0 else -2
        return obs

    def render(self):
        """Отрисовывает текущее состояние поля и агентов."""
        display_grid = np.ones((GRID_SIZE * SCALE_FACTOR, GRID_SIZE * SCALE_FACTOR + 100, 3), dtype=np.uint8) * 255

        # Отображение картошки и пометок
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if self.grid[y][x] == 1:
                    center = (x * SCALE_FACTOR + SCALE_FACTOR // 2, y * SCALE_FACTOR + SCALE_FACTOR // 2)
                    axes = (SCALE_FACTOR // 4, SCALE_FACTOR // 6)
                    cv2.ellipse(display_grid, center, axes, 0, 0, 360, (0, 255, 255), -1)
                elif self.grid[y][x] == -1:
                    self._draw_cross(display_grid, x, y, (255, 0, 0))
                elif self.grid[y][x] == -2:
                    self._draw_cross(display_grid, x, y, (0, 0, 255))

        # Отображение тракторов
        for agent, pos in self.positions.items():
            color = (255, 0, 0) if agent == "tractor_red" else (0, 0, 255)
            cv2.rectangle(display_grid,
                          (pos[0] * SCALE_FACTOR, pos[1] * SCALE_FACTOR),
                          ((pos[0] + 1) * SCALE_FACTOR, (pos[1] + 1) * SCALE_FACTOR),
                          color, -1)

        # Отображение награды и заряда
        for i, agent in enumerate(self.possible_agents):
            reward_text = f"{agent}: {self.rewards[agent]:.1f} | Charge: {self.charges[agent]}"
            cv2.putText(display_grid, reward_text, (10, 20 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        return display_grid

    def _draw_cross(self, grid, x, y, color):
        """Рисует крест на позиции (x, y) с заданным цветом."""
        cv2.line(grid,
                 (x * SCALE_FACTOR + SCALE_FACTOR // 4, y * SCALE_FACTOR + SCALE_FACTOR // 4),
                 (x * SCALE_FACTOR + 3 * SCALE_FACTOR // 4, y * SCALE_FACTOR + 3 * SCALE_FACTOR // 4),
                 color, 2)
        cv2.line(grid,
                 (x * SCALE_FACTOR + SCALE_FACTOR // 4, y * SCALE_FACTOR + 3 * SCALE_FACTOR // 4),
                 (x * SCALE_FACTOR + 3 * SCALE_FACTOR // 4, y * SCALE_FACTOR + SCALE_FACTOR // 4),
                 color, 2)

    def close(self):
        pass

# Простая реализация нейронной сети с ручным вычислением градиентов
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=1e-4):
        # Инициализация весов небольшими случайными значениями
        self.W1 = np.random.randn(hidden_size, input_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(output_size, hidden_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros((output_size, 1))
        self.learning_rate = learning_rate

    def forward(self, x):
        """
        Прямой проход через сеть.
        x: входная матрица (input_size x batch_size)
        Возвращает: выходные активации и промежуточные переменные для обратного прохода
        """
        z1 = np.dot(self.W1, x) + self.b1  # (hidden_size, batch_size)
        a1 = self.relu(z1)                 # (hidden_size, batch_size)
        z2 = np.dot(self.W2, a1) + self.b2 # (output_size, batch_size)
        a2 = z2                            # (output_size, batch_size)
        cache = (x, z1, a1, z2, a2)
        return a2, cache

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def backward(self, dout, cache):
        """
        Обратный проход через сеть.
        dout: градиент функции потерь по выходу (output_size x batch_size)
        cache: промежуточные переменные из прямого прохода
        Возвращает: градиенты по W1, b1, W2, b2
        """
        x, z1, a1, z2, a2 = cache
        # Градиент по W2 и b2
        dW2 = np.dot(dout, a1.T)  # (output_size, hidden_size)
        db2 = np.sum(dout, axis=1, keepdims=True)  # (output_size, 1)

        # Градиент по a1
        da1 = np.dot(self.W2.T, dout)  # (hidden_size, batch_size)

        # Градиент по z1
        dz1 = da1 * self.relu_derivative(z1)  # (hidden_size, batch_size)

        # Градиент по W1 и b1
        dW1 = np.dot(dz1, x.T)  # (hidden_size, input_size)
        db1 = np.sum(dz1, axis=1, keepdims=True)  # (hidden_size, 1)

        return dW1, db1, dW2, db2

    def update(self, dW1, db1, dW2, db2):
        """
        Обновляет параметры сети с использованием вычисленных градиентов.
        """
        # Ограничение градиентов для предотвращения взрыва градиентов
        self.W1 += self.learning_rate * np.clip(dW1, -1, 1)
        self.b1 += self.learning_rate * np.clip(db1, -1, 1)
        self.W2 += self.learning_rate * np.clip(dW2, -1, 1)
        self.b2 += self.learning_rate * np.clip(db2, -1, 1)

# Реализация PPO-агента
class PPOAgent:
    def __init__(self, obs_size, action_size, hidden_size=64, lr=1e-4, gamma=0.99, epsilon=0.2, lam=0.95):
        self.policy_network = NeuralNetwork(obs_size, hidden_size, action_size, learning_rate=lr)
        self.value_network = NeuralNetwork(obs_size, hidden_size, 1, learning_rate=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.lam = lam

    def softmax(self, logits):
        e = np.exp(logits - np.max(logits, axis=0, keepdims=True))
        return e / e.sum(axis=0, keepdims=True)

    def select_action(self, state):
        """
        Выбирает действие на основе текущей политики.
        state: наблюдение как одномерный массив numpy
        Возвращает: действие (int), log_prob (float), value (float), cache_policy, cache_value
        """
        state = state.reshape(-1, 1)  # Преобразование в столбцовый вектор (obs_size, 1)
        logits, cache_policy = self.policy_network.forward(state)  # (action_size, 1)
        probs = self.softmax(logits)  # (action_size, 1)
        
        # Проверка на NaN или Inf в вероятностях
        if np.isnan(probs).any() or np.isinf(probs).any():
            print("Обнаружены NaN или Inf в вероятностях политики. Сброс вероятностей.")
            probs = np.ones_like(probs) / probs.size  # Сброс к равномерным вероятностям
        
        probs = np.clip(probs, 1e-8, 1.0)  # Избежание вероятностей ровно 0
        probs /= probs.sum(axis=0, keepdims=True)  # Нормализация

        action = np.random.choice(len(probs), p=probs.ravel())
        log_prob = np.log(probs[action, 0] + 1e-8)  # Добавление малого значения для предотвращения log(0)

        value, cache_value = self.value_network.forward(state)  # (1, 1)
        value = value[0, 0]  # Извлечение скалярного значения
        return action, log_prob, value, cache_policy, cache_value

    def compute_advantages(self, rewards, dones, values, next_value):
        """
        Вычисляет Generalized Advantage Estimation (GAE)
        """
        advantages = []
        gae = 0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * (next_value if not dones[step] else 0) - values[step]
            gae = delta + self.gamma * self.lam * gae
            advantages.insert(0, gae)
            next_value = values[step]
        return advantages

    def update_policy(self, trajectories, epochs=10, batch_size=32):
        """
        Обновляет политики и сети значений с использованием целевой функции PPO.
        trajectories: список данных траекторий
        """
        # Развертка траекторий
        states = np.vstack([t['state'] for t in trajectories])  # (num_samples, obs_size)
        actions = np.array([t['action'] for t in trajectories])  # (num_samples,)
        old_log_probs = np.array([t['log_prob'] for t in trajectories])  # (num_samples,)
        advantages = np.array([t['advantage'] for t in trajectories])  # (num_samples,)
        returns = np.array([t['return'] for t in trajectories])  # (num_samples,)

        # Нормализация преимуществ
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        num_samples = states.shape[0]
        for _ in range(epochs):
            # Перемешивание данных
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                mb_idx = indices[start:end]
                mb_states = states[mb_idx].T  # (obs_size, batch_size)
                mb_actions = actions[mb_idx]  # (batch_size,)
                mb_old_log_probs = old_log_probs[mb_idx]  # (batch_size,)
                mb_advantages = advantages[mb_idx]  # (batch_size,)
                mb_returns = returns[mb_idx]  # (batch_size,)

                batch_size_actual = mb_states.shape[1]

                # Прямой проход для политики
                logits, cache_policy = self.policy_network.forward(mb_states)  # (action_size, batch_size)
                probs = self.softmax(logits)  # (action_size, batch_size)

                # Проверка на NaN или Inf в вероятностях
                if np.isnan(probs).any() or np.isinf(probs).any():
                    print("Обнаружены NaN или Inf в вероятностях политики во время обновления. Ограничение вероятностей.")
                    probs = np.clip(probs, 1e-8, 1.0)
                    probs /= probs.sum(axis=0, keepdims=True)  # Нормализация

                selected_probs = probs[mb_actions, np.arange(batch_size_actual)]  # (batch_size,)
                selected_probs = np.clip(selected_probs, 1e-8, 1.0)  # Предотвращение log(0)
                log_probs = np.log(selected_probs)  # (batch_size,)

                # Отношение для обрезки PPO
                ratios = np.exp(log_probs - mb_old_log_probs)  # (batch_size,)

                # Потери PPO
                surr1 = ratios * mb_advantages  # (batch_size,)
                surr2 = np.clip(ratios, 1 - self.epsilon, 1 + self.epsilon) * mb_advantages  # (batch_size,)
                policy_loss = -np.mean(np.minimum(surr1, surr2))

                # Вычисление градиентов для сети политики
                # Градиент потерь по отношениям
                dL_dratios = -np.minimum(ratios, np.clip(ratios, 1 - self.epsilon, 1 + self.epsilon)) * mb_advantages  # (batch_size,)
                # Градиент отношений по log_prob
                dratios_dlogp = ratios  # d(ratio)/d(log_prob) = ratio
                # Правило цепочки
                dL_dlogp = dL_dratios * dratios_dlogp  # (batch_size,)

                # Инициализация градиента для logits
                dout_policy = np.zeros_like(probs)  # (action_size, batch_size)
                dout_policy[mb_actions, np.arange(batch_size_actual)] = dL_dlogp  # (action_size, batch_size)

                # Обратный проход для сети политики
                dW1_p, db1_p, dW2_p, db2_p = self.policy_network.backward(dout_policy, cache_policy)
                # Обновление сети политики с усреднёнными градиентами
                self.policy_network.update(dW1_p / batch_size_actual, db1_p / batch_size_actual,
                                          dW2_p / batch_size_actual, db2_p / batch_size_actual)

                # Прямой проход для сети значений
                values, cache_value = self.value_network.forward(mb_states)  # (1, batch_size)
                values = values.flatten()  # (batch_size,)
                value_loss = np.mean((values - mb_returns) ** 2)

                # Градиент потерь по значениям
                dvalue = 2 * (values - mb_returns) / batch_size_actual  # (batch_size,)

                # Инициализация градиента для сети значений
                dout_value = dvalue.reshape(1, -1)  # (1, batch_size)

                # Обратный проход для сети значений
                dW1_v, db1_v, dW2_v, db2_v = self.value_network.backward(dout_value, cache_value)
                # Обновление сети значений с усреднёнными градиентами
                self.value_network.update(dW1_v / batch_size_actual, db1_v / batch_size_actual,
                                         dW2_v / batch_size_actual, db2_v / batch_size_actual)

# Обучение Multi-Agent PPO
def train_multi_agent_ppo():
    env = PotatoFieldEnv(render_mode="rgb_array")
    obs_size = GRID_SIZE * GRID_SIZE
    action_size = 6  # Количество дискретных действий
    agents = {
        "tractor_red": PPOAgent(obs_size, action_size),
        "tractor_blue": PPOAgent(obs_size, action_size)
    }

    frames = []

    for episode in range(NUM_EPISODES):
        observations = env.reset()
        done = {agent: False for agent in env.agents}
        trajectories = {agent: [] for agent in env.agents}

        for step in range(INITIAL_CHARGE):
            actions = {}
            for agent_id in env.agents:
                if not done[agent_id]:
                    obs = observations[agent_id].flatten()
                    agent = agents[agent_id]
                    action, log_prob, value, cache_policy, cache_value = agent.select_action(obs)
                    actions[agent_id] = action
                    trajectories[agent_id].append({
                        'state': obs,
                        'action': action,
                        'log_prob': log_prob,
                        'reward': 0,  # Будет обновлено после шага
                        'value': value,
                        'done': False
                    })

            env.step(actions)

            for agent_id in env.agents:
                if not done[agent_id]:
                    reward = env.rewards[agent_id]
                    done_flag = env.terminations[agent_id]
                    trajectories[agent_id][-1]['reward'] = reward
                    trajectories[agent_id][-1]['done'] = done_flag

            observations = {agent: env.observe(agent) for agent in env.agents}

            if step % FRAME_SKIP == 0:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

            if all(env.terminations.values()):
                break

        # После завершения эпизода, обработка траекторий для каждого агента
        for agent_id, agent in agents.items():
            traj = trajectories[agent_id]
            if len(traj) == 0:
                continue
            rewards = [t['reward'] for t in traj]
            dones = [t['done'] for t in traj]
            values = [t['value'] for t in traj]
            next_value = 0 if traj[-1]['done'] else agent.value_network.forward(traj[-1]['state'].reshape(-1,1))[0].item()
            advantages = agent.compute_advantages(rewards, dones, values, next_value)
            returns = advantages + values

            for i in range(len(traj)):
                traj[i]['advantage'] = advantages[i]
                traj[i]['return'] = returns[i]

            # Обновление политик и сетей значений
            agent.update_policy(traj)

            # Обновление максимальных наград
            env.max_rewards[agent_id] = max(env.max_rewards[agent_id], env.rewards[agent_id])

        # Вывод прогресса
        mean_reward = np.mean([env.rewards[agent] for agent in env.agents])
        print(f"Эпизод {episode + 1}/{NUM_EPISODES}, Средняя награда: {mean_reward:.2f}")

        # Опционально: ранняя остановка при слишком высоких или низких наградах
        # if np.isnan(mean_reward) or np.isinf(mean_reward):
        #     print("Обнаружены NaN или Inf в средних наградах. Остановка обучения.")
        #     break

    # Сохранение видео
    try:
        with imageio.get_writer("potato_field_simulation_optimized.mp4", fps=10) as writer:
            for frame in frames:
                # Убедитесь, что кадр в формате RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                writer.append_data(frame_rgb)
        print("Видео сохранено как potato_field_simulation_optimized.mp4")
    except Exception as e:
        print(f"Не удалось сохранить видео: {e}")

    # Вывод максимальных наград для каждого агента за все эпизоды
    print("\nМаксимальные награды за все эпизоды:")
    for agent, reward in env.max_rewards.items():
        print(f"{agent}: {reward:.1f}")

if __name__ == "__main__":
    train_multi_agent_ppo()
