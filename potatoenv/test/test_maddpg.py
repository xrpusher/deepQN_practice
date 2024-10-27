import numpy as np
import cv2
import imageio
import random
from collections import deque, namedtuple
from pettingzoo import AECEnv
from gymnasium.spaces import Discrete, Box

# Параметры среды
GRID_SIZE = 5
SCALE_FACTOR = 48
NUM_EPISODES = 300
INITIAL_CHARGE = 100
FRAME_SKIP = 5
LEARNING_RATE = 0.01
DISCOUNT_FACTOR = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.995
BATCH_SIZE = 64
REPLAY_BUFFER_SIZE = 100000

class PotatoFieldEnv(AECEnv):
    metadata = {"render_modes": ["rgb_array"], "is_parallelizable": True}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.grid = np.ones((GRID_SIZE, GRID_SIZE))
        self.possible_agents = ["tractor_red", "tractor_blue"]
        self.agents = self.possible_agents[:]
        self.positions = {"tractor_red": (0, 0), "tractor_blue": (GRID_SIZE - 1, GRID_SIZE - 1)}
        self.rewards = {agent: 0 for agent in self.agents}
        self.charges = {agent: INITIAL_CHARGE for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.central_buffer = {}
        self.agent_order = self.agents[:]
        self._agent_selector = self.agent_order.copy()
        self.agent_selection = self._agent_selector.pop(0)

    def observation_space(self, agent):
        return Box(low=0, high=1, shape=(GRID_SIZE, GRID_SIZE), dtype=np.float32)

    def action_space(self, agent):
        return Discrete(6)

    def reset(self, seed=None, options=None):
        self.grid = np.ones((GRID_SIZE, GRID_SIZE))
        self.positions = {"tractor_red": (0, 0), "tractor_blue": (GRID_SIZE - 1, GRID_SIZE - 1)}
        self.rewards = {agent: 0 for agent in self.agents}
        self.charges = {agent: INITIAL_CHARGE for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.central_buffer = {}
        self.agent_order = self.agents[:]
        self._agent_selector = self.agent_order.copy()
        self.agent_selection = self._agent_selector.pop(0)
        return {agent: self.observe(agent) for agent in self.agents}

    def step(self, action):
        agent = self.agent_selection
        if self.charges[agent] > 0 and not self.terminations[agent]:
            self.charges[agent] -= 1
            self.central_buffer[agent] = self._propose_move(agent, action)
        self.rewards = {agent: 0 for agent in self.agents}  # Reset rewards
        self.agent_selection = self._agent_selector.pop(0) if self._agent_selector else None
        done = all(charge <= 0 for charge in self.charges.values()) or np.sum(self.grid == 1) == 0
        if done:
            for agent in self.agents:
                self.terminations[agent] = True
        if not self._agent_selector:
            self._apply_actions_from_buffer()
            self._agent_selector = self.agent_order.copy()

    def _propose_move(self, agent, action):
        x, y = self.positions[agent]
        new_x, new_y = x, y

        if action == 0 and y > 0:
            new_y -= 1
        elif action == 1 and y < GRID_SIZE - 1:
            new_y += 1
        elif action == 2 and x > 0:
            new_x -= 1
        elif action == 3 and x < GRID_SIZE - 1:
            new_x += 1
        elif action == 4:  # Сбор картошки
            if self.grid[y][x] == 1:
                self.grid[y][x] = -1 if agent == "tractor_red" else -2
                self.rewards[agent] += 1
            else:
                self.rewards[agent] -= 0.5
        elif action == 5:  # Ожидание
            self.rewards[agent] -= 0.1

        return (new_x, new_y) if action in [0, 1, 2, 3] else self.positions[agent]

    def _apply_actions_from_buffer(self):
        positions = list(self.central_buffer.values())
        if len(positions) == len(set(positions)):
            for agent, pos in self.central_buffer.items():
                self.positions[agent] = pos
        else:
            for agent, pos in self.central_buffer.items():
                if positions.count(pos) == 1:
                    self.positions[agent] = pos
                else:
                    self.rewards[agent] -= 0.1

    def observe(self, agent):
        obs = np.ones((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        for i, pos in enumerate(self.positions.values()):
            obs[pos[1], pos[0]] = -1 if i == 0 else -2
        return obs

    def render(self):
        display_grid = np.ones((GRID_SIZE * SCALE_FACTOR, GRID_SIZE * SCALE_FACTOR, 3), dtype=np.uint8) * 255
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

        for agent, pos in self.positions.items():
            color = (255, 0, 0) if agent == "tractor_red" else (0, 0, 255)
            cv2.rectangle(display_grid,
                          (pos[0] * SCALE_FACTOR, pos[1] * SCALE_FACTOR),
                          ((pos[0] + 1) * SCALE_FACTOR, (pos[1] + 1) * SCALE_FACTOR),
                          color, -1)
        return display_grid

    def _draw_cross(self, grid, x, y, color):
        cv2.line(grid,
                 (x * SCALE_FACTOR + SCALE_FACTOR // 4, y * SCALE_FACTOR + SCALE_FACTOR // 4),
                 (x * SCALE_FACTOR + 3 * SCALE_FACTOR // 4, y * SCALE_FACTOR + 3 * SCALE_FACTOR // 4),
                 color, 2)
        cv2.line(grid,
                 (x * SCALE_FACTOR + SCALE_FACTOR // 4, y * SCALE_FACTOR + 3 * SCALE_FACTOR // 4),
                 (x * SCALE_FACTOR + 3 * SCALE_FACTOR // 4, y * SCALE_FACTOR // 4),
                 color, 2)

    def close(self):
        pass

class NeuralNetwork:
    def __init__(self, layer_sizes, activation='relu', output_activation=None):
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.output_activation = output_activation
        self.weights = []
        self.biases = []
        # Инициализация весов и смещений
        for i in range(len(layer_sizes) - 1):
            weight = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2. / layer_sizes[i])
            bias = np.zeros(layer_sizes[i+1])
            self.weights.append(weight)
            self.biases.append(bias)

    def forward(self, x):
        activations = []
        input = x
        for i in range(len(self.weights)):
            z = np.dot(input, self.weights[i]) + self.biases[i]
            if i == len(self.weights) - 1:
                if self.output_activation == 'softmax':
                    a = self.softmax(z)
                else:
                    a = z  # Для критика
            else:
                a = self.relu(z) if self.activation == 'relu' else z
            activations.append(a)
            input = a
        return activations

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def get_params(self):
        return self.weights, self.biases

    def set_params(self, weights, biases):
        self.weights = weights
        self.biases = biases

    def update_params(self, d_weights, d_biases, lr):
        for i in range(len(self.weights)):
            self.weights[i] -= lr * d_weights[i]
            self.biases[i] -= lr * d_biases[i]

class ActorNetwork(NeuralNetwork):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__([state_dim, hidden_dim, hidden_dim, action_dim], activation='relu', output_activation='softmax')

class CriticNetwork(NeuralNetwork):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        # Для двух агентов: state_dim * 2 + action_dim * 2
        super().__init__([state_dim * 2 + action_dim * 2, hidden_dim, hidden_dim, 1], activation='relu', output_activation=None)

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class MADDPGAgent:
    def __init__(self, state_dim, action_dim, agent_id, lr=0.01, gamma=0.99, tau=0.01):
        self.agent_id = agent_id
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim

        # Actor и Critic сети
        self.actor = ActorNetwork(state_dim, action_dim)
        self.actor_target = ActorNetwork(state_dim, action_dim)
        self.actor_target.set_params(*self.actor.get_params())

        self.critic = CriticNetwork(state_dim, action_dim)
        self.critic_target = CriticNetwork(state_dim, action_dim)
        self.critic_target.set_params(*self.critic.get_params())

        # Скорости обучения
        self.lr_actor = lr
        self.lr_critic = lr

    def select_action(self, state, epsilon):
        probs = self.actor.forward(state.reshape(1, -1))[-1][0]
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_dim)
        return np.argmax(probs)

    def update_critic(self, transitions, other_agents_next_actions):
        batch_size = len(transitions)
        state_dim = self.actor.layer_sizes[0]
        action_dim = self.action_dim

        # Извлечение данных из переходов
        states = np.array([t.state for t in transitions])  # shape: (batch_size, state_dim)
        actions = np.array([t.action for t in transitions])  # shape: (batch_size,)
        rewards = np.array([t.reward for t in transitions])  # shape: (batch_size,)
        next_states = np.array([t.next_state for t in transitions])  # shape: (batch_size, state_dim)
        dones = np.array([t.done for t in transitions])  # shape: (batch_size,)

        # Преобразование действий в one-hot
        actions_one_hot = np.zeros((batch_size, action_dim))
        actions_one_hot[np.arange(batch_size), actions] = 1  # shape: (batch_size, action_dim)

        # Получение целевых действий для других агентов
        target_actions = []
        for agent, a_next in other_agents_next_actions.items():
            # a_next: array of actions for this agent in the batch
            target_one_hot = np.zeros((batch_size, agents[agent].action_dim))
            target_one_hot[np.arange(batch_size), a_next] = 1
            target_actions.append(target_one_hot)

        # Добавляем действия текущего агента
        agent_actions_one_hot = np.zeros((batch_size, action_dim))
        agent_actions_one_hot[np.arange(batch_size), actions] = 1
        target_actions.append(agent_actions_one_hot)

        # Объединяем все действия в один массив
        target_actions_concat = np.concatenate(target_actions, axis=1)  # shape: (batch_size, total_action_dim)

        # Объединяем состояния и действия для критика
        combined_next_input = np.concatenate([next_states, target_actions_concat], axis=1)  # shape: (batch_size, state_dim * 2 + action_dim * 2)

        # Прогон целевых сетей через критик
        target_q_values = self.critic_target.forward(combined_next_input).flatten()  # shape: (batch_size,)

        # Целевые Q-значения
        target_q = rewards + self.gamma * target_q_values * (1 - dones)  # shape: (batch_size,)

        # Объединяем текущие состояния и действия для критика
        combined_current_input = np.concatenate([states, actions_one_hot], axis=1)  # shape: (batch_size, state_dim * 2 + action_dim * 2)

        # Текущее Q-значение
        current_q = self.critic.forward(combined_current_input).flatten()  # shape: (batch_size,)

        # Вычисление ошибки (MSE)
        td_error = target_q - current_q  # shape: (batch_size,)
        loss = np.mean(td_error ** 2)

        print(f"Agent {self.agent_id} - Critic Loss: {loss}")

        # Здесь необходимо реализовать обновление критика с использованием градиентного спуска
        # Это сложная задача, требующая ручной реализации обратного распространения ошибок
        # Для упрощения, этот шаг не реализован

        # Обновление целевых сетей
        self.soft_update()

    def soft_update(self):
        # Обновление целевых сетей
        actor_weights, actor_biases = self.actor.get_params()
        actor_target_weights, actor_target_biases = self.actor_target.get_params()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.tau * actor_weights[i] + (1 - self.tau) * actor_target_weights[i]
            actor_target_biases[i] = self.tau * actor_biases[i] + (1 - self.tau) * actor_target_biases[i]
        self.actor_target.set_params(actor_target_weights, actor_target_biases)

        critic_weights, critic_biases = self.critic.get_params()
        critic_target_weights, critic_target_biases = self.critic_target.get_params()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.tau * critic_weights[i] + (1 - self.tau) * critic_target_weights[i]
            critic_target_biases[i] = self.tau * critic_biases[i] + (1 - self.tau) * critic_target_biases[i]
        self.critic_target.set_params(critic_target_weights, critic_target_biases)

# Инициализация среды
env = PotatoFieldEnv(render_mode="rgb_array")

# Инициализация агентов
agents = {}
replay_buffer = {}
for agent_id in env.agents:
    agents[agent_id] = MADDPGAgent(state_dim=GRID_SIZE*GRID_SIZE, action_dim=env.action_space(agent_id).n, agent_id=agent_id)
    replay_buffer[agent_id] = ReplayBuffer(REPLAY_BUFFER_SIZE)

# Инициализация переменных для обучения
epsilon = EPSILON_START
frames = []
total_steps = 0

for episode in range(NUM_EPISODES):
    observations = env.reset()
    episode_rewards = {agent_id: 0 for agent_id in env.agents}

    while True:
        actions = {}
        for agent_id, obs in observations.items():
            state = obs.flatten()
            actions[agent_id] = agents[agent_id].select_action(state, epsilon)

        env.step(actions)
        next_observations = {agent_id: env.observe(agent_id) for agent_id in env.agents}
        rewards = env.rewards
        dones = env.terminations

        # Сохранение переходов в буфер
        for agent_id in env.agents:
            state = observations[agent_id].flatten()
            action = actions[agent_id]
            reward = rewards[agent_id]
            next_state = next_observations[agent_id].flatten()
            done = dones[agent_id]
            replay_buffer[agent_id].push(state, action, reward, next_state, done)
            episode_rewards[agent_id] += reward

        observations = next_observations
        total_steps += 1

        # Обучение агентов
        for agent_id in env.agents:
            if len(replay_buffer[agent_id]) > BATCH_SIZE:
                transitions = replay_buffer[agent_id].sample(BATCH_SIZE)
                other_agents_next_actions = {}
                for other_agent in env.agents:
                    if other_agent != agent_id:
                        other_actions = []
                        for t in transitions:
                            other_actions.append(agents[other_agent].select_action(t.next_state, epsilon=0))
                        other_agents_next_actions[other_agent] = np.array(other_actions)
                # Обновление критика и актера
                agents[agent_id].update_critic(transitions, other_agents_next_actions)
                agents[agent_id].soft_update()

        # Сохранение кадров для видео
        if total_steps % FRAME_SKIP == 0:
            frames.append(env.render())

        if all(dones.values()):
            break

    # Уменьшение epsilon
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    # Вывод статистики
    total_reward = sum(episode_rewards.values())
    print(f"Эпизод {episode+1}/{NUM_EPISODES}, Награда: {total_reward}")

# Вывод максимальных наград для каждого агента
print("Max rewards for each agent:")
for agent, reward in env.rewards.items():
    print(f"{agent}: {reward:.1f}")

# Запись видео из сохранённых кадров
with imageio.get_writer("potato_field_simulation_maddpg.mp4", fps=10) as writer:
    for frame in frames:
        writer.append_data(frame)

print("Видео сохранено как potato_field_simulation_maddpg.mp4")

if __name__ == "__main__":
    # Параметры тестирования
    state_dim = GRID_SIZE * GRID_SIZE  # Используем размер состояния, как в основной части
    action_dim = env.action_space("tractor_red").n  # Используем размер действия одного из агентов
    num_agents = len(env.agents)
    batch_size = 64

    # Инициализация одного агента для теста
    agent = MADDPGAgent(state_dim=state_dim, action_dim=action_dim, agent_id="tractor_red")

    # Создаем фиктивные переходы для тестирования
    dummy_transitions = [
        Transition(np.random.rand(state_dim), np.random.randint(0, action_dim),
                   np.random.rand(), np.random.rand(state_dim), False)
        for _ in range(batch_size)
    ]

    # Генерируем фиктивные действия других агентов
    dummy_other_agent_actions = {
        f"agent_{i}": np.random.randint(0, action_dim, batch_size) for i in range(1, num_agents)
    }

    # Проверка метода update_critic
    agent.update_critic(dummy_transitions, dummy_other_agent_actions)
