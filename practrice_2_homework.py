import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Initialize the Riverraid environment
env = gym.make("ALE/Riverraid-v5", render_mode="rgb_array")

# Check action space and observation space
print("Action Space:", env.action_space)
print("Observation Space:", env.observation_space)
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
      x = self.conv(x)
      x = x.reshape(x.size(0), -1)  # Используем reshape вместо view
      return self.fc(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), action, reward, np.array(next_state), done
    
    def __len__(self):
        return len(self.buffer)
class DQNAgent:
    def __init__(self, input_shape, num_actions):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_actions = num_actions
        self.epsilon = 1.0  # Initial exploration
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.gamma = 0.99  # Discount factor
        self.learning_rate = 0.00025
        self.batch_size = 32
        self.memory_capacity = 10000
        self.target_update = 1000
        
        # DQN and target network
        self.policy_net = DQN(input_shape, num_actions).to(self.device)
        self.target_net = DQN(input_shape, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is not trained
        
        # Replay memory
        self.memory = ReplayBuffer(self.memory_capacity)
        
        # Optimizer and loss function
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        
    def select_action(self, state):
        # Перестановка осей: (H, W, C) -> (C, H, W)
        state = np.transpose(state, (2, 0, 1))
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        q_values = self.policy_net(state)
        return q_values.max(1)[1].item()

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Перестановка осей для состояний и следующих состояний
        states = np.transpose(np.array(states), (0, 3, 1, 2))  # (N, H, W, C) -> (N, C, H, W)
        next_states = np.transpose(np.array(next_states), (0, 3, 1, 2))

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        q_values = self.policy_net(states).gather(1, actions)
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = self.loss_fn(q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
from tqdm import tqdm  # Импортируем tqdm для отображения прогресса

num_episodes = 1000
target_update_frequency = 1000
max_steps_per_episode = 10000

# Get input shape and action space
input_shape = (3, 210, 160)  # For RGB images
num_actions = env.action_space.n

# Initialize agent
agent = DQNAgent(input_shape, num_actions)

# Training loop with progress bar
for episode in tqdm(range(num_episodes), desc="Training Progress"):
    state, _ = env.reset()
    total_reward = 0
    
    for step in range(max_steps_per_episode):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        agent.memory.push(state, action, reward, next_state, done)
        agent.update()
        
        state = next_state
        total_reward += reward
        
        if done:
            break
    
    # Update target network periodically
    if episode % target_update_frequency == 0:
        agent.update_target_network()
    
    # Log progress after each episode
    print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.4f}")

env.close()
import imageio

def record_video(env, agent, out_path, fps=30):
    frames = []
    state, _ = env.reset()
    
    done = False
    while not done:
        frame = env.render()
        frames.append(frame)
        
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
    
    # Save video
    imageio.mimsave(out_path, frames, fps=fps)

# Record video of the trained agent
record_video(env, agent, "riverraid_play.mp4")
