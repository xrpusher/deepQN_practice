# **Riverraid Reinforcement Learning**

This repository contains implementations of Deep Q-Network (DQN) and
Proximal Policy Optimization (PPO) to train and test agents on the
\"Riverraid\" game from the Atari Learning Environment using gymnasium.
The repository supports training models, saving weights, and recording
gameplay videos.

## **Project Structure**

- DQN_homework_train_save_weight.py: Trains a DQN agent and saves the
  > trained weights.

- DQN_load_train_test.py: Loads saved DQN weights, tests the agent, and
  > records gameplay as a video.

- PPO_homework_train_save_weight.py: Trains a PPO agent, saves weights,
  > and uses callbacks for training checkpoints.

- PPO_load_train_test.py: Loads a saved PPO model, tests the agent, and
  > records gameplay as a video.

- dqn_riverraid_final.pth: Pre-trained weights for the DQN agent.

- riverraid_ppo_play.mp4: Recorded gameplay video of the PPO agent.

- requirements.txt: Python dependencies required for the project.

- README.md: Overview of the project.

## **Getting Started**

### **Installation**

Clone this repository:  
bash  
Copy code  
git clone \<repository-url\>

cd \<repository-directory\>


Install dependencies:  
bash  
Copy code  
pip install -r requirements.txt


### **Usage**

**Training DQN:  
**bash  
Copy code  
python DQN_homework_train_save_weight.py


**Testing DQN and Recording Video:** Ensure dqn_riverraid_final.pth is
in the project directory:  
bash  
Copy code  
python DQN_load_train_test.py


**Training PPO:  
**bash  
Copy code  
python PPO_homework_train_save_weight.py


**Testing PPO and Recording Video:** Ensure ppo_riverraid_final.zip is
in the project directory:  
bash  
Copy code  
python PPO_load_train_test.py

### **Results**

- The models generate gameplay videos demonstrating their performance:

  - dqn_riverraid_play.mp4

  - riverraid_ppo_play.mp4

## **Requirements**

- Python 3.8 or higher

- CUDA-compatible GPU (optional, for faster training)

### **Dependencies**

See requirements.txt for all dependencies.
