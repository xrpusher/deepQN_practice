# Atari Riverraid Deep Q-Learning Agent

This project implements a Deep Q-Learning (DQN) algorithm to train an agent to play **Riverraid** on the Atari platform, using Gymnasium and PyTorch.

## Overview

The project demonstrates how to use Deep Q-Learning to train an agent to play the Atari game **Riverraid**. The agent uses a convolutional neural network (CNN) to approximate Q-values (action-value predictions) and makes decisions based on the RGB frames of the game.

### Gameplay Demonstration

You can watch a video demonstrating the trained agent's gameplay at the following link:
[Gameplay Video](https://github.com/xrpusher/deepQN_practice/blob/practice_2/riverraid_play.mp4)

### Riverraid Environment

Riverraid is a classic Atari arcade game where the player controls a plane flying over a river. The goal is to destroy enemy objects, avoid collisions, and replenish fuel by flying over fuel depots. The game provides rewards for destroying objects like tankers, helicopters, jets, and bridges.

You can read more about the Riverraid environment and its variations on the official Gymnasium documentation: [Riverraid Environment Documentation](https://gymnasium.farama.org/environments/atari/riverraid/)

# Statistics

1) RTX 2060 super
2) 1.5h
3) Max reward in video - 680, in episode - 970

### License
This project is released under the MIT License.
