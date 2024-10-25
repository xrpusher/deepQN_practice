# MARL Tanks Simulation

## Overview

This project is a **multi-agent reinforcement learning (MARL)** simulation called **"MARL Tanks"**, where multiple agents (tanks) move on a grid, shoot each other, and earn rewards based on their actions. The game is implemented using the **PettingZoo** library for multi-agent environments and is designed to demonstrate MARL in a competitive scenario.

### Game Description

The environment is a simple **10x10 grid**, where three agents (tanks) navigate, rotate, and shoot each other. Each agent can:
- Move in its current direction.
- Rotate left or right.
- Shoot bullets in the direction it is facing.

The agents receive rewards based on the following actions:
- **+1 reward** for hitting an opponent with a bullet.
- **-1 reward** for being hit by a bullet.
- **-0.1 reward** for every step they take (to discourage aimless movement).
- Agents are marked as "dead" when they are hit, and they no longer move or shoot, but their reward and status are still tracked.

### Game Dynamics
- **Grid**: The grid size is 10x10 cells, with each cell representing a position where a tank or bullet can be located.
- **Agents**: Three agents (tanks) are randomly placed on the grid at the start of each episode. They move and interact based on their policies.
- **Bullets**: When a tank shoots, a bullet moves in a straight line in the direction the tank is facing. Bullets disappear when they go off the grid or hit a tank.

### Goals of the Game
The goal for each agent is to maximize its cumulative reward over the course of the game by avoiding getting hit while shooting opponents. Agents must balance movement, shooting, and positioning to outplay the others.

### Learning Environment
The environment is designed to simulate a **multi-agent competitive scenario**, where each agent is trying to improve its strategy over multiple episodes. This makes it an ideal setup for demonstrating multi-agent reinforcement learning algorithms.

## Installation

To run this simulation, you'll need to install the following Python packages:

```bash
pip install numpy imageio opencv-python pettingzoo gymnasium

## Requirements:
- **Python 3.8 or higher**
- **numpy** for numerical operations
- **imageio** for saving video outputs
- **opencv-python** (`cv2`) for rendering the game
- **PettingZoo** for the multi-agent reinforcement learning environment

## Running the Simulation

You can run the simulation and see how the agents perform in their environment by executing the Python script:

```bash
python marl_tanks_simulation.py

## Simulation Parameters:

- **GRID_SIZE**: 10x10 grid where the tanks move and interact.
- **NUM_AGENTS**: 3 tanks that compete against each other.
- **CELL_SIZE**: 32 pixels per grid cell for visualization.
- **BULLET_SIZE**: 1/4 of a grid cell to represent bullets.
- **MAX_STEPS**: 100 steps per episode, after which the episode ends.
- **NUM_EPISODES**: 50 episodes for agents to interact and gather rewards.

## Game Mechanics

### Agent Actions:

- **Move forward**: The tank moves one cell in the direction it is currently facing.
- **Turn left/right**: The tank rotates 90 degrees to the left or right.
- **Shoot**: The tank fires a bullet in its current direction.

### Rewards:

- **+1** for hitting another agent with a bullet.
- **-1** for getting hit by another agent's bullet.
- **-0.1** for every step taken (small penalty to encourage meaningful movement).

### Termination:

- If an agent is hit by a bullet, it is marked as "dead," but it remains visible on the grid with a cross mark.

## Output: Video Recording

The simulation records each episode and saves it as a video file. The video will show:

- Each agent's movement and shooting.
- Bullets moving on the grid.
- Agents getting hit and marked as "dead" with a cross.
- A score display at the bottom showing each agent's cumulative reward throughout the episode.

The video file is saved as `tank_simulation_marl.mp4` in the current working directory.

## Customization

You can customize the simulation parameters by modifying the following variables in the script:

- **GRID_SIZE**: Change the grid dimensions to create larger or smaller environments.
- **NUM_AGENTS**: Increase or decrease the number of tanks in the environment.
- **NUM_EPISODES**: Increase the number of episodes to train agents longer.
- **CELL_SIZE** and **BULLET_SIZE**: Adjust the size of the grid cells and bullets for better visualization.

## Example Output

During the simulation, the agents will move, shoot, and try to eliminate each other. Below is an example of the game's layout:

- **Green tank**: Agent 0
- **Red tank**: Agent 1
- **Blue tank**: Agent 2
- **Black squares**: Bullets fired by the agents
- **Scoreboard**: At the bottom of the screen, showing the cumulative reward for each agent, with "dead" status when an agent is eliminated.

Here’s a screenshot from the simulation:

![Tank Simulation](path_to_screenshot.png)

## Future Work

This project can be expanded by:

- Adding more complex environments (e.g., obstacles or power-ups).
- Implementing more sophisticated MARL algorithms (e.g., MADDPG, QMIX).
- Introducing cooperation between agents for team-based dynamics.

## Conclusion

The MARL Tanks simulation is a simple and effective demonstration of multi-agent reinforcement learning in a competitive environment. It provides a base for further exploration of MARL algorithms and their applications in more complex domains.

Feel free to explore, modify, and extend this simulation to learn more about MARL!
