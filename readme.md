# MY LunarLander: An OpenAI Gymnasium and PyTorch-based project [Update 2023]

MY LunarLander is a simplified Python project that teaches how to use OpenAI's Gymnasium and PyTorch to train a
reinforcement learning agent to land a lunar module. This project uses a QNetwork to model the agent's learning process
and is designed to be easy to understand, modify, and run. The project addresses the problem of outdated projects and
fast API changes by providing a code that is compatible with current APIs and requires minimal changes for execution.

![LunarLander Gameplay](doc/captured.gif)

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)

## Requirements

- Python 3.7+
- OpenAI Gymnasium
- PyTorch
- NumPy

## Installation

1. Clone this repository:

```
https://github.com/pajuhaan/LunarLander/LunarLander.git
```

2. Create a virtual environment and activate it:

I highly recommend utilizing **Anaconda** + **PyCharm** for your Python projects!

What's Conda? Yes if you are just start Python read [this](#Conda-in-Python).

3. Install the required dependencies:

```
pip install -r requirements.txt
```

## Project Structure

- `Agent.py`: This file defines the neural network architecture, which includes the Actor (Policy) model. The agent
  learns the optimal policy using the QNetwork.
- `Learn.py`: This file initializes the environment, trains the agent, and saves the learned model as a checkpoint.
- `Runner.py`: This file loads the trained model checkpoint and plays the game using the learned agent.
- `Capture.py`: This file saves a video of the trained agent playing the game.

### Environment

[Lunar Lander environment](https://gymnasium.farama.org/environments/box2d/lunar_lander/) is part of the Box2D
environments which contains general information about the environment.

| Space             | Description                                                                                          |
|-------------------|------------------------------------------------------------------------------------------------------|
| Action Space      | Discrete(4)                                                                                          |
| Observation Space | Box([-90. -90. -5. -5. -3.1415927 -5. -0. -0. ], [90. 90. 5. 5. 3.1415927 5. 1. 1. ], (8,), float32) |

### Action Space

| Code | Action                        |
|------|-------------------------------|
| 0    | do nothing                    |
| 1    | fire left orientation engine  |
| 2    | fire main engine              |
| 3    | fire right orientation engine |

### Observation

| Index | Parameter         | Description                                |
|-------|-------------------|--------------------------------------------|
| 0     | x-coordinate      | Lander's x-coordinate                      |
| 1     | y-coordinate      | Lander's y-coordinate                      |
| 2     | x-linear velocity | Lander's linear velocity in x-direction    |
| 3     | y-linear velocity | Lander's linear velocity in y-direction    |
| 4     | angle             | Lander's angle                             |
| 5     | angular velocity  | Lander's angular velocity                  |
| 6     | left leg contact  | Boolean: left leg contact with the ground  |
| 7     | right leg contact | Boolean: right leg contact with the ground |

### Learn

This will save the learned model as a checkpoint file named `checkpoint.pth`.

**qnetwork_local**: This network, also called the "local" or "online" network, is the main neural network used to predict the Q-values for a given state. It is updated at each learning step using gradient descent to minimize the difference between its predicted Q-values and the target Q-values.

**qnetwork_target**: This network, also known as the "target" or "fixed" network, is a separate neural network that is used to stabilize the learning process. The target network has the same architecture as the local network but with a time-delayed copy of the local network's weights. It is used to compute the target Q-values for the next state during the learning process. The target network's weights are updated less frequently than the local network's weights, usually by a soft update rule (a weighted average of the local and target network weights) or by a hard update (copying the local network weights to the target network after a certain number of learning steps).


The main idea behind using two separate networks is to address the issue of overestimation bias in Q-learning. Overestimation bias occurs when the Q-values are updated based on the maximum Q-value of the next state, which could lead to an overly optimistic estimate of the action values. By using a separate, slowly updated target network, DQN reduces the correlation between the predicted Q-values and the target Q-values, leading to more stable training and better performance.

### Runner

This will load the `checkpoint.pth` file and play the game using the learned model.

### Capturing a Video

This will save a video file of the trained agent playing the game.

Enjoy your LunarLander project and have fun learning reinforcement learning with OpenAI Gym and PyTorch!

## Conda in Python

Conda is a powerful cross-platform package manager and environment management system for Python and other programming
languages. It was primarily developed for the Anaconda distribution, which is a popular distribution of Python and R
programming languages for data science, machine learning, and scientific computing. Conda makes it easy to install
packages and manage dependencies, as well as to create, export, and share isolated environments for different projects.

## Basics


| Concept     | Description                                                                                        |
|-------------|----------------------------------------------------------------------------------------------------|
| Agent       | The entity that learns and makes decisions in the reinforcement learning process.                  |
| State       | The current situation or condition of the environment, as perceived by the agent.                  |
| Action      | The choices or moves made by the agent to interact with the environment.                           |
| Reward      | The feedback signal provided by the environment to the agent, indicating the quality of an action. |
| Environment | The context in which the agent interacts and learns, providing states and rewards to the agent.    |


If you're interested in gaining a deeper understanding of concepts such as agents, states, actions, rewards, and
environments, I recommend watching this video by Lex Fridman.

[![Video Thumbnail](https://img.youtube.com/vi/zR11FLZ-O9M/maxresdefault.jpg)](https://www.youtube.com/watch?v=zR11FLZ-O9M)

### Benefits of Conda

1. **Cross-platform compatibility**: Conda works seamlessly across Windows, macOS, and Linux, making it easy to manage
   packages and environments on different operating systems.
2. **Environment management**: Conda enables you to create isolated environments for different projects, ensuring that
   packages and dependencies do not conflict with one another. This is particularly useful when working with projects
   that require different versions of the same package.
3. **Reproducibility**: Conda makes it simple to share and reproduce environments by exporting and importing environment
   configurations, ensuring that your project can be easily set up by others.
4. **Large package ecosystem**: Conda includes an extensive collection of packages, including those related to data
   science, machine learning, and scientific computing, which can be easily installed and managed.
5. **Easy package installation**: Conda simplifies the process of installing packages by handling dependencies
   automatically and providing pre-built binary packages that save time and reduce the need for manual compilation.

To get started with Conda, you can download and install the Anaconda distribution, which includes Conda as well as a
large collection of pre-installed Python packages. Visit
the [Anaconda website](https://www.anaconda.com/products/distribution) to download the appropriate installer for your
operating system and follow the installation instructions.

## Purpose

My goal is to explore simple environments and demonstrate the creation of basic networks, helping users understand the
output and behavior of these networks. This foundation will enable the implementation of more complex models in the
future. While training large models on intricate environments can be exciting, implementing concepts in the simplest
possible environments provides valuable insights. Additionally, this approach makes it easier for beginners in
reinforcement learning to comprehend and execute the code.

## Author

Pajuhaan

## License

BSD
