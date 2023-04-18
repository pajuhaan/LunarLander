"""
Pajuhaan

https://gymnasium.farama.org/environments/box2d/lunar_lander/
"""
import os
from collections import deque

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from Agent import Agent

# For visualization

env = gym.make('LunarLander-v2',
               # render_mode="human",
               continuous=False,
               gravity=-10.0,
               enable_wind=True,
               wind_power=15.0,
               turbulence_power=1.5,
               )
print('\n━━━━━━━━━━ Initialize Device ━━━━━━━━━━')
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)
checkpoint_path = 'model/checkpoint.pth'


def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    best_avg_score = -np.inf  # Before the for i_episode in range(1, n_episodes + 1): loop, add a variable to track the best average score:

    for i_episode in range(1, n_episodes + 1):
        state = env.reset()[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            observation, reward, terminated, truncated, info = env.step(action)
            agent.step(state, action, reward, observation, terminated)
            state = observation
            score += reward
            if terminated:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            avg_score = np.mean(scores_window)
            if avg_score > best_avg_score:  # Modify the saving condition inside the loop to only save if the average score is better than the previous best:
                best_avg_score = avg_score
                torch.save(agent.qnetwork_local.state_dict(), checkpoint_path)
                print('\r━━━> Episode {}\tAverage Score: {:.2f} | 🆙 Saved'.format(i_episode, avg_score))
            else:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, avg_score))

        if np.mean(scores_window) >= 200.0:
            print('\n━━━━━━━━━━━━━━━ SOLVED ━━━━━━━━━━━━━━━')
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            break
    return scores


agent = Agent(state_size=8, action_size=4, seed=0)

print('\n━━━━━━━━━━━━━━━ LOAD MODEL ━━━━━━━━━━━━━━━')
if os.path.exists(checkpoint_path):
    print("✅ Checkpoint has been loaded.")
    # Load the checkpoint
    agent.qnetwork_local.load_state_dict(torch.load(checkpoint_path))
    agent.qnetwork_target.load_state_dict(torch.load(checkpoint_path))
else:
    # Do something else if the checkpoint doesn't exist
    print("❌ Checkpoint file does not exist.")

scores = dqn()

# plot the scores

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
