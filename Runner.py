import gymnasium as gym
import torch

# For visualization
from Agent import Agent

agent = Agent(state_size=8, action_size=4, seed=0)

env = gym.make("LunarLander-v2", render_mode="human",
               continuous=False,
               gravity=-10.0,
               enable_wind=True,
               wind_power=15.0,
               turbulence_power=1.5,
               )
agent.qnetwork_local.load_state_dict(torch.load('model/checkpoint.pth'))

state= env.reset()
observation = state[0]
print('Initial state',state)

for _ in range(1000):
    action = agent.act(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated:
        print('━━━━━━━━━━━━━━━ DONE ━━━━━━━━━━━━━━━')
        print('Reward: ', reward)
        print('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')
        exit()
    if truncated:
        print('━━━━━━━━━━━━━ Truncated ━━━━━━━━━━━━━')
        print('Reward: ', reward)
        print('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')
        exit()
