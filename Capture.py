import gymnasium as gym
import torch
# For visualization
from gym.wrappers.monitoring import video_recorder

from Agent import Agent


def capture_video(agent, env_name):
    env = gym.make(env_name, render_mode="rgb_array",
                   continuous=False,
                   gravity=-10.0,
                   enable_wind=True,
                   wind_power=15.0,
                   turbulence_power=1.5,
                   )

    vid = video_recorder.VideoRecorder(env, path="video/captured.mp4")
    agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
    observation = env.reset()[0]
    terminated = False
    while not terminated:
        vid.capture_frame()

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
    env.close()


agent = Agent(state_size=8, action_size=4, seed=0)
capture_video(agent, 'LunarLander-v2')
