import numpy as np
from utils import stack_frames
import td3.constants as cons
import torch


class Runner:
    """Carries out the environment steps and adds experiences to memory"""

    def __init__(self, env, agent, replay_buffer, arm):
        self.env = env
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.obs = env.reset(arm)
        self.done = False
        self.arm = arm
        self.observation_steps = 50  # 200, need to change average to match in train.py
        self.stacked_frames = 0
        self.episode = 0
        self.video_record = True
        self.frames_total = 0
        self.state = self.env.getInputImage()
        self.state, self.stacked_frames = stack_frames(stack_frames, self.state, True)
        self.score = []
        self.video_array = []
        self.distance = self.env.calcDistance()
        self.solved = False
        self.index = 0
        if self.episode % cons.VIDEO_INTERVAL == 0:
            self.video_record = True
        else:
            self.video_record = False
        self.video_array.append(self.env.getVideoImage())

    def next_step(self, episode_timesteps):

        # todo get initial state
        self.frames_total += 1

        # Perform action
        action = self.agent.select_action(np.array(self.obs))
        if self.arm == 'right':
            self.env.step_right(action)
        else:
            self.env.step_left(action)

        # todo get new state, reward, done

        if episode_timesteps + 1 == self.observation_steps:
            done_bool = 0
            done = True
        else:
            done_bool = float(done)
        state = list(raw_state.values())
        next_state = list(raw_next_state.values())

        # Store data in replay buffer
        self.replay_buffer.add(raw_state, raw_next_state, action, reward, done_bool)

        self.obs = next_state

        if done:
            # TODO change to random state
            self.obs = self.env.reset_random(self.arm)
            # moves arm to a random starting position
            self.done = False

            return reward, True

        return reward, done
