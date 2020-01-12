import numpy as np
import torch
from vrepsim import VrepSim
import td3.constants as cons
# from td3.evaluate_policy import evaluate_policy
from td3.experience.priority_replay_buffer import PrioritizedReplayBuffer
from td3.experience.replay_buffer import ReplayBuffer
from td3.td3 import TD3
from td3.train import train


if __name__ == '__main__':

    # Set seeds
    if cons.set_seed:
        torch.manual_seed(cons.SEED)
        np.random.seed(cons.SEED)
        agent = TD3()

        sim = VrepSim()

        replay_buffer = PrioritizedReplayBuffer(cons.BUFFER_SIZE, alpha=1)
        #replay_buffer = ReplayBuffer()

        print("\nInitializing experience replay buffer...")
        # fill replay buffer with random data, state and next_state are the same, no movement/reward, done=False.
        for x in range(cons.BUFFER_SIZE):
            state = np.random.uniform(low=0, high=255, size=(84, 84))
            next_state = state
            action = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
            reward = 0
            done = False
            replay_buffer.add(state, action, reward, next_state, done)
        print("\nExperience replay buffer initialized.")

        # observe(sim, replay_buffer, cons.OBSERVATION, "right")
        train(agent, sim, replay_buffer)

        # possible evaluation
        agent.load()


        # for i in range(100):
            # need to pass arm, add arm to evaluate policy
            # evaluate_policy(agent, sim, "left")
