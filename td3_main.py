import numpy as np
import torch
from vrepsim import VrepSim
import td3.constants as cons
# from td3.evaluate_policy import evaluate_policy
from td3.experience.priority_replay_buffer import PrioritizedReplayBuffer
# from td3.experience.replay_buffer import ReplayBuffer
from td3.td3 import TD3
from td3.train import train
from td3.populate import populate_buffer


if __name__ == '__main__':

    # Set seeds
    if cons.set_seed:
        torch.manual_seed(cons.SEED)
        np.random.seed(cons.SEED)
        agent = TD3()

        sim = VrepSim()
        sim.reset_sim()

        replay_buffer = PrioritizedReplayBuffer(cons.BUFFER_SIZE, alpha=1)
        # replay_buffer = ReplayBuffer()

        populate_buffer(sim, replay_buffer)

        train(agent, sim, replay_buffer)

        # possible evaluation
        agent.load()
        # for i in range(100):
            # need to pass arm, add arm to evaluate policy
            # evaluate_policy(agent, sim, "left")
