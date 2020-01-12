import numpy as np
import td3.constants as cons
# from torch.utils.tensorboard import SummaryWriter  # used for evaluate episode only


def train(agent, replay_buffer, step):
    """Train the agent for exploration steps
        Args:
            :param step: (NextStep)  move to the next step
            :param replay_buffer: (ReplayBuffer) replay buffer for arm
            :param agent: (Agent): agent to use
    """

    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    episode_reward = 0
    episode_timesteps = 0

    done = False
    # evaluations = [] # FOR EVALUATE EPISODE
    rewards = []
    best_avg = -2  # need a dummy value below any possible reward average
    # writer = SummaryWriter(comment="-TD3_BAXTER_VREP")  # used for evaluate episode only
    while total_timesteps < cons.EXPLORATION:
        if done:
            if total_timesteps != 0:
                rewards.append(episode_reward/episode_timesteps)
                avg_reward = np.mean(rewards[-100:])

                if best_avg < avg_reward:
                    best_avg = avg_reward
                    agent.save("best_avg", "saves")

                if avg_reward >= cons.REWARD_THRESH:
                    break

                # trains with the TD3 function
                agent.train(replay_buffer, cons.BATCH_SIZE)

                # Evaluate episode: NOT USED CURRENTLY
                # if timesteps_since_eval >= EVAL_FREQUENCY:
                #   timesteps_since_eval %= EVAL_FREQUENCY
                #   eval_reward = evaluate_policy(agent, test_env)
                #   evaluations.append(avg_reward)
                #   writer.add_scalar("eval_reward", eval_reward, total_timesteps)
                #   if best_avg < eval_reward:
                #       best_avg = eval_reward
                #       print("saving best model....\n")
                #       agent.save("best_avg","saves")

                # reset the values for a new episode, increment number of episodes
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

        # run a new step
        reward, done = step.next_step(episode_timesteps)
        # add the reward to the episode total for later averaging
        episode_reward += reward

        # increment the timesteps
        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1
