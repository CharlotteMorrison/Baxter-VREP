import numpy as np
import td3.constants as cons
from utils import output_video, plot_results
# from utils import stack_frames
import time
import psutil
from utils import plot_loss
import torch
# --------------------------------
# for code performance profiling
# import cProfile
# import pstats
# --------------------------------


def train(agent, sim, replay_buffer):
    """Train the agent for exploration steps
        Args:
            :param replay_buffer: (ReplayBuffer) replay buffer for arm
            :param sim: (robot environment) vrep simulation
            :param agent: (Agent) agent to use, TD3 Algorithm
    """
    # profile = cProfile.Profile()
    arm = 'right'  # replace later with parameter, or etc. for dual arms
    total_timesteps = 0

    rewards_total_frame = []
    rewards_total_episode = []
    rewards_total_average = []
    actor_loss_episode = []
    critic_loss_episode = []
    # stacked_frames = 0
    episode = 0
    start_time = time.time()

    if cons.WRITE_TO_FILE:
        file_frame = open(cons.FILE_NAME + "_frame.csv", "w")
        file_frame.write("Frame,Reward,Avg_Reward_Last_1000,Avg_Reward_Last_10000,Avg_Reward_All,"
                         "Time_Elapsed")
        file_episode = open(cons.FILE_NAME + "_episode.csv", "w")
        file_episode.write("Episode,Reward,Avg_Reward_Last_" + str(cons.REPORT_INTERVAL) +
                           ",Avg_Reward_Last_100,Avg_Reward_All,Frames_total,Time_Elapsed,Solved,"
                           "Distance,Memory_Usage")
    else:  # just to get rid of unassigned variable errors.
        file_frame = ''
        file_episode = ''

    while total_timesteps < cons.EXPLORATION:
        print('Timesteps: {}/{}.'.format(total_timesteps, cons.EXPLORATION))
        episode += 1
        state = sim.get_input_image()

        score = []
        video_array = []
        distance = sim.calc_distance()

        solved = False
        index = 0  # track the number of bad moves made.

        if episode % cons.VIDEO_INTERVAL == 0:
            video_record = True
        else:
            video_record = False
        video_array.append(sim.get_video_image())
        temp_steps = 0  # tracks the number of tries- if above 30, is done, reset.
        while True:
            total_timesteps += 1

            action = agent.select_action(np.array(state), noise=cons.POLICY_NOISE).tolist()[0]

            if arm == 'right':
                sim.step_right(action)
            else:
                sim.step_left(action)

            new_distance = sim.calc_distance()
            new_state = sim.get_input_image()

            video_array.append(sim.get_input_image())
            # TODO create a more robust reward, move to function and apply to this and populate
            # determine reward after movement

            # try using pure distance for reward
            reward = distance - new_distance
            if new_distance > distance:
                index += 1
            else:
                index = 0

            # TODO update for multi-arm
            # check for collision state/ if done

            right_arm_collision_state = sim.get_collision_state()

            if new_distance < cons.SOLVED_DISTANCE:
                done = True
                solved = True
            elif right_arm_collision_state:
                solved = False
                done = True
                reward = -1
            else:
                done = False

            if index >= 7:
                done = True

            score.append(reward)
            replay_buffer.add(state, torch.tensor(action, dtype=torch.float32), reward, new_state, done)

            # profile.enable()
            agent.train(replay_buffer, cons.BATCH_SIZE)
            # profile.disable()
            # ps = pstats.Stats(profile)
            # ps.sort_stats('cumtime')
            # ps.print_stats()
            state = new_state
            distance = new_distance

            if solved:
                print('Solved on Episode: {}'.format(episode))

            temp_steps += 1
            if temp_steps == 30:
                done = True  # stop after 30 attempts, was getting stuck flipping from bad to good.
                temp_steps = 0

            system_info = psutil.virtual_memory()

            if cons.WRITE_TO_FILE:
                rewards_total_frame.append(reward)
                mean_reward_1000_frame = sum(rewards_total_frame[-1000:])/1000
                mean_reward_10000_frame = sum(rewards_total_frame[-10000:]) / 10000
                mean_reward_all_frame = round(sum(rewards_total_frame)/len(rewards_total_frame), 2)
                elapsed_time_frame = time.time() - start_time

                if total_timesteps >= 10000:
                    file_frame.write('\n{},{},{},{},{},{}'.format(total_timesteps, reward, mean_reward_1000_frame,
                                                                  mean_reward_10000_frame, mean_reward_all_frame,
                                                                  time.strftime("%H:%M:%S",
                                                                                time.gmtime(elapsed_time_frame))))
                elif total_timesteps >= 1000:
                    file_frame.write('\n{},{},{},null,{},{}'.format(total_timesteps, reward, mean_reward_1000_frame,
                                                                    mean_reward_all_frame,
                                                                    time.strftime("%H:%M:%S",
                                                                                  time.gmtime(elapsed_time_frame))))
                else:
                    file_frame.write('\n{},{},null,null,{},{}'.format(total_timesteps, reward, mean_reward_1000_frame,
                                                                      time.strftime("%H:%M:%S",
                                                                                    time.gmtime(elapsed_time_frame))))

            if done:
                rewards_total_episode.append(sum(score))

                mean_reward_100 = sum(rewards_total_episode[-100:]) / 100
                mean_reward_interval = sum(rewards_total_episode[-cons.REPORT_INTERVAL:]) / cons.REPORT_INTERVAL
                mean_reward_all = round(sum(rewards_total_episode) / len(rewards_total_episode), 2)

                rewards_total_average.append(mean_reward_all)

                # add in loss values
                actor_loss_episode.append(sum(agent.actor_loss_plot) / len(agent.actor_loss_plot))
                critic_loss_episode.append(sum(agent.critic_loss_plot) / len(agent.critic_loss_plot))
                # reset the agent lists
                agent.actor_loss_plot = []
                agent.critic_loss_plot = []

                elapsed_time = time.time() - start_time

                if video_record:
                    print('should be video')
                    output_video(episode, video_array, cons.SIZE, "td3/videos/" + cons.DEFAULT_NAME)

                if solved:
                    output_video(episode, video_array, cons.SIZE, "td3/videos/" + cons.DEFAULT_NAME)

                if cons.WRITE_TO_FILE:

                    if episode < cons.REPORT_INTERVAL:
                        file_episode.write('\n{},{},null,null,{},{},{},{},{},{}'.
                                           format(episode, str(sum(score)), mean_reward_all, total_timesteps,
                                                  time.strftime("%H:%M:%S", time.gmtime(elapsed_time)),
                                                  solved, distance, system_info.used))
                    elif episode < 100:
                        file_episode.write('\n{},{},{},null,{},{},{},{},{},{}'.
                                           format(episode, str(sum(score)), mean_reward_interval, mean_reward_all,
                                                  total_timesteps, time.strftime("%H:%M:%S", time.gmtime(elapsed_time)),
                                                  solved, distance, system_info.used))
                    else:
                        file_episode.write('\n{},{},{},{},{},{},{},{},{},{}'.
                                           format(episode, str(sum(score)), mean_reward_interval, mean_reward_100,
                                                  mean_reward_all, total_timesteps,
                                                  time.strftime("%H:%M:%S", time.gmtime(elapsed_time)), solved,
                                                  distance, system_info.used))

                if episode % cons.REPORT_INTERVAL == 0 and episode > 0:
                    plot_results(rewards_total_episode, cons.PLOT_NAME)
                    plot_results(rewards_total_average, 'td3/results/plots/Baxter_TD3_avg_reward.png')
                    plot_loss(actor_loss_episode, critic_loss_episode, 'td3/results/plots/Baxter_TD3_loss_plot.png')

                    print("\n*** Episode " + str(episode) + " ***")
                    print("Avg_Reward [last " + str(cons.REPORT_INTERVAL) + "]: " + str(
                        mean_reward_interval) + ", [last 100]: " + str(mean_reward_100) + ", [all]: " + str(
                        mean_reward_all))
                    print("Frames Total: " + str(total_timesteps))
                    print("Elapsed Time: ", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
                    print("Memory Usage: " + str(system_info.percent) + "%")

                ''' 
                if right_arm_collision_state:
                    sim.full_sim_reset()
                else:
                    sim.reset_sim()
                '''
                sim.reset_sim()
                break

        system_info = psutil.virtual_memory()

        if system_info.percent > 98:
            break

    plot_results(rewards_total_episode, cons.PLOT_NAME)
    plot_results(rewards_total_average, 'td3/results/plots/Baxter_TD3_avg_reward.png')
    plot_loss(actor_loss_episode, critic_loss_episode, 'td3/results/plots/Baxter_TD3_loss_plot.png')
