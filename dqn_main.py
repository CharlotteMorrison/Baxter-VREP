import random
import time
import torch
import psutil
from dqn import dqn_constants as cons
from dqn.dqn_algorithm import QNetAgent
from vrepsim import VrepSim
from utils import stack_frames, output_video, plot_results
from dqn.dqn_calculations import calculate_epsilon
from dqn.dqn_experience_replay import ExperienceReplay


if __name__ == '__main__':
    if cons.set_seed:
        torch.manual_seed(cons.seed_value)
        random.seed(cons.seed_value)

    if cons.write_to_file:
        file_episode = open(cons.file_name + "_episode.csv", "w")
        file_episode.write("Episode,Reward,Avg_Reward_Last_" + str(cons.report_interval) +
                           ",Avg_Reward_Last_100,Avg_Reward_All,Frames_total,Time_Elapsed,Solved,"
                           "Epsilon,Distance,Memory_Usage")
        file_frame = open(cons.file_name + "_frame.csv", "w")
        file_frame.write("Frame,Reward,Avg_Reward_Last_1000,Avg_Reward_Last_10000,Avg_Reward_All,"
                         "Time_Elapsed,Solved,Done,Epsilon,Distance,Memory_Usage")

    memory = ExperienceReplay(cons.replay_mem_size)
    q_agent = QNetAgent(memory)

    sim = VrepSim()

    rewards_total_episode = []
    rewards_total_frame = []

    frames_total = 0

    solved_after_episode = 0
    solved_after_frame = 0
    solved = False

    boltzmann = True

    stacked_frames = 0
    episode = 0

    boltz_frames = 0

    temperature = -1

    size = (512, 512)

    time.sleep(2)

    start_time = time.time()

    while frames_total < cons.num_frames:

        episode += 1

        state = sim.get_input_image()

        state, stacked_frames = stack_frames(stacked_frames, state, True, cons.num_frames_stacked)

        score = []

        video_array = []

        distance = sim.calc_distance()

        solved = False

        index = 0

        if episode % cons.video_interval == 0:
            video_record = True
        else:
            video_record = False

        video_array.append(sim.get_video_image())

        while True:

            frames_total += 1

            epsilon = calculate_epsilon(frames_total)

            action = q_agent.select_action_egreedy(state, epsilon)

            sim.move_joint(action)

            new_distance = sim.calc_distance()

            new_state = sim.get_input_image()

            new_state, stacked_frames = stack_frames(stacked_frames, new_state, False, cons.num_frames_stacked)

            video_array.append(sim.get_video_image())

            if new_distance > distance:
                reward = -1
            elif new_distance == distance:
                reward = 0
            else:
                reward = 1

            right_arm_collision_state = sim.get_collision_state()

            if new_distance < cons.solved_distance:
                done = True
                solved = True

            elif right_arm_collision_state:
                solved = False
                done = True
                reward = -1

            else:
                done = False

            if new_distance > distance:
                index += 1
            else:
                index = 0

            if index >= 3:
                done = True

            score.append(reward)

            memory.push(state, action, new_state, reward, done)
            q_agent.optimize()

            state = new_state
            distance = new_distance

            if solved:
                print("Solved on Episode:" + str(episode))

            system_info = psutil.virtual_memory()

            if cons.write_to_file:

                rewards_total_frame.append(reward)
                mean_reward_1000_frame = sum(rewards_total_frame[-1000:]) / 1000
                mean_reward_10000_frame = sum(rewards_total_frame[-10000:]) / 10000
                mean_reward_all_frame = round(sum(rewards_total_frame) / len(rewards_total_frame), 2)
                elapsed_time_frame = time.time() - start_time

                if frames_total >= 10000:
                    file_frame.write("\n" + str(frames_total) + "," + str(reward) + "," + str(mean_reward_1000_frame) + "," + str(
                            mean_reward_10000_frame) + "," + str(mean_reward_all_frame) + "," + time.strftime(
                            "%H:%M:%S", time.gmtime(elapsed_time_frame)) + "," + str(solved) + "," + str(
                            done) + "," + str(round(epsilon, 2)) + "," + str(distance) + "," + str(system_info.used))
                elif frames_total >= 1000:
                    file_frame.write("\n" + str(frames_total) + "," + str(reward) + "," + str(
                        mean_reward_1000_frame) + ",null," + str(mean_reward_all_frame) + "," + time.strftime(
                        "%H:%M:%S", time.gmtime(elapsed_time_frame)) + "," + str(solved) + "," + str(done) + "," + str(
                        round(epsilon, 2)) + "," + str(distance) + "," + str(system_info.used))
                else:
                    file_frame.write("\n" + str(frames_total) + "," + str(reward) + ",null,null," + str(
                        mean_reward_all_frame) + "," + time.strftime("%H:%M:%S",
                                                                     time.gmtime(elapsed_time_frame)) + "," + str(
                        solved) + "," + str(done) + "," + str(round(epsilon, 2)) + "," + str(distance) + "," + str(
                        system_info.used))

            if done:
                rewards_total_episode.append(sum(score))

                mean_reward_100 = sum(rewards_total_episode[-100:]) / 100
                mean_reward_interval = sum(rewards_total_episode[-cons.report_interval:]) / cons.report_interval
                mean_reward_all = round(sum(rewards_total_episode) / len(rewards_total_episode), 2)

                elapsed_time = time.time() - start_time

                if video_record:
                    output_video(episode, video_array, size, "dqn/videos/" + cons.default_name)

                if solved:
                    output_video(episode, video_array, size, "dqn/videos/" + cons.default_name)

                if cons.write_to_file:

                    if episode < cons.report_interval:
                        file_episode.write("\n" + str(episode) + "," + str(sum(score)) + ",null,null," + str(
                            mean_reward_all) + "," + str(frames_total) + "," + time.strftime("%H:%M:%S", time.gmtime(
                            elapsed_time)) + "," + str(solved) + "," + str(round(epsilon, 2)) + "," + str(
                            distance) + "," + str(system_info.used))
                    elif episode < 100:
                        file_episode.write("\n" + str(episode) + "," + str(sum(score)) + "," + str(
                            mean_reward_interval) + ",null," + str(mean_reward_all) + "," + str(
                            frames_total) + "," + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)) + "," + str(
                            solved) + "," + str(round(epsilon, 2)) + "," + str(distance) + "," + str(system_info.used))
                    else:
                        file_episode.write(
                            "\n" + str(episode) + "," + str(sum(score)) + "," + str(mean_reward_interval) + "," + str(
                                mean_reward_100) + "," + str(mean_reward_all) + "," + str(
                                frames_total) + "," + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)) + "," + str(
                                solved) + "," + str(round(epsilon, 2)) + "," + str(distance) + "," + str(
                                system_info.used))

                if episode % cons.report_interval == 0 and episode > 0:
                    plot_results(rewards_total_episode, cons.plot_name)

                    print("\n*** Episode " + str(episode) + " ***")
                    print("Avg_Reward [last " + str(cons.report_interval) + "]: " + str(
                        mean_reward_interval) + ", [last 100]: " + str(mean_reward_100) + ", [all]: " + str(
                        mean_reward_all))
                    print("Epsilon: " + str(round(epsilon, 2)))
                    print("Temperature: " + str(temperature))
                    print("Frames Total: " + str(frames_total))
                    print("Elapsed Time: ", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
                    print("Memory Usage: " + str(system_info.percent) + "%")

                sim.reset_sim()
                break

        system_info = psutil.virtual_memory()

        if system_info.percent > 98:
            break

    plot_results(rewards_total_episode, cons.plot_name)
