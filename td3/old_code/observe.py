import sys
import pickle


def observe(sim, replay_buffer, observation_steps, arm):
    """run episodes while taking random actions and filling replay_buffer
        Args:
            :param sim: (env): gym environment
            :param replay_buffer: (ReplayBuffer): buffer to store experience replay
            :param observation_steps: (int): how many steps to observe for
            :param arm: (string): right or left for arm
    """
    time_steps = 0
    sim.resetSim()
    buffer_storage = []

    # store and load the initial replay values
    # once replay buffer is full, use the pre-made one to populate observe step
    replay_counter = 0
    try:
        pk_file = open("/home/charlotte/PycharmProjects/Baxter/td3algorithm/temp/buffer.pkl", "rb")
        data = pickle.load(pk_file)

        for test in data:
            replay_buffer.add(test[0], test[1], test[2], test[3], test[4])
            buffer_storage.append([test[0], test[1], test[2], test[3], test[4]])
            replay_counter += 1
    except EOFError:
        pass

    observation_steps -= replay_counter
    print(observation_steps)

    # chooses the correct arm action, default left
    if arm == "right":
        obs = sim.right_state()
    else:
        obs = sim.left_state()
    while time_steps < observation_steps:
        if arm == "left":
            new_obs, action, reward, done = env.random_step_left()
        else:
            new_obs, action, reward, done = env.random_step_right()

        # add the observation to the replay buffer
        replay_buffer.add(obs, new_obs, action, reward, done)

        # save the observations, for testing , remove later after testing
        buffer_storage.append([obs, new_obs, action, reward, done])
        if time_steps % 25 == 0:
            save_buffer = open("/home/charlotte/PycharmProjects/Baxter/td3algorithm/temp/buffer.pkl", "wb")
            pickle.dump(buffer_storage, save_buffer)
            save_buffer.close()
            print("buffer updated")

        # set the current observation to the new observation
        obs = new_obs
        time_steps += 1

        # if finished, reset the arm and get the reset state
        if done:
            env.reset(arm)
            if arm == "left":
                obs = env.left_state()
            else:
                obs = env.right_state()

        if time_steps % 100 == 0:
            print("\rPopulating Buffer {}/{}.".format(time_steps, observation_steps))
        sys.stdout.flush()