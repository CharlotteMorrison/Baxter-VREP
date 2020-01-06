import torch
import cv2
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import dqn_constants as cons


def load_model():
    return torch.load(cons.model_file)


def save_model(model):
    torch.save(model.state_dict(), cons.model_file)


def preprocess_frame(frame):
    frame = torch.from_numpy(frame)
    frame = frame.to(cons.device, dtype=torch.float32)
    frame = frame.unsqueeze(0)
    return frame


def output_video(episode, video_array, size):
    out = cv2.VideoWriter("videos/" + cons.default_name + "_episode-" + str(episode) + ".avi", cv2.VideoWriter_fourcc(*'DIVX'),
                          15, size)

    last_frame = video_array[len(video_array) - 1]

    for x in range(5):
        video_array.append(last_frame)

    for x in range(len(video_array)):
        out.write(video_array[x])

    out.release()


def stack_frames(stacked_frames, frame, is_new_episode, num_frames_stacked):
    if is_new_episode:

        stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(num_frames_stacked)], maxlen=4)

        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)

        stacked_state = np.stack(stacked_frames, axis=0)

    else:
        stacked_frames.append(frame)

        stacked_state = np.stack(stacked_frames, axis=0)

    return stacked_state, stacked_frames


def plot_results(rewards_total_episode):
    plt.figure(figsize=(12, 5))
    plt.title("Rewards Per Episode")
    plt.plot(rewards_total_episode, alpha=0.6, color='red')
    plt.savefig("episode_reward_" + cons.plot_name)
    plt.close()
