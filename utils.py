import torch
import cv2
from collections import deque
import numpy as np
import matplotlib.pyplot as plt


def load_model(model_file):
    return torch.load(model_file)


def save_model(model, model_file):
    torch.save(model.state_dict(), model_file)


def preprocess_frame(frame, device):
    frame = torch.from_numpy(frame)
    frame = frame.to(device, dtype=torch.float32)
    frame = frame.unsqueeze(0)
    return frame


def output_video(episode, video_array, size, default_name):
    out = cv2.VideoWriter("videos/" + default_name + "_episode-" + str(episode) + ".avi",
                          cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

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


def plot_results(rewards_total_episode, plot_name):
    plt.figure(figsize=(12, 5))
    plt.title("Rewards Per Episode")
    plt.plot(rewards_total_episode, alpha=0.6, color='red')
    plt.savefig(plot_name)
    plt.close()


def plot_loss(actor_loss, critic_loss, plot_name):
    plt.figure(figsize=(12, 5))
    plt.title("Loss Per Episode")
    plt.ylim(top=1)
    plt.xlabel('Episode Number')
    plt.ylabel('Average Loss Per Episode')
    plt.plot(actor_loss, alpha=0.6, color='blue')
    plt.plot(critic_loss, alpha=0.6, color='green')
    plt.savefig(plot_name)
    plt.close()

# dHash: distance based hashing, used for feature reduction
# Hashes of similar images are close in numerical value.
# from: pyimagesearch.com/2017/11/27/image-hashing-opencv-python


def d_hash(image, hash_size=8):
    # convert the image to black and white
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # resize the input image adding a single column (width)
    # so the horizontal gradient can be computed
    resized = cv2.resize(image, (hash_size + 1, hash_size))

    # compute the relative horizontal gradient between
    # adjacent column pixels
    diff = resized[:, 1:] > resized[:, :-1]

    img_hash = sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])
    return np.asarray([int(x) for x in str(img_hash)]).astype(float)
