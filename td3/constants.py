import torch

# flags
set_seed = True

DEFAULT_NAME = "Baxter_TD3"
model_file = "td3/results/models/" + DEFAULT_NAME + "_model.pth"
FILE_NAME = "td3/results/" + DEFAULT_NAME + "_results"
PLOT_NAME = "td3/results/plots/" + DEFAULT_NAME + "_results.png"

# Program run constants
SEED = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_frames = 300000
VIDEO_INTERVAL = 1  # change to 1 to record all videos
NUM_FRAMES_STACKED = 4
SOLVED_DISTANCE = 0.12
WRITE_TO_FILE = True
REPORT_INTERVAL = 1
SIZE = (512, 512)


# TD3 hyperparameters from addressing function approx. err paper
EXPLORATION = 5000000
OBSERVATION = 10000
EXPLORE_NOISE = 0.1
REWARD_THRESH = 1.95
BATCH_SIZE = 100
BUFFER_SIZE = 500000   # reduced from 1000000 for laptop,  gpu is failing at 1,000000, reduced by half
GAMMA = 0.99  # discount
TAU = 0.005
POLICY_NOISE = 0.2  # adjusted from .2, due to scale of movement
NOISE_CLIP = 0.5    # adjusted from .5, due to scale of movement
POLICY_FREQ = 2

# environment parameters
STATE_DIM = torch.empty(84, 84)
MAX_ACTION = 0.1
MIN_ACTION = -0.1
ACTION_DIM = 7





