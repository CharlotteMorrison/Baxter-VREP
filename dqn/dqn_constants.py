import torch
import matplotlib.pyplot as plt

# Specify Name for Plots, Recordings, Database, & Model
default_name = "Baxter_DQN"
model_file = "dqn/results/models/" + default_name + "_model.pth"
file_name = "dqn/results/" + default_name + "_results"
plot_name = "dqn/results/plots/" + default_name + "_results.png"
database = "dqn/results/" + default_name + ".db"

# Checks if Cuda is available
use_cuda = torch.cuda.is_available()
# Sets Device for Calculations
device = torch.device("cuda:0" if use_cuda else "cpu")
# Sets if Results are written to a csv file
write_to_file = True
# Sets if Error Clipping is Used
clip_error = True
# Sets if Image is Normalized
normalize_image = True
# Sets if Seed is used
set_seed = True
# Sets if a Model should be loaded
model_load = False
# Sets if memory is saved to DB
mem_save = False
# Sets if memory is loaded from DB
mem_load = False

# Sets style for Matplotlib
plt.style.use("ggplot")

solved_distance = 0.12

# Learning rate of Neural Network
learning_rate = 0.0001
# Gamma for Neural Network Learning
gamma = 0.99
# Sets Number of Hidden Layers for Neural Network
hidden_layer = 512
# Sets Number of Stacked Frames for Input to Neural Network
num_frames_stacked = 4
# Output Size for Network - Num of possible actions
output_size = 14

# Sets Number of Training Episodes
num_frames = 300000

# Sets Size of Replay Buffer
replay_mem_size = 75000
# Sets Size of Replay Batch for Learning
batch_size = 32

# Sets if Boltzmann Selection is used
boltzmann = True
# Sets weight for boltzmann selection
boltzmann_weight = 5
# Sets max temperature for boltzmann selection
max_temperature = 1000
# Sets min temperature for boltzmann selection
min_temperature = 1
# Sets temperature decay for boltzmann selection
decay_temperature = .01

# Sets the Frequency to Update Target Network
update_target_frequency = 5000
# Sets the interval for Saving Model by Frames
save_model_frequency = 10000
# Sets the interval for Updating Plots
report_interval = 10
# Sets the interval for video recording
video_interval = 1

# Sets value of Seed
seed_value = 23

greedy = .1

# Sets egreedy Value
egreedy = 0.9
# Sets Min Value for Egreedy
egreedy_final = 0.1
# Sets Egreedy Decay
egreedy_decay = 15000

Tensor = torch.Tensor
LongTensor = torch.LongTensor
