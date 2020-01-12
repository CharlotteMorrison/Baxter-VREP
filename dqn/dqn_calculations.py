import numpy as np
from dqn import dqn_constants as cons
import math


def calculate_temperature(num_frames):
    temp = -cons.decay_temperature * num_frames
    temp = np.exp(temp)
    temp = temp * cons.max_temperature + 1
    return temp


def calculate_epsilon(frames):
    epsilon = cons.egreedy_final + (cons.egreedy - cons.egreedy_final) * \
              math.exp(-1. * frames / cons.egreedy_decay)
    return epsilon
