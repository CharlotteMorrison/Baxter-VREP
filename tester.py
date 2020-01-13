from vrepsim import VrepSim
import torch
import td3.constants as cons

if __name__ == "__main__":
    sim = VrepSim()
    sim.get_min_max()
