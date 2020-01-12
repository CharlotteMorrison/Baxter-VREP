from vrepsim import VrepSim
import torch
import td3.constants as cons

if __name__ == "__main__":
    # sim = VrepSim()

    # sim.step_right([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    print(cons.STATE_DIM.flatten().shape[0])
    print(cons.ACTION_DIM.flatten().shape)
