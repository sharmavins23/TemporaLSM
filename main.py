# Entrypoint for the project

import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from icecream import ic

# ===== Driver code ============================================================


def main():
    # Set up CUDA
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA for computations.")
    else:
        device = torch.device("cpu")
        print("WARN: CUDA not available - Using CPU.")

    torch.set_default_device(device)
    # Computation will probably still be fairly slow, as the GPU is not nearly
    #  saturated by any of these matrix operations

    # TODO: Run the TLSM on the dataset


if __name__ == '__main__':
    main()
