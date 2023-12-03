# Entrypoint for the project

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from icecream import ic

from src.InputVector import InputVector
from src.seeds import barabasiAlbert, erdosRenyi, fullyConnected, wattsStrogatz
from src.TemporalMC import TemporalMC
from src.TLSM import TLSM

# ===== Helpers ================================================================


# Print GPU information
def printGPUInformation() -> bool:
    isTorchAvailable: bool = torch.cuda.is_available()
    if isTorchAvailable:
        print(
            f'GPU: {torch.cuda.get_device_name(0)} (running on {torch.cuda.device(0)})')
    else:
        print('ERR: No GPU available, using CPU instead.')

    return isTorchAvailable


# ===== Driver code ============================================================


def main():
    printGPUInformation()

    # * Parameters
    # Meta
    n = 10  # Number of training samples
    # Liquid
    liquidCount = 150  # Number of liquid neurons
    tMax = 20  # Maximal time for the liquid to 'settle'
    # Neuron
    wMaxExp = 3  # Exponent for the maximum weight
    thetaScaleFactor = 0.1  # Scale factor for the theta (spiking cutoff) value
    uCaptureExp = 2  # Exponent for the capture time
    uBackoffExp = 7  # Exponent for the backoff time
    uSearchExp = 10  # Exponent for the search time
    # Temporal Mini-Column
    outputCount = 10  # Number of output neurons (from data)
    outputTimeRes = 10  # Output resolution
    # Seed Function
    seedFunc = fullyConnected
    seedName = "fullyConnected"

    # Filepath creation - For data and analysis
    filepath = f'results/n={n}/seed={seedName}/stdpParams=C{uCaptureExp}B{uBackoffExp}S{uSearchExp}/tMax={tMax}/'
    # Create this directory if it doesn't exist
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    # Create a new input vector
    iv = InputVector(outputTimeRes)
    # Create a new TLSM
    tlsm = TLSM(
        seedFunc(liquidCount),
        liquidCount,
        wMaxExp,
        thetaScaleFactor,
        uCaptureExp,
        uBackoffExp,
        uSearchExp
    )
    # Create a temporal mini-column
    tmc = TemporalMC(
        liquidCount,
        outputCount,
        wMaxExp,
        thetaScaleFactor,
        uCaptureExp,
        uBackoffExp,
        uSearchExp
    )

    # Save correctness values
    correctnessValues = []
    correct = 0
    total = 0

    # Print increments - For progress
    printIncrement = max(1, int(n / 10))

    # Iterate through the samples
    for i in range(n):
        if i % printIncrement == 0:
            ic(i)

        total += 1

        # Generate a new input vector
        inputVector, label = iv.getInputVector()

        # Pass it into the TLSM
        tlsmFirings = tlsm.feedforward(inputVector, tMax)

        # Iterate through these vectors and pass them into the TMC
        for t in range(tMax):
            # Get the current firing vector
            firingVector = tlsmFirings[:, t]

            # Pass it into the TMC
            tmc.feedforward(firingVector, t)

        # Train!
        tlsm.train()

        # Get which index fired
        tmcOutput = tmc.columnOutput
        firingIndex = int(tmcOutput.argmax())
        matches = firingIndex == label
        tmc.trainSTDP(reward=True, matches=matches)

        # Reset for the next quanta
        tlsm.resetForNextQuanta()
        tmc.resetForNextQuanta()

        # Save the correctness values for plotting
        if matches:
            correct += 1
        correctnessValues.append(correct / total)

    # Plot the correctness values over time
    plt.plot(correctnessValues)
    plt.xlabel('Training Sample')
    plt.ylabel('Correctness')
    plt.title('Correctness over time')
    plt.savefig(filepath + 'correctness.png')
    plt.close()

    # Iterate through the neurons and make an adjacency matrix CSV
    adjacencyMatrix = torch.zeros((liquidCount, liquidCount))
    for i in range(liquidCount):
        for j in range(liquidCount):
            adjacencyMatrix[i][j] = tlsm.neurons[i].neighborWeights[j]

    # Export to CSV
    np.savetxt(
        filepath + 'adjacencyMatrix.csv',
        adjacencyMatrix.numpy(),
        delimiter=','
    )

    # Also export an image with the weights
    plt.imshow(adjacencyMatrix)
    plt.colorbar()
    plt.savefig(filepath + 'adjacencyMatrix.png')
    plt.close()


if __name__ == '__main__':
    main()
