# Class definition for the temporal liquid state machine (TLSM) structure
from typing import List

import torch

from src.neurons.Neuron import Neuron
from src.neurons.NeuronTypes import NeuronType
from src.neurons.SNLNeuron import SNLNeuron


class TemporalLiquidStateMachine:

    def __init__(
        self,
        device: torch.device,
        inputVectorSize: int,
        seedMatrix: torch.Tensor,
        neuronType: NeuronType,
        threshold: int,
        quantaMax: int,
        allowLinkage: bool,
        outputSize: int,
    ):
        """
        Initializes the TLSM structure and readies for execution.

        device: CPU or GPU (for PyTorch)
        inputVectorSize: Number of input spike lines to the network
        seedMatrix: Adjacency matrix for initial connectivity
        neuronType: Type of neuron to use in the network
        quantaMax: Max time the LSM has to process a single input
        allowLinkage: Allow the network to grow new connections
        outputSize: Number of output spike line classifications
        """

        # Save # of neurons
        self.n = seedMatrix.shape[0]

        # Create the reservoir based on neuron type
        self.reservoir = []
        if neuronType == NeuronType.SNL:
            for i in range(self.n):
                self.reservoir.append(SNLNeuron(threshold, inputVectorSize))

        # Save the last time each neuron fired in the quanta
        # This is a tensor of size n with all float('inf') values
        self.reservoirLastFirings = torch.full((self.n,), float('inf'))

        # Default the weight matrix to the seed matrix
        self.weightMatrix = seedMatrix

        # Save the maximal quanta value
        self.quantaMax = quantaMax

    def __call__(
        self,
        inputVector: torch.Tensor,
        supervised: bool,
    ):
        """
        Calling the TLSM performs a single time quanta of the network, handling
        the (online) supervised/unsupervised training of an entire input.

        inputVector: Input spike vector for the current time quanta; Encoded in
            time delta scale
        supervised: Network is being trained with expected outputs
        """

        # Iterate from 0 to quantaMax
        for currentTime in range(self.quantaMax):
            # Iterate over each neuron in the reservoir
            for i in range(self.n):
                # Transform the input vector into time dirac scale
                diracInputVector = torch.where(
                    inputVector == currentTime,
                    1,
                    0
                )

                # Call the neuron with the input vector
                self.reservoir[i](currentTime, diracInputVector)
