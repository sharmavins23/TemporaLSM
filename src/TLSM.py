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
        inputSize: int,
        seedMatrix: torch.Tensor,
        neuronType: NeuronType,
        threshold: int,
        quantaMax: int,
        outputSize: int,
    ):
        """
        Initializes the TLSM structure and readies for execution.

        device: CPU or GPU (for PyTorch)
        inputSize: Number of input spike lines to the network
        seedMatrix: Adjacency matrix for initial connectivity
        neuronType: Type of neuron to use in the network
        quantaMax: Max time the LSM has to process a single input
        outputSize: Number of output spike line classifications
        """

        # Save # of neurons
        self.n = seedMatrix.shape[0]

        # Create the reservoir based on neuron type
        self.reservoir = []
        if neuronType == NeuronType.SNL:
            for i in range(self.n):
                self.reservoir.append(SNLNeuron(threshold, inputSize))

        # Save the last time each neuron fired in the quanta
        self.lastTimeFired = [float(inf)] * self.n
