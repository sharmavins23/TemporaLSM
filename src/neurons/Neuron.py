# Definition for an abstract neuron
from abc import ABC, abstractmethod

import torch

from src.neurons.NeuronTypes import NeuronType


class Neuron:
    @abstractmethod
    def __init__(self, threshold: int, inputVectorSize: int):
        self.threshold = threshold

    @abstractmethod
    def __call__(self, inputVector: torch.Tensor) -> bool:
        pass

    @abstractmethod
    def resetForNextQuanta(self):
        pass

    @abstractmethod
    def getType(self) -> NeuronType:
        pass
