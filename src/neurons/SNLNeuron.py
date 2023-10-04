# Class definition for a Step No-Leak neuron
from src.neurons.Neuron import Neuron
from src.neurons.NeuronTypes import NeuronType


class SNLNeuron(Neuron):
    def __init__(self, threshold: int, inputSize: int):
        self.threshold = threshold

        # Body potential starts at 0 for the neuron
        self.bodyPotential = 0

        # Create initial weights for the neuron
        # TODO: Default to 0 for now - Should probably be changed later
        self.inputWeights = torch.ones(inputSize)

    def __call__(self, inputVector: torch.Tensor) -> bool:
        """
        Calling a neuron performs a single time step of the neuron, including
        any potential firing and STDP updating (if required).
        """

        # Start with input potentials

    def getType(self) -> NeuronType:
        return NeuronType.SNL
