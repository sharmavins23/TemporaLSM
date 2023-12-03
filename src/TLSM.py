import torch
from icecream import ic

from src.Neuron import Neuron


class TLSM:
    def __init__(
        self,
        seedAdjacencyMatrix: torch.Tensor,
        liquidCount: int = 150,
        wMaxExp: int = 3,
        thetaScaleFactor: float = 0.1,
        uCaptureExp: int = 2,
        uBackoffExp: int = 2,
        uSearchExp: int = 10
    ):
        # * Parameters!
        self.neuronCount = liquidCount

        # Create a list of Neurons
        self.neurons: list[Neuron] = []
        for i in range(self.neuronCount):
            self.neurons.append(Neuron(
                784,
                self.neuronCount,
                seedAdjacencyMatrix[i],
                wMaxExp,
                thetaScaleFactor,
                uCaptureExp,
                uBackoffExp,
                uSearchExp
            ))

    def feedforward(self, inputs: torch.Tensor, tMax: int) -> torch.Tensor:
        # Ensure the maximal time is greater than the input time size
        assert (tMax > inputs.shape[1])

        timeVariantNeuronFirings = torch.zeros(self.neuronCount, tMax)

        for time in range(tMax):
            # Get the input vector at this time, if it exists
            if time < inputs.shape[1]:
                inputVector = inputs[:, time]
            else:
                inputVector = torch.zeros(inputs.shape[0])

            # Feed in the input vector and save the neuron's firings
            neuronFirings = torch.zeros(self.neuronCount)
            for i in range(self.neuronCount):
                neuronFirings[i] = self.neurons[i].feedforwardIV(
                    inputVector,
                    time
                )

            # Feed all these firings to the neighbors
            neighborNeuronFirings = torch.zeros(self.neuronCount)
            for i in range(self.neuronCount):
                neighborNeuronFirings[i] = self.neurons[i].feedforwardNeighbors(
                    neuronFirings,
                    time
                )

            # Combine these firings together
            combinedNeuronFirings = torch.where(
                (neuronFirings == 1)
                | (neighborNeuronFirings == 1),
                1,
                0
            )
            # Save this to the time variant neuron firings
            timeVariantNeuronFirings[:, time] = combinedNeuronFirings

            # Update the time
            time += 1

        return timeVariantNeuronFirings

    def train(self):
        # Iterate through the neurons and train them through STDP
        for i in range(self.neuronCount):
            self.neurons[i].trainSTDP()

    def resetForNextQuanta(self):
        # Iterate through the neurons and reset them
        for i in range(self.neuronCount):
            self.neurons[i].resetForNextQuanta()
