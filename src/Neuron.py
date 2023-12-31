import torch
from icecream import ic


class Neuron:
    def __init__(
        self,
        numIVInputs: int,
        numNeighbors: int,
        neighborWeightEnable: torch.Tensor,
        wMaxExp: int = 3,
        thetaScaleFactor: float = 0.1,
        uCaptureExp: int = 2,
        uBackoffExp: int = 2,
        uSearchExp: int = 10
    ):
        # * Parameters!
        self.wMin = 0  # This is not a true parameter
        self.wMax = 2 ** wMaxExp
        self.theta = max(
            int(self.wMax * (numIVInputs + numNeighbors) * thetaScaleFactor),
            1
        )
        self.uCapture = 1 / (2 ** uCaptureExp)
        self.uBackoff = 1 / (2 ** uBackoffExp)
        self.uSearch = 1 / (2 ** uSearchExp)

        self.numIVInputs = numIVInputs
        self.numNeighbors = numNeighbors

        # * State variables
        self.bodyPotential = 0

        # Input vector firing times - Used for STDP
        self.IVInputFiringTimes = torch.ones(numIVInputs) * float('Inf')
        # Neighbors firing times - Used for STDP
        self.neighborFiringTimes = torch.ones(numNeighbors) * float('Inf')
        # Firing time of the neuron
        self.firingTime = float('Inf')

        # * Weights
        # IV input weights
        self.IVInputWeights = torch.randint(
            self.wMin,
            self.wMax,
            (numIVInputs,)
        )
        # Neighbor weights
        self.neighborWeights = torch.randint(
            self.wMin,
            self.wMax,
            (numNeighbors,)
        )
        # Only enable the weights specified in the seed adjacency matrix
        self.neighborWeights *= neighborWeightEnable.long()

    def feedforwardIV(self, inputs: torch.Tensor, time: int) -> int:
        # If already fired, don't do anything
        if self.firingTime < float('Inf'):
            return 0

        # Update the input firing times
        self.IVInputFiringTimes = torch.where(
            (inputs == torch.ones(inputs.shape))
            & (time < self.IVInputFiringTimes),
            time,
            self.IVInputFiringTimes
        )

        # Compute the weighted inputs
        weightedInputs = inputs * self.IVInputWeights

        # Add to the body potential
        self.bodyPotential += weightedInputs.sum()

        # Check if fired
        if self.bodyPotential >= self.theta:
            self.firingTime = time
            return 1

        return 0  # No firing yet OR firing already occurred

    def feedforwardNeighbors(self, inputs: torch.Tensor, time: int) -> int:
        # If already fired, don't do anything
        if self.firingTime < float('Inf'):
            return 0

        # Update the neighbor firing times
        self.neighborFiringTimes = torch.where(
            (inputs == torch.ones(inputs.shape))
            & (time < self.neighborFiringTimes),
            time,
            self.neighborFiringTimes
        )

        # Compute the weighted inputs
        weightedInputs = inputs * self.neighborWeights

        # Add to the body potential
        self.bodyPotential += weightedInputs.sum()

        # Check if fired
        if self.bodyPotential >= self.theta:
            # Firing on the neighbor layer occurs at this timestep
            self.firingTime = time
            return 1

        return 0  # No firing yet OR firing already occurred

    def trainSTDP(self):
        # Perform training for IV inputs
        self.trainIVSTDP()
        # Perform training for neighboring inputs
        self.trainNeighborSTDP()

    def trainIVSTDP(self):
        # * Capture
        # Create the weight adjustment matrix
        captureMatrix = torch.bernoulli(
            torch.ones(self.numIVInputs) * self.uCapture
        )
        # Only capture when input firing time <= neuron firing time
        #  and the neuron fired
        captureMatrix = torch.where(
            (self.IVInputFiringTimes <= self.firingTime)
            & (self.firingTime < float('Inf')),
            captureMatrix,
            torch.zeros(self.numIVInputs)
        )

        # * Backoff
        # Create the weight adjustment matrix
        backoffMatrix = torch.bernoulli(
            torch.ones(self.numIVInputs) * self.uBackoff
        )
        # Only backoff when input firing time > neuron firing time
        backoffMatrix = torch.where(
            self.IVInputFiringTimes > self.firingTime,
            backoffMatrix,
            torch.zeros(self.numIVInputs)
        )

        # * Search
        # Create the weight adjustment matrix
        searchMatrix = torch.bernoulli(
            torch.ones(self.numIVInputs) * self.uSearch
        )
        # Only search when the input fired and the neuron didn't
        searchMatrix = torch.where(
            (self.IVInputFiringTimes < float('Inf'))
            & (self.firingTime == float('Inf')),
            searchMatrix,
            torch.zeros(self.numIVInputs)
        )

        # Apply STDP updates
        self.IVInputWeights += captureMatrix.long()
        self.IVInputWeights -= backoffMatrix.long()
        self.IVInputWeights += searchMatrix.long()

        # Clamp the weights to [self.wMin, self.wMax]
        self.IVInputWeights = torch.clamp(
            self.IVInputWeights,
            self.wMin,
            self.wMax
        )

    def trainNeighborSTDP(self):
        # * Capture
        # Create the weight adjustment matrix
        captureMatrix = torch.bernoulli(
            torch.ones(self.numNeighbors) * self.uCapture
        )
        # Only capture when input firing time <= neuron firing time
        #  and the neuron fired
        captureMatrix = torch.where(
            (self.neighborFiringTimes <= self.firingTime)
            & (self.firingTime < float('Inf')),
            captureMatrix,
            torch.zeros(self.numNeighbors)
        )

        # * Backoff
        # Create the weight adjustment matrix
        backoffMatrix = torch.bernoulli(
            torch.ones(self.numNeighbors) * self.uBackoff
        )
        # Only backoff when input firing time > neuron firing time
        backoffMatrix = torch.where(
            self.neighborFiringTimes > self.firingTime,
            backoffMatrix,
            torch.zeros(self.numNeighbors)
        )

        # * Search
        # Create the weight adjustment matrix
        searchMatrix = torch.bernoulli(
            torch.ones(self.numNeighbors) * self.uSearch
        )
        # Only search when the input fired and the neuron didn't
        searchMatrix = torch.where(
            (self.neighborFiringTimes < float('Inf'))
            & (self.firingTime == float('Inf')),
            searchMatrix,
            torch.zeros(self.numNeighbors)
        )

        # Apply STDP updates
        self.neighborWeights += captureMatrix.long()
        self.neighborWeights -= backoffMatrix.long()
        self.neighborWeights += searchMatrix.long()

        # Clamp the weights to [self.wMin, self.wMax]
        self.neighborWeights = torch.clamp(
            self.neighborWeights,
            self.wMin,
            self.wMax
        )

    def resetForNextQuanta(self):
        # Reset the body potential
        self.bodyPotential = 0

        # Reset the input firing times
        self.inputFiringTimes = torch.ones(self.numIVInputs) * float('Inf')
        # Reset the neighbor firing times
        self.neighborFiringTimes = torch.ones(self.numNeighbors) * float('Inf')
        # Reset the firing time
        self.firingTime = float('Inf')
