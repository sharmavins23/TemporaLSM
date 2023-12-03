import torch
from icecream import ic


# Implementation for a temporal mini-column
class TemporalMC:
    def __init__(
        self,
        numInputs: int,
        numCells: int,
        wMaxExp: int = 3,
        thetaScaleFactor: float = 0.1,
        uCaptureExp: int = 2,
        uBackoffExp: int = 2,
        uSearchExp: int = 10
    ):
        self.wMin = 0
        self.wMax = 2 ** wMaxExp
        self.theta = max(
            int(self.wMax * numInputs * thetaScaleFactor),
            1
        )
        self.uCapture = 1 / (2 ** uCaptureExp)
        self.uBackoff = 1 / (2 ** uBackoffExp)
        self.uSearch = 1 / (2 ** uSearchExp)
        self.numInputs = numInputs
        self.numCells = numCells

        # * State variables
        self.bodyPotentials = torch.zeros(numCells)

        # Input vector firing times - Used for STDP
        self.inputFiringTimes = torch.ones(numInputs) * float('Inf')
        # Firing time of the column
        self.columnFiredTime = float('Inf')
        self.columnOutput = torch.zeros(numCells)

        # * Weights
        # Create an numInputs x numCells matrix of weights
        # Random values between wmin and wmax
        self.weights = torch.randint(
            self.wMin, self.wMax, (numInputs, numCells))

        # # Create an numInputs x numCells matrix of weights
        # # Random values between wmin and wmax
        # self.weights = torch.randint(
        #     self.wMin, self.wMax, (numInputs, numCells))

        # # Initialize body potentials to 0
        # self.bodyPotentials = torch.zeros(numCells)

        # # Initialize dendritic (incoming) potentials
        # # These are yet to enter the cell body and are used for 'ramping'
        # self.dendritePotentials = torch.zeros(numCells, 2)  # 2 for min time

        # # Stores firing times of the inputs - Used for STDP
        # self.inputFiringTimes = torch.ones(numInputs) * float('Inf')

        # # Stores the time in this quanta
        # self.time = 0
        # self.columnFiredTime = float('Inf')
        # self.columnOutput = torch.zeros(numCells)

    def feedforward(self, inputs: torch.Tensor, time: int) -> torch.Tensor:
        # If already fired, don't do anything
        if self.columnFiredTime < float('Inf'):
            return torch.zeros(self.numCells)

        # Update the input firing times
        self.inputFiringTimes = torch.where(
            (inputs == torch.ones(inputs.shape))
            & (time < self.inputFiringTimes),
            time,
            self.inputFiringTimes
        )

        # For each neuron...
        for i in range(self.numCells):
            # Get the weighted inputs
            weightedInputs = inputs * self.weights[:, i]

            # Add to the body potential
            self.bodyPotentials[i] += weightedInputs.sum()

        # Check if any firings occurred
        firings = torch.where(self.bodyPotentials >= self.theta, 1, 0)
        if torch.any(firings):
            self.columnFiredTime = time
            self.columnOutput = firings

        return firings

    def trainSTDP(self, reward: bool = False, matches: bool = False):
        # Iterate through all neurons
        for i in range(self.numCells):
            # * Capture
            # Create the weight adjustment matrix
            captureMatrix = torch.bernoulli(
                torch.ones(self.numInputs) * self.uCapture
            )
            captureMatrix = torch.where(
                (self.inputFiringTimes <= self.columnFiredTime)
                & (self.columnFiredTime < float('Inf')),
                captureMatrix,
                torch.zeros(self.numInputs)
            )

            # * Backoff
            # Create the weight adjustment matrix
            backoffMatrix = torch.bernoulli(
                torch.ones(self.numInputs) * self.uBackoff
            )
            backoffMatrix = torch.where(
                self.inputFiringTimes > self.columnFiredTime,
                backoffMatrix,
                torch.zeros(self.numInputs)
            )

            # * Search
            # Create the weight adjustment matrix
            searchMatrix = torch.bernoulli(
                torch.ones(self.numInputs) * self.uSearch
            )
            searchMatrix = torch.where(
                (self.inputFiringTimes < float('Inf'))
                & (self.columnFiredTime == float('Inf')),
                searchMatrix,
                torch.zeros(self.numInputs)
            )

            # * STDP Updates
            if reward:
                if matches:
                    # Apply capture and backoff as normal
                    self.weights[:, i] += captureMatrix.long()
                    self.weights[:, i] -= backoffMatrix.long()
                else:
                    # Apply -capture, no backoff, and search
                    self.weights[:, i] -= captureMatrix.long()
                    self.weights[:, i] += searchMatrix.long()
            else:
                # Apply capture, backoff, and search
                self.weights[:, i] += captureMatrix.long()
                self.weights[:, i] -= backoffMatrix.long()
                self.weights[:, i] += searchMatrix.long()

    def resetForNextQuanta(self):
        # Reset the column firing time
        self.columnFiredTime = float('Inf')
        self.columnOutput = torch.zeros(self.numCells)
        # Reset the body potentials
        self.bodyPotentials = torch.zeros(self.numCells)
        # Reset the input firing times
        self.inputFiringTimes = torch.ones(self.numInputs) * float('Inf')
