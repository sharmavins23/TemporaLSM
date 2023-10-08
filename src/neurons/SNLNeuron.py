# Class definition for a Step No-Leak neuron
from src.neurons.Neuron import Neuron
from src.neurons.NeuronTypes import NeuronType
from src.STDPParams import STDPParams


class SNLNeuron(Neuron):

    def __init__(
        self,
        threshold: int,
        weightMax: int,
        inputVectorSize: int,
        nAdjacent: int,
        stdpParams: STDPParams
    ):
        """
        Initializes a Step No-Leak (SNL) neuron.

        threshold: Threshold potential for the neuron to fire
        weightMax: Maximum weight for any synapse
        inputVectorSize: Number of input spike lines (based on IV size)
        nAdjacent: Total number of neurons in the liquid reservoir
        stdpParams: Probability parameters for STDP training
        """

        self.threshold = threshold
        # Body potential starts at 0 for the neuron
        self.bodyPotential = 0

        self.weightMax = weightMax

        # Create initial weights for the neuron
        # TODO: Default to 1 for now - Should probably be changed later
        self.inputVectorSize = inputVectorSize
        self.inputVectorWeights = torch.ones(inputVectorSize)
        self.inputVectorLastFirings = torch.full(
            (inputVectorSize,), float('inf'))

        # Create adjacent weights for the adjacent neurons
        self.nAdjacent = nAdjacent
        self.adjacentWeights = torch.zeros(nAdjacent)
        self.adjacentLastFirings = torch.full((nAdjacent,), float('inf'))

        # Set the last time the neuron fired to infinity as well
        self.lastFiring = float('inf')

        # Save STDP training parameters
        self.stdpParams = stdpParams

    def __call__(
        self,
        currentTime: int,
        inputVector: torch.Tensor,
        adjacentVector: torch.Tensor
    ) -> bool:
        """
        Calling a neuron performs a single time step of the neuron, including
        any potential firing and STDP updating (if required). The function also
        returns whether it fired or not.

        currentTime: The current time (offset from 0) in the current quanta
        inputVector: Input spike vector for the current time step; Encoded in
            time dirac scale
        adjacentVector: Adjacent spike vector for the current time step; Encoded
            in time dirac scale
        """

        # * Input vector handling
        # Sum up any and all weights
        self.bodyPotential += torch.sum(self.inputVectorWeights * inputVector)
        # Set the last firing time if the input vector fired
        self.inputVectorLastFirings[inputVector == 1] = currentTime

        # * Adjacent neuron handling
        # Sum up any and all weights
        self.bodyPotential += torch.sum(self.adjacentWeights * adjacentVector)
        # Set the last firing time if the adjacent vector fired
        self.adjacentLastFirings[adjacentVector == 1] = currentTime

        fired = False
        if self.bodyPotential >= self.threshold:
            fired = True
            self.lastFiring = currentTime

        # * Handle STDP training
        if fired:
            self.trainWeights()

        return fired

    def trainWeights(
        self,
        currentTime: int,
    ):
        """
        Trains any weights of neurons through STDP training rules. This process
        is completely unsupervised. The function may only be called when the
        neuron spikes.
        """

        # * Capture
        # Capture any spikes that fired in the input vector before now
        captureAddVectorIV = torch.bernoulli(torch.where(
            self.inputVectorLastFirings <= currentTime,  # If the spike fired before now
            self.stdpParams.capture,  # Capture it with probability p
            0  # Otherwise, don't capture it
        ))
        self.inputVectorWeights += torch.round(captureAddVectorIV)
        captureAddVectorAdj = torch.bernoulli(torch.where(
            self.adjacentLastFirings <= currentTime,  # If the spike fired before now
            self.stdpParams.capture,  # Capture it with probability p
            0  # Otherwise, don't capture it
        ))
        self.adjacentWeights += torch.round(captureAddVectorAdj)

        # * Backoff
        # Back off any spikes that fired in the input vector after now
        backoffSubVectorIV = torch.bernoulli(torch.where(
            self.inputVectorLastFirings > currentTime,  # If the spike fired after now
            self.stdpParams.backoff,  # Back off with probability p
            0  # Otherwise, don't back off
        ))
        self.inputVectorWeights -= torch.round(backoffSubVectorIV)
        backoffSubVectorAdj = torch.bernoulli(torch.where(
            self.adjacentLastFirings > currentTime,  # If the spike fired after now
            self.stdpParams.backoff,  # Back off with probability p
            0  # Otherwise, don't back off
        ))
        self.adjacentWeights -= torch.round(backoffSubVectorAdj)

        # Clamp weights between 0 and weightMax
        self.inputVectorWeights = torch.clamp(
            self.inputVectorWeights,
            0,
            self.weightMax
        )
        self.adjacentWeights = torch.clamp(
            self.adjacentWeights,
            0,
            self.weightMax
        )

        # Finally, 'clean' all firing times if the neuron fired
        self.inputVectorLastFirings = torch.full(
            (self.inputVectorSize,), float('inf'))
        self.adjacentLastFirings = torch.full((self.nAdjacent,), float('inf'))

    def resetForNextQuanta(self):
        """
        Reset any data related to a neuron for the next time quanta. STDP search
        is also performed here.
        """

        # * STDP search
        # Search for a spike by bubbling up all weights by a small increment
        searchAddVectorIV = torch.bernoulli(
            torch.full(
                (self.inputVectorWeights.shape),
                self.stdpParams.search
            )
        )
        self.inputVectorWeights += torch.round(searchAddVector)
        searchAddVectorAdj = torch.bernoulli(
            torch.full(
                (self.adjacentWeights.shape),
                self.stdpParams.search
            )
        )
        self.adjacentWeights += torch.round(searchAddVectorAdj)

        self.inputVectorWeights = torch.clamp(
            self.inputVectorWeights,
            0,
            self.weightMax
        )
        self.adjacentWeights = torch.clamp(
            self.adjacentWeights,
            0,
            self.weightMax
        )

        # * Neuron resetting

        # Body potential should go to zero
        self.bodyPotential = 0

        # The neuron hasn't fired for the next quanta
        self.lastFiring = float('inf')

        # Reset the last firing times to infinity
        self.inputVectorLastFirings = torch.full(
            (self.inputVectorSize,), float('inf'))
        self.adjacentLastFirings = torch.full((self.nAdjacent,), float('inf'))

    def getType(self) -> NeuronType:
        return NeuronType.SNL
