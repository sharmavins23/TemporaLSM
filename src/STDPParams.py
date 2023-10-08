# Struct for STDP parameters


class STDPParams:
    def __init__(
        self,
        capture: float,
        backoff: float,
        search: float,
    ):
        """
        Structure containing STDP parameters. Also handles type validation.

        capture: Probability a neuron will 'capture' a spike, increasing its
            weights by 1
        backoff: Probability a neuron will 'backoff' a spike, decreasing its
            weights by 1
        search: Probability a neuron will 'search' for a spike, increasing its
            weights by 1
        """

        assert (0 <= capture <= 1)
        assert (0 <= backoff <= 1)
        assert (0 <= search <= 1)

        self.capture = capture
        self.backoff = backoff
        self.search = search
