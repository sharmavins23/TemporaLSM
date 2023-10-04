# Contains functions for generating seed matrices
import torch

# Notes on creating good adjacency matrices:
# - All matrices are square, and have a hyper-parameter of n nodes
# - All matrices should be binarized to 1 or 0 values only
# - Matrices may be asymmetric as they are representative of directed graphs;
#    Symmetry may harm network functionality
# - Adjacency matrices are not allowed to have autaptic loops (1s on the main
#    diagonal), as by STDP update logic, the synapses would train to overpower
#    all other synapses


# ===== Matrix generators ======================================================


def fullyConnected(n: int) -> torch.Tensor:
    """
    Creates a fully connected graph.

    n: Number of nodes
    """
    assert (0 < n)

    return torch.ones((n, n)) - torch.eye(n)


def erdosRenyi(n: int, p: float) -> torch.Tensor:
    """
    Creates an Erdos-Renyi random graph.

    n: Number of nodes
    p: Probability of an edge between any two nodes
    """
    assert (0 <= p <= 1)
    assert (0 < n)

    # Potential matrix may have autaptic loops
    potentialMatrix: torch.Tensor = torch.bernoulli(torch.full((n, n), p))
    # Remove any autaptic loops while keeping values in {0, 1}
    erdosRenyi = potentialMatrix - (torch.eye(n) * potentialMatrix)
    return erdosRenyi


# ===== Tester functions =======================================================


def sanityCheckSeedMatrix(seedMatrix: torch.Tensor) -> bool:
    """
    Sanity checks a seed matrix to ensure it is of valid structure. Use this
     function to ensure a seed matrix is valid for construction of a network.
    """

    # Check that the matrix is square
    if seedMatrix.shape[0] != seedMatrix.shape[1]:
        return False

    # Check that the main diagonal is all 0s
    if torch.any(torch.diag(seedMatrix) != 0):
        return False

    # Check that all values are either 0 or 1
    if torch.any((seedMatrix != 0) & (seedMatrix != 1)):
        return False

    return True
