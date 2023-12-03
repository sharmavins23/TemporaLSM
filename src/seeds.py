# Contains functions for generating seed matrices
import networkx as nx
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


def wattsStrogatz(n: int, p: float, k: int) -> torch.Tensor:
    """
    Creates a Watts-Strogatz Small World random graph.

    n: Number of nodes
    p: Probability of an edge being rewired
    k: Number of edges in base lattice
    """

    G = nx.watts_strogatz_graph(n, k, p)
    watts_strogatz = torch.from_numpy(nx.to_numpy_array(G))
    return watts_strogatz


def barabasiAlbert(n: int, m: int):
    """
    Creates a Barabasi-Albert random graph.

    n: Number of nodes
    m: Number of edges to attach from a new node to existing nodes
    """

    G = nx.barabasi_albert_graph(n, m)
    barabasi_albert = torch.from_numpy(nx.to_numpy_array(G))
    return barabasi_albert


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
