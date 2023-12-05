import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from icecream import ic

random.seed(23)

for n in [10, 100, 1000]:  # Training samples
    k = 50
    p = 0.75
    folder = f"results/n={n}/seed=wattsStrogatzk{k}p{p}/"
    graphName = "Watts-Strogatz (k={k}, p={p})"
    # Load the adjacencyMatrix.csv as a networkX graph
    adjacencyMatrix = np.loadtxt(folder + "adjacencyMatrix.csv", delimiter=",")
    G = nx.from_numpy_array(adjacencyMatrix)
    # Remove any weight=0 edges
    G.remove_edges_from([(u, v)
                        for u, v, d in G.edges(data=True) if d['weight'] == 0])
    # Draw the graph
    nx.draw(G, with_labels=True)
    plt.savefig(folder + "graph.png")
    plt.close()

    # Compute clustering coefficient
    clusteringCoefficient = nx.average_clustering(G)
    ic(clusteringCoefficient)
    # Compute average shortest path length
    averageShortestPathLength = nx.average_shortest_path_length(G)
    ic(averageShortestPathLength)
    # Graph degree distribution
    degreeSequence = sorted([d for n, d in G.degree()], reverse=True)
    degreeCount = np.unique(degreeSequence, return_counts=True)
    # Plot the degree distribution
    plt.bar(degreeCount[0], degreeCount[1])
    plt.xlabel('Degree')
    plt.ylabel('Count')
    plt.title('Degree distribution')
    plt.savefig(folder + "degreeDistribution.png")
    plt.close()
    # Print average degree
    averageDegree = np.mean(degreeSequence)
    ic(averageDegree)

    # ===== Spreading model ====================================================

    # States will only be 'informed (I)' or 'uninformed (U)'
    states = {node: 'U' for node in G.nodes()}
    t = 0
    St = []
    Ut = []

    # Infect node 0 to start
    states[0] = 'I'

    # Neuron parameters - Used for spreading probabilities
    neuronCount = 150
    wMax = 2 ** 6
    theta = wMax * (784 + 150) * 0.1

    printedTime: bool = False

    while True:
        nextSetStates = states.copy()

        for node in G.nodes:
            # Iterate through all of the node's neighbors
            for neighbor in G.neighbors(node):
                # If the node is informed and the neighbor is uninformed, inform the
                #  neighbor with probability weight/theta
                if states[node] == 'I' and states[neighbor] == 'U':
                    # Get the weight
                    weight = G[node][neighbor]['weight']
                    if random.random() < (weight / theta):
                        nextSetStates[neighbor] = 'I'

        # Append the counts of each state to the lists
        St.append(len([node for node in states if states[node] == 'I']))
        Ut.append(len([node for node in states if states[node] == 'U']))

        # Run until there are no more uninformed nodes
        if Ut[-1] == 0:
            break

        # If 50% of the nodes are informed, print the time
        if St[-1] > len(states) / 2 and not printedTime:
            print(f"t = {t}")
            printedTime = True

        # Increment time
        t += 1

        # Update the states
        states = nextSetStates.copy()

    # Convert the lists to fractional values
    St = [s / len(states) for s in St]
    Ut = [u / len(states) for u in Ut]

    # Plot the results
    plt.plot(St, label='St')
    plt.plot(Ut, label='Ut')
    plt.xlabel('Time')
    plt.ylabel('Fraction of nodes')
    plt.title(
        f'LSM Information spreading w/ seed {graphName}, {n} train steps')
    plt.legend()
    plt.savefig(folder + "spreading.png")
    plt.close()
