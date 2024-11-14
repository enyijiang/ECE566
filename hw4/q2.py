import numpy as np
from itertools import product

# Define the number of nodes and possible binary states
nodes = 5
states = [0, 1]

# Define the compatibility functions between each pair of variables as given
# Since xor (mod-2 addition) is defined as (a XOR b), we can use 1 - (a == b) to define it
def compatibility(x_i, x_j, positive=True):
    if positive:
        return np.exp((x_i != x_j))  # Positive interaction
    else:
        return np.exp(-(x_i != x_j))  # Negative interaction

# Define edges with positive and negative interactions as described
edges = [
    (0, 1, False),  # X1 and X2 with - sign (negative interaction)
    (0, 2, False),  # X1 and X3 with - sign
    (1, 2, False),  # X2 and X3 with - sign
    (1, 3, False),  # X2 and X4 with - sign
    (2, 4, True),   # X3 and X5 with + sign
    (3, 4, True)    # X4 and X5 with + sign
]

# Initialize messages as a dictionary where each key is a tuple (i, j) representing an edge
# Each message is a numpy array with two elements (for states 0 and 1)
messages = {(i, j): np.ones(2) for i, j, _ in edges}
messages.update({(j, i): np.ones(2) for i, j, _ in edges})

# Max-product belief propagation for (c)
def max_product_belief_propagation(iterations=10):
    for _ in range(iterations):
        new_messages = {}
        for i, j, positive in edges:
            for x_j in states:
                max_val = -np.inf
                for x_i in states:
                    product_val = compatibility(x_i, x_j, positive) * np.prod(
                        [messages[(k, i)][x_i] for k, _, _ in edges if k != j and (k, i) in messages]
                    )
                    max_val = max(max_val, product_val)
                new_messages[(i, j)] = np.array([max_val if x == x_j else 1 for x in states])

    # Calculate beliefs for each node
    beliefs = {}
    for i in range(nodes):
        beliefs[i] = np.ones(2)
        for j in range(nodes):
            if (j, i) in new_messages:
                beliefs[i] *= new_messages[(j, i)]
    
    # Get most likely configuration by choosing the state with maximum belief for each node
    most_likely_x = [np.argmax(beliefs[i]) for i in range(nodes)]
    return most_likely_x

# Sum-product belief propagation to calculate beliefs
def sum_product_belief_propagation(iterations=10):
    for _ in range(iterations):
        new_messages = {}
        for i, j, positive in edges:
            for x_j in states:
                sum_val = 0
                for x_i in states:
                    product_val = compatibility(x_i, x_j, positive) * np.prod(
                        [messages[(k, i)][x_i] for k, _, _ in edges if k != j and (k, i) in messages]
                    )
                    sum_val += product_val
                new_messages[(i, j)] = np.array([sum_val if x == x_j else 1 for x in states])

    # Calculate beliefs for each node
    beliefs = {}
    for i in range(nodes):
        beliefs[i] = np.ones(2)
        for j in range(nodes):
            if (j, i) in new_messages:
                beliefs[i] *= new_messages[(j, i)]
        beliefs[i] /= beliefs[i].sum()  # Normalize
    
    return beliefs

# Approximate partition function Z by combining beliefs
def approximate_partition_function(beliefs):
    Z = 1.0

    # Multiply beliefs for each node
    for i in range(nodes):
        Z *= beliefs[i].sum()  # Sum over both states (0 and 1)

    # Divide by edge terms to correct for over-counting
    for i, j, positive in edges:
        edge_sum = sum(
            compatibility(x_i, x_j, positive) * beliefs[i][x_i] * beliefs[j][x_j]
            for x_i in states for x_j in states
        )
        Z /= edge_sum  # Divide by the edge belief product
    
    return Z

# Run max-product for most likely configuration
most_likely_x = max_product_belief_propagation()
print("Most likely configuration (c):", most_likely_x)

beliefs = sum_product_belief_propagation()
print("Z", approximate_partition_function(beliefs))