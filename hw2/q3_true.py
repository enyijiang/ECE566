import numpy as np
from scipy.stats import norm

# Define the function I(a) as given in the formula
def I_a(a):
    # Q(0) and Q(1) using the Gaussian survival function (1 - CDF)
    Q_0 = norm.sf(0)  # Q(0)
    Q_1 = norm.sf((2*a)**0.5)  # Q(1)

    # Compute the term inside the parentheses
    term = np.sqrt(np.pi / a) * (Q_0 - Q_1)
    
    # Raise the term to the power of 10
    result = term ** 10
    
    return result

# Compute I(a) for a = 0.1 and a = 10
a_values = [0.1, 10]
for a in a_values:
    result = I_a(a)
    print(f"I({a}) = {result}")