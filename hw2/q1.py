import numpy as np
import matplotlib.pyplot as plt

# Define the gradient descent parameters
alpha = 1/8
x0 = 1
iterations = 500

# Define the function f(x) = x^4 and its gradient
def gradient(x):
    return 4 * x**3

# Initialize x and store the sequence of iterates
x = x0
sequence = [x]

# Perform gradient descent
for _ in range(iterations):
    x = x - alpha * gradient(x)
    sequence.append(x)

# Convert sequence to log scale for plotting
sequence = np.array(sequence)
log_sequence = np.log(sequence)

# Plot the sequence of iterates on a log scale
plt.figure(figsize=(8, 6))
plt.plot(np.log([i+1 for i in range(iterations+1)]), log_sequence, marker='o', label="GD Iterates (log scale)")
plt.title("Gradient Descent Iterates for $f(x) = x^4$ (log scale)")
plt.xlabel("Log of Iteration")
plt.ylabel("Log of x_k")
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig('q1.pdf')