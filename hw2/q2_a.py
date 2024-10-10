import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
n_iterations = 1000
X0 = 1  # Initial value X_0 = 1
f_prime_means = [lambda x: np.sign(x)*(np.abs(x)**0.5), lambda x: 2*x, lambda x: 3*(x**3)]
variance = 1

# Step size schedules
def step_size_1(k):
    return k**(-0.6)

def step_size_2(k):
    return k**(-1)

# SGD function
def stochastic_gradient_descent(f_prime_mean, step_size_func, n_iterations=1000):
    X = np.zeros(n_iterations)
    X[0] = X0
    
    for k in range(1, n_iterations):
        # if k >= 2 and np.abs(X[k-1] - X[k-2]) < 1e-3:
            # break
        Y =  f_prime_mean(X[k-1]) + np.random.normal(0, np.sqrt(variance))  # Noisy gradient
        a_k = step_size_func(k)
        X[k] = X[k-1] - a_k * Y
    
    return X

# Run SGD for both step size schedules
for i in range(len(f_prime_means)):
    f_prime_mean = f_prime_means[i]
    X_k_1 = stochastic_gradient_descent(f_prime_mean, step_size_1, n_iterations)
    X_k_2 = stochastic_gradient_descent(f_prime_mean, step_size_2, n_iterations)

    # Plot sample path of SGD
    plt.figure(figsize=(10, 6))
    plt.plot(X_k_1, label='Step size: k^(-0.6)')
    plt.plot(X_k_2, label='Step size: k^(-1)')
    plt.title('Sample Path of SGD')
    plt.xlabel('Iteration')
    plt.ylabel('$X_k$')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'q2/q2_a_sample_path_{i}.pdf')

    # Plot |X_k| vs ln(k)
    k_values = np.arange(1, n_iterations+1)
    ln_k = np.log(k_values)

    plt.figure(figsize=(10, 6))
    plt.plot(ln_k, np.abs(X_k_1), label='Step size: k^(-0.6)')
    plt.plot(ln_k, np.abs(X_k_2), label='Step size: k^(-1)')
    plt.title('$|X_k|$ vs $\\ln(k)$')
    plt.xlabel('$\\ln(k)$')
    plt.ylabel('$|X_k|$')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'q2/q2_a_seq_ln_{i}.pdf')