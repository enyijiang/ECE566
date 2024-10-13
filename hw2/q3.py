import numpy as np
# from scipy.stats import truncnorm
from scipy.stats import multivariate_normal

# Define the Monte Carlo integrand for I(a)
def integrand(x, a):
    norm_squared = np.sum(x**2)
    return np.exp(-a * norm_squared)

def integrand_b(x, a):
    norm_x_squared = np.sum(x**2)
    sum_sqrt_x = np.sum(np.sqrt(x))
    return np.exp(-a * norm_x_squared) * sum_sqrt_x

# Monte Carlo estimation function for I(a) with accuracy (standard error)
def monte_carlo_estimate(a, n, dimensions):
    samples = np.random.uniform(0, 1, (n, dimensions))
    estimates = np.array([integrand(sample, a) for sample in samples])
    mean_estimate = np.mean(estimates)
    standard_error = np.std(estimates) / np.sqrt(n)
    return mean_estimate, standard_error

# Monte Carlo for part (b)
def monte_carlo_estimate_b(a, n, dimensions):
    samples = np.random.uniform(0, 1, (n, dimensions))
    estimates = np.array([integrand_b(sample, a) for sample in samples])
    mean_estimate = np.mean(estimates)
    standard_error = np.std(estimates) / np.sqrt(n)
    return mean_estimate, standard_error

# Auxiliary distribution: Multivariate normal with mean 0 and variance 1/(2*10)
def sample_from_auxiliary_distribution(n_samples, dim, sigma):
    # Sample from N(0, sigma^2) and truncate to [0, 1]
    samples = np.random.normal(0, sigma, size=(n_samples, dim))
    samples = np.clip(samples, 0, 1)  # Truncate to the unit cube
    return samples

# Density of the auxiliary distribution q(x)
def auxiliary_distribution_pdf(x, sigma):
    dim = len(x)
    # Multivariate normal PDF for each sample point
    return multivariate_normal.pdf(x, mean=np.zeros(dim), cov=sigma**2)

# Importance sampling with the auxiliary distribution
def importance_sampling_with_auxiliary(n_samples=1000, a=0.1, dim=10, sigma=np.sqrt(1/(20*10))):
    samples = sample_from_auxiliary_distribution(n_samples, dim, sigma)
    integral_estimates = []
    
    for x in samples:
        f_x = integrand_b(x, a)
        q_x = auxiliary_distribution_pdf(x, sigma)
        weight = f_x / q_x
        integral_estimates.append(weight)
    
    integral_mean = np.mean(integral_estimates)
    integral_error = np.std(integral_estimates) / np.sqrt(n_samples)
    return integral_mean, integral_error

# Define the number of samples and dimensions
n = 1000
dimensions = 10

# Estimating I(a) using Monte Carlo for a = 0.1 and a = 10
a_01 = 0.1
a_10 = 10

# Monte Carlo Estimation with accuracy
I_a_01_monte_carlo, se_a_01 = monte_carlo_estimate(a_01, n, dimensions)
I_a_10_monte_carlo, se_a_10 = monte_carlo_estimate(a_10, n, dimensions)

# I_a_01_monte_carlo_b, se_a_01_b = monte_carlo_estimate_b(a_01, n, dimensions)
I_a_10_monte_carlo_b, se_a_10_b = monte_carlo_estimate_b(a_10, n, dimensions)

# Importance Sampling Estimation with accuracy using a truncated Gaussian proposal
# I_a_01_importance_sampling, se_importance_sampling_a01 = importance_sampling_with_auxiliary(n_samples=n, a=a_01)
I_a_10_importance_sampling, se_importance_sampling_a10 = importance_sampling_with_auxiliary(n_samples=n, a=a_10)

# Print the results with confidence intervals
print(f"Monte Carlo estimate for I(0.1) Part (A): {I_a_01_monte_carlo} ± {se_a_01}")
print(f"Monte Carlo estimate for I(10) Part (A): {I_a_10_monte_carlo} ± {se_a_10}")
# print(f"Monte Carlo estimate for I(0.1) Part (B): {I_a_01_monte_carlo_b} ± {se_a_01_b}")
print(f"Monte Carlo estimate for I(10) Part (B): {I_a_10_monte_carlo_b} ± {se_a_10_b}")
# print(f"Importance sampling estimate for I(0.1) Part (B): {I_a_01_importance_sampling} ± {se_importance_sampling_a01}")
print(f"Importance sampling estimate for I(10) Part (B): {I_a_10_importance_sampling} ± {se_importance_sampling_a10}")