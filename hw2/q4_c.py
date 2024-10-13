import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, laplace
# import seaborn as sns


def bootstrap_variance_estimator(n, K=1000, dist='normal'):
    # Step 1: Generate data
    if dist == 'normal':
        data = np.random.normal(0, 1, n)
    elif dist == 'laplace':
        data = laplace.rvs(loc=0, scale=1/np.sqrt(2), size=n)  # unit variance Laplace

    theta_hat = np.mean(data**2)
    bootstrap_estimates = []

    # Step 2: Perform Bootstrap
    for _ in range(K):
        resampled_data = np.random.choice(data, size=n, replace=True)
        theta_hat_resampled = np.mean(resampled_data**2)
        # Normalized error for bootstrapped variance estimate
        normalized_error = np.sqrt(n) * (theta_hat_resampled - 1)  # theta = 1 for both distributions
        bootstrap_estimates.append(normalized_error)

    return np.array(bootstrap_estimates), theta_hat, np.mean(bootstrap_estimates), np.var(bootstrap_estimates)

# Parameters
n_small = 6
n_large = 100
K = 1000

from scipy.stats import gaussian_kde

def fit_gaussian(bootstrap_estimates):
    # Fit Gaussian to the bootstrapped estimates
    estimated_std = np.std(bootstrap_estimates, ddof=1)  # unbiased estimator for sample std
    return estimated_std

def plot_fitted_gaussian(bootstrap_estimates, dist_name):
    # KDE for the empirical distribution
    kde = gaussian_kde(bootstrap_estimates)
    x_vals = np.linspace(min(bootstrap_estimates), max(bootstrap_estimates), 1000)
    # print(x_vals)
    kde_pdf = kde(x_vals)

    # Fit Gaussian
    std = fit_gaussian(bootstrap_estimates)
    fitted_pdf = norm.pdf(x_vals, 0, std)

    # Plot
    plt.clf()
    plt.plot(x_vals, kde_pdf, label='Empirical PDF', color='blue')
    
    plt.plot(x_vals, fitted_pdf, label=f'Gaussian Fit (mu={0:.2f}, std={std:.2f})', color='red')
    plt.fill_between(x_vals, np.abs(kde_pdf - fitted_pdf), color='gray', alpha=0.3, label='$L_1$ Error')
    plt.legend()
    plt.title(f"Gaussian Fit vs Empirical PDF ({dist_name})")
    plt.xlabel("Normalized Error")
    plt.ylabel("Density")
    plt.savefig(f'q4_gau_{dist_name}.pdf')

    # Compute L1 error
    L1_error = np.sum(np.abs(kde_pdf - fitted_pdf))
    return L1_error

# Run bootstrap for n = 6 and n = 100, Laplace Distribution
bootstrap_small_laplace, _, mean_small, var_small = bootstrap_variance_estimator(n_small, K, dist='laplace')
bootstrap_large_laplace, _, mean_large, var_large = bootstrap_variance_estimator(n_large, K, dist='laplace')

# Plot results
plt.hist(bootstrap_small_laplace, bins=30, density=True, alpha=0.6, color='g', label='n=6')
plt.hist(bootstrap_large_laplace, bins=30, density=True, alpha=0.6, color='b', label='n=100')
plt.title("Bootstrap Estimates for Laplace Distribution (n=6 vs n=100)")
plt.xlabel("Normalized Error")
plt.ylabel("Density")
plt.legend()
plt.savefig('q4_error_laplace.pdf')

print('n=6, ', f'mean = {mean_small}, var = {var_small}')
print('n=100, ', f'mean = {mean_large}, var = {var_large}')

# Plot and fit Gaussian for Laplace distribution, n=6
L1_small_laplace = plot_fitted_gaussian(bootstrap_small_laplace, 'n=6 Laplace Distribution')

# Plot and fit Gaussian for Laplace distribution, n=100
L1_large_laplace = plot_fitted_gaussian(bootstrap_large_laplace, 'n=100 Laplace Distribution')

print(f"L1 Error for n=6 (Laplace): {L1_small_laplace:.4f}")
print(f"L1 Error for n=100 (Laplace): {L1_large_laplace:.4f}")