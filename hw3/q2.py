import numpy as np
from scipy.stats import multivariate_normal
import copy
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Generate synthetic data (replace this with your actual data)
n = 500

# True parameters for two Gaussians
# pi_true = 0.6
# mu1_true = np.array([1, 1])
# mu2_true = np.array([3, 3])
# cov1_true = np.array([[2, 0.1], [0.1, 2]])
# cov2_true = np.array([[1, 0.5], [0.5, 1]])

# # Generate samples
# y1 = np.random.multivariate_normal(mu1_true, cov1_true, size=int(pi_true * n))
# y2 = np.random.multivariate_normal(mu2_true, cov2_true, size=int((1 - pi_true) * n))
# data = np.vstack((y1, y2))

data = np.loadtxt('data.txt')

# Helper functions for E-step and M-step
def e_step(data, pi, mu1, cov1, mu2, cov2):
    phi1 = multivariate_normal.pdf(data, mean=mu1, cov=cov1)
    phi2 = multivariate_normal.pdf(data, mean=mu2, cov=cov2)
    
    gamma1 = pi * phi1
    gamma2 = (1 - pi) * phi2
    gamma_sum = gamma1 + gamma2
    
    gamma1 /= gamma_sum
    gamma2 /= gamma_sum
    
    return gamma1, gamma2

def m_step(data, gamma1, gamma2):
    N1 = np.sum(gamma1)
    N2 = np.sum(gamma2)
    
    pi_new = N1 / (N1 + N2)
    
    # Update means
    mu1_new = np.sum(gamma1[:, np.newaxis] * data, axis=0) / N1
    mu2_new = np.sum(gamma2[:, np.newaxis] * data, axis=0) / N2
    
    # Update covariance matrices
    cov1_new = np.dot(((data - mu1_new)).T, gamma1[:, np.newaxis] *  (data - mu1_new)) / N1
    cov2_new = np.dot(( (data - mu2_new)).T, gamma2[:, np.newaxis] * (data - mu2_new)) / N2
    
    return pi_new, mu1_new, cov1_new, mu2_new, cov2_new

def log_likelihood(data, pi, mu1, cov1, mu2, cov2):
    phi1 = multivariate_normal.pdf(data, mean=mu1, cov=cov1)
    phi2 = multivariate_normal.pdf(data, mean=mu2, cov=cov2)
    
    return np.sum(np.log(pi * phi1 + (1 - pi) * phi2))

# EM algorithm
def EM_algorithm(data, method, max_iter=200, tol=1e-6):
    # Initialize parameters
    # kmeans = KMeans(n_clusters=2, random_state=42)
    # kmeans.fit(data)

    # Get the cluster centers
    # centers = kmeans.cluster_centers_
    # mu1 = centers[0,:]
    # mu2 = centers[1,:]

    # some calculatations
    if method == 'ours':
        pi = 0.5
        # mu1 = np.mean(data[:250], axis=0) 
        # mu2 = np.mean(data[250:], axis=0)
        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans.fit(data)

        # Get the cluster centers
        centers = kmeans.cluster_centers_
        mu1 = centers[0,:]
        mu2 = centers[1,:]
        
        cov1 = np.cov(data[:100].T)
        cov2 = np.cov(data[100:].T)
    
    # random initialization
    if method == 'random':
        pi = np.random.uniform()
        mu1 = np.random.rand(2)
        mu2 = np.random.rand(2)
        A = np.random.rand(2, 2)
        cov1 = np.dot(A, A.T)
        A = np.random.rand(2, 2)
        cov2 = np.dot(A, A.T)
    
    log_likelihoods = []
    errors = []

    for iteration in range(max_iter):
        # E-step
        gamma1, gamma2 = e_step(data, pi, mu1, cov1, mu2, cov2)

        old_mu1 = mu1
        
        # M-step
        pi, mu1, cov1, mu2, cov2 = m_step(data, gamma1, gamma2)
        
        # Compute log-likelihood and check for convergence
        ll = log_likelihood(data, pi, mu1, cov1, mu2, cov2)
        log_likelihoods.append(ll)
        
        errors.append(np.sum(abs(old_mu1 - mu1)))

        if iteration > 0 and np.sum(abs(old_mu1 - mu1)) < tol:
            print(f"Converged at iteration {iteration}")
            # break
    
    return pi, mu1, cov1, mu2, cov2, errors#, log_likelihoods

# Run the EM algorithm
pi_est, mu1_est, cov1_est, mu2_est, cov2_est, errors_1 = EM_algorithm(data, 'random')
pi_est, mu1_est, cov1_est, mu2_est, cov2_est, errors_2 = EM_algorithm(data, 'ours')



# Print the final estimated parameters
# print(f"Estimated pi: {pi_est}")
# print(f"Estimated mu1: {mu1_est}")
# print(f"Estimated cov1: \n{cov1_est}")
# print(f"Estimated mu2: {mu2_est}")
# print(f"Estimated cov2: \n{cov2_est}")

# Plot the first line
plt.plot(errors_1, label='Random', color='blue', linewidth=2)

# Plot the second line
plt.plot(errors_2, label='Ours', color='red', linewidth=2)

# Add title and labels
plt.title('Plot of erros with different initializations')
plt.xlabel('Iterations')
plt.ylabel('Error')

# Add legend
plt.legend()

# Add grid
plt.grid(True)

# Show the plot
plt.savefig('error_diff.png')