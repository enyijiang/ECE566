import numpy as np
from scipy.special import psi  # Digamma function
from numpy.linalg import inv, det
import numpy as np
import copy
from em import EM_algorithm
import matplotlib.pyplot as plt

# np.random.seed(20)

def Variational_Inference(data, max_iter=200, tol=1e-6):
    n, d = data.shape[0], data.shape[1]
    # initialize parameters
    Y_mean = np.mean(data, axis=0)
    Y_cov = np.cov(data.T)

    m = 2
    c0 = 1
    v0 = 2
    mu0 = Y_mean
    B_inv_0 = v0 * Y_cov  # B inverse = nu * empirical covariance
    a0 = np.array([1./2, 1./2])  # Beta prior parameters for arcsin distribution

    c = np.ones(m)
    v = np.ones(m)
    old_mu = np.array([mu0 for _ in range(m)])
    mu = np.random.randn(m, d)
    B_inv = np.array([B_inv_0 for _ in range(m)])
    a = copy.deepcopy(a0)
    # Responsibilities (r_ij)
    # r = np.random.rand(n, m)
    r = np.random.dirichlet([1.0] * m, size=n)
    errors = []

    for iteration in range(max_iter):
        # Evaluate the parameters of (7.42)
        for j in range(m):
            a[j] = a0[0] + r[:, j].sum()
            c[j] = c0 + r[:, j].sum()
            v[j] = v0 + r[:, j].sum()
            B_inv[j] = B_inv_0 + sum(r[i, j] * np.outer(data[i], data[i]) for i in range(n)) + c0 * (mu0 @ mu0.T)
            mu[j] = (c0 * mu0 + sum(r[i, j] * data[i] for i in range(n))) / (c0 + r[:, j].sum())
        
        # Evaluate three expectations of (7.43)
        E_ln_omega = psi(a) - psi(a.sum())
        E_ln_Lambda = np.zeros(m)
        E_quad = np.zeros((n, m))

        for j in range(m):
            E_ln_Lambda[j] = sum(psi((v[j] + 1 - i) / 2) for i in range(1, d + 1)) + d * np.log(2) + np.log(det(inv(B_inv)[j]))
            for i in range(n):
                diff = data[i] - mu[j] # shape (1, 2)
                E_quad[i, j] = (
                    d / c[j]
                    + v[j] * (diff.T @ inv(B_inv)[j] @ diff)
                )
        
        # Update the responsibilities
        rho = np.zeros((n, m))

        for i in range(n):
            for j in range(m):
                rho[i, j] = np.exp(
                    E_ln_omega[j]
                    + 0.5 * E_ln_Lambda[j]
                    - 0.5 * E_quad[i, j]
                )
            r[i, :] = rho[i, :] / rho[i, :].sum()
        

        errors.append(np.abs(old_mu - mu).sum())
        
        if np.abs(old_mu - mu).sum() < tol:
            print(f"Converged at iteration {iteration}")
            # break

        old_mu = copy.deepcopy(mu)
    
    # Results
    posterior_mean = mu
    posterior_covariance = [1. / (c[j] * v[j]) * B_inv[j] for j in range(m)]  # Covariance from precision
    mean_Lambda = [v[j] * inv(B_inv)[j] for j in range(m)]  # Precision matrix expectation
    mean_omega = a / a.sum()
    var_omega = (a[0] * a[1]) / ((a.sum())**2 * (a.sum() + 1))

    # Display Results
    print("Gaussian-Wishart Parameters:")
    for j in range(m):
        print(f"Component {j+1}:")
        print(f"  mu: {mu[j]}")
        print(f"  c: {c[j]}")
        print(f"  B: \n{inv(B_inv)[j]}")
        print(f"  nu: {v[j]}")

    print("\nPosterior Mean and Covariance:")
    for j in range(m):
        print(f"Component {j+1}:")
        print(f"  Posterior Mean: {posterior_mean[j]}")
        print(f"  Posterior Covariance: \n{posterior_covariance[j]}")

    print("\nMean of Lambda (Precision Matrices):")
    for j in range(m):
        print(f"Component {j+1}: \n{mean_Lambda[j]}")

    print("\nDirichlet (Beta) Parameters:")
    print(f"Alpha: {a}")
    print(f"Posterior Mean of Omega: {mean_omega}")
    print(f"Posterior Variance of Omega: {var_omega}")

    return errors

if __name__ == '__main__':
    # load in data
    data = np.loadtxt('data.txt') # shape (200,2)

    # pi_est, mu1_est, cov1_est, mu2_est, cov2_est, errors_em_random = EM_algorithm(data, 'random', max_iter=50)
    pi_est, mu1_est, cov1_est, mu2_est, cov2_est, errors_em_ours = EM_algorithm(data, 'ours', max_iter=200)
    errors_var = Variational_Inference(data, max_iter=200)

    # Plot the second line
    plt.plot(errors_var, label='Variational', color='blue', linewidth=2)
    plt.plot(errors_em_ours, label='EM', color='red', linewidth=2)
    # plt.plot(errors_em_random, label='EM_random', color='green', linewidth=2)

    # Add title and labels
    plt.title('Plot of erros with EM and VI')
    plt.xlabel('Iterations')
    plt.ylabel('Error')

    # Add legend
    plt.legend()

    # Add grid
    plt.grid(True)

    # Show the plot
    plt.savefig('error_diff_hw5.png')




