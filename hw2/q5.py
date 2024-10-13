import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.stats import norm
import time

# Set parameters for the system
a, b, c, omega = 0.5, 25, 8, 1.2
n_particles = 500
n_timesteps = 500

# Noise standard deviations
U_std = np.sqrt(10)
V_std = 1

# Initialize particles and weights
particles = np.random.normal(0.1, 0, n_particles)
weights = np.ones(n_particles) / n_particles

# Function to propagate state
def propagate_state(X, t):
    U_t = np.random.normal(0, U_std, size=n_particles)
    return a * X + b * X / (1 + X ** 2) + c * np.cos(omega * (t - 1)) + U_t

# Function to generate measurement
def generate_measurement(X):
    V_t = np.random.normal(0, V_std, size=n_particles)
    return (1/20) * X**2 + V_t

# Function to update weights
def update_weights(particles, yt):
    weights = np.array([norm.pdf(yt, loc=1./20 * particles[i]**2, scale=V_std) for i in range(len(particles))])
    weights /= np.sum(weights)
    return weights

# Function to resample particles
def resample_particles(particles, weights):
    indices = np.random.choice(range(n_particles), size=n_particles, p=weights)
    return particles[indices]

# Initialize arrays to store values
X_true = np.zeros(n_timesteps)
Y_true = np.zeros(n_timesteps)

for t in range(n_timesteps):
    if t == 0:
        X_true[t] = propagate_state(0.1, t)[0]
    else:
        X_true[t] = propagate_state(X_true[t-1], t)[0]

    # true Measurement
    measurement = generate_measurement(X_true[t])[0]
    Y_true[t] = measurement
    


# X_est = np.zeros(n_timesteps)
estimation_errors = np.zeros(n_timesteps)
prediction_errors = np.zeros(n_timesteps)
particle_lst = []
particle_lst.append(particles)

# Run particle filter
for t in range(n_timesteps):
    
    # Propagate particles
    old_particles = particles
    # X*
    particles = propagate_state(particles, t)
    
    # Update weights based on measurement
    weights = update_weights(particles, Y_true[t])
    
    # Estimate the current state
    # X_est[t] = np.sum(particles * weights)
    estimation_errors[t] = np.mean(old_particles - X_true[t])
    if t < n_timesteps - 1:
        prediction_errors[t] = np.mean(particles - X_true[t+1])

    # Resample particles
    particles = resample_particles(particles, weights)

    particle_lst.append(particles)

# (b) Plot the state sequence, estimation error, and prediction error
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(range(n_timesteps), X_true, label='True state $X_t$', color='blue')
# plt.plot(range(n_timesteps), X_est, label='Estimated state $\\tilde{X}_{t|t}$', color='red')
plt.title('State Sequence X_t')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(range(n_timesteps), estimation_errors, label='Estimation Error', color='green')
plt.title('Estimation Error')

plt.subplot(3, 1, 3)
plt.plot(range(n_timesteps - 1), prediction_errors[:-1], label='Prediction Error', color='orange')
plt.title('Prediction Error')

plt.tight_layout()
plt.savefig('q5_state_error.pdf')

# (c) Show distribution of particles at t = 5 and t = 50

time_points = [5, 50]
for t in time_points:
    plt.clf()
    # if t == 5:
        # plt.figure(figsize=(8, 4))
    kde = gaussian_kde(particle_lst[t])
    x_vals = np.linspace(-50, 50, 5000)
    plt.plot(x_vals, kde(x_vals), label=f't = {t}')
    plt.title(f'Distribution of particles at t = {t}')
    plt.legend()
    plt.savefig(f'q5_particles_{t}.pdf')

# (d) Monte Carlo simulation for mean-squared errors
n_simulations = 100
mse_estimation = np.zeros((n_simulations, n_particles))
mse_prediction = np.zeros((n_simulations, n_particles))

for sim in range(n_simulations):
    # print(sim)
    if sim % 5 == 0:
        print(sim)
    st = time.time()
    particles = np.random.normal(0.1, 0, n_particles)
    estimation_errors = np.zeros(n_timesteps)
    prediction_errors = np.zeros(n_timesteps)
    for t in range(n_timesteps): 
        # Propagate particles
        old_particles = particles
        # X*
        particles = propagate_state(particles, t)
        
        # Update weights based on measurement
        weights = update_weights(particles, Y_true[t])
        
        # Estimate the current state
        # X_est[t] = np.sum(particles * weights)
        estimation_errors[t] = np.mean((old_particles - X_true[t])**2)
        if t < n_timesteps - 1:
            prediction_errors[t] = np.mean((particles - X_true[t+1])**2)

        # Resample particles
        particles = resample_particles(particles, weights)

    # print(estimation_errors.shape)
    # print(prediction_errors.shape)

    mse_estimation[sim,:] = estimation_errors
    mse_prediction[sim,:] = prediction_errors

    # print(time.time() - st)

# mse_estimation = np.array(mse_estimation)
# mse_prediction = np.array(mse_prediction)
mse_estimation = np.mean(mse_estimation, axis=0)
mse_prediction = np.mean(mse_prediction, axis=0)

# print(mse_estimation.shape, mse_prediction.shape)
# mse_estimation /= n_timesteps
# mse_prediction /= n_timesteps

# print(f"Mean-squared estimation error: {np.mean(mse_estimation)}")
# print(f"Mean-squared prediction error: {np.mean(mse_prediction)}")

plt.clf()
plt.subplot(2, 1, 1)
plt.plot(range(n_timesteps), mse_estimation, label='Expected Estimation Error', color='green')
plt.title('Mean-squared Estimation Error')

plt.subplot(2, 1, 2)
plt.plot(range(n_timesteps - 1), mse_prediction[:-1], label='Expected Prediction Error', color='orange')
plt.title('Mean-squared Prediction Error')

plt.tight_layout()
plt.savefig('q5_state_error_mento_carlo.pdf')