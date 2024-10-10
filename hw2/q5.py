import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

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
def update_weights(particles, measurement):
    predicted_measurements = generate_measurement(particles)
    # predicted_measurements = (1/20) * (particles ** 2)
    # weights = np.exp(-0.5 * (measurement - predicted_measurements)**2 / V_std**2)
    weights = np.exp(-(measurement - predicted_measurements)**2)
    weights /= np.sum(weights)
    return weights

# Function to resample particles
def resample_particles(particles, weights):
    indices = np.random.choice(range(n_particles), size=n_particles, p=weights)
    return particles[indices]

# Initialize arrays to store values
X_true = np.zeros(n_timesteps)
X_est = np.zeros(n_timesteps)
estimation_errors = np.zeros(n_timesteps)
prediction_errors = np.zeros(n_timesteps)
particle_lst = []
particle_lst.append(particles)

# Run particle filter
for t in range(n_timesteps):
    # True state
    if t == 0:
        X_true[t] = 0.1
    else:
        X_true[t] = propagate_state(X_true[t-1], t)[0]
    
    # Propagate particles
    particles = propagate_state(particles, t)
    
    # Measurement
    measurement = generate_measurement(X_true[t])
    
    # Update weights based on measurement
    weights = update_weights(particles, measurement)
    
    # Estimate the current state
    X_est[t] = np.sum(particles * weights)
    estimation_errors[t] = np.mean(particles - X_true[t])

    # Resample particles
    particles = resample_particles(particles, weights)

    particle_lst.append(particles)
    
    # Calculate errors
    if t > 0:      
        if t < n_timesteps - 1:
            # Predict the next state (using the resampled particles at t)
            particles_pred = propagate_state(particles, t)
            # X_pred = np.sum(particles_pred * weights)  # Predicted state at t+1
            prediction_errors[t] = np.mean(particles_pred - X_true[t+1])


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
mse_estimation = np.zeros(n_simulations)
mse_prediction = np.zeros(n_simulations)

for sim in range(n_simulations):
    particles = np.random.normal(0.1, 0, n_particles)
    for t in range(n_timesteps):
        # Propagate particles
        particles = propagate_state(particles, t)
        measurement = generate_measurement(X_true[t])
        weights = update_weights(particles, measurement)
        X_est_t = np.sum(particles * weights)
        # Compute errors
        mse_estimation[sim] += np.mean((particles - X_true[t])**2)

        particles = resample_particles(particles, weights)
        
        
        if t < n_timesteps - 1:
            # Predict the next state (using the resampled particles at t)
            particles_pred = propagate_state(particles, t)
            # X_pred = np.sum(particles_pred * weights)  # Predicted state at t+1
            mse_prediction[sim] += np.mean((particles_pred - X_true[t+1])**2)

mse_estimation /= n_timesteps
mse_prediction /= n_timesteps

print(f"Mean-squared estimation error: {np.mean(mse_estimation)}")
print(f"Mean-squared prediction error: {np.mean(mse_prediction)}")