import numpy as np
import matplotlib.pyplot as plt
% matplotlib inline

# Plot settings
plt.rcParams['figure.figsize'] = (9, 6)
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['xtick.bottom'] = False
plt.rcParams['ytick.left'] = False
pal = ["#FBB4AE", "#B3CDE3", "#CCEBC5", "#CFCCC4"]

# SDE model parameters
mu, sigma, X0 = 2, 1, 1

# Simulation parameters
T, N = 1, 2 ** 7
dt = 1.0 / N
t = np.arange(dt, 1 + dt, dt)  # Start at dt because Y = X0 at t = 0

# Initiate plot object
plt.title('Sample Solution Paths for Geometric Brownian Motion')
plt.ylabel('Y(t)');
plt.xlabel('t')

# Create and plot sample paths
for i in range(len(pal)): # run it 4 times for 4 paths

    # Create Brownian Motion
    np.random.seed(i) # seed set-off for each path separately
    # drawing 128 random numbers from a N(0,1), and drawn altogether, no independently?
    dB = np.sqrt(dt) * np.random.randn(N) # this is where the whole random process comes in, now just gets accumulated
    B = np.cumsum(dB) # = Brownian motion, is what turns exact solution into vector

    # Compute exact solution
    Y = X0 * np.exp((mu - 0.5 * sigma ** 2) * t + sigma * B)

    # Add line to plot
    plt.plot(t, Y, label="Sample Path " + str(i + 1), color=pal[i])

# Add legend
plt.legend(loc=2);


plt.savefig('../Brownian_motion_paths.png')
plt.close()