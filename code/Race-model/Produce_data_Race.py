"""

2D Wiener process (independent race model) simulation


- rewriting Andrews race model Matlab code for our purpose

- We got the loglikelihood for a 2D Wiener process (Fokker-planck equation) and want to compare the outputs to this manual 2D-race model simulation
    so that we can safely fit mice with the loglikelihood in pycharm (check out Andrews code for how this works)

- Desired output:

1. Simulate responses for particular alpha and beta parameters (drift proportionality constant) trajectories (vectors)
and  save data. Get output data in a format that can be taken in by the LearningRaceModel.py

## when step 1 is done, also adjust a second LearningRaceModel.py that instead of reading in mouse data from file, takes the artificial data for the variables below directly


2. Psychometric curves of choices & RT plots that result from the simulated choices above


"""

import os
# keep the names small and concise
path = "/Users/paulaparpart/PycharmProjects/TF_Test/RL/Race-model"

os.chdir(path)
os.getcwd()

sys.path.append(".")
import torch
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf # need to put Python 3.6 for that
import torch.optim as optim
import inspect
import pandas as pd
import scipy.optimize
from compute_trials_functions_data import compute_trial_data


# The output that we need, in order to fit the artificial mice with LearningRaceModel.py;
# In sum, a vector for responses, reaction times, and contrasts shown
P = len(resp)  # Number of trials
P = P
resp = torch.tensor(resp.values)  # Response, True (1) = Right, False (0) = Left
T = torch.tensor(RT.values)  # Response Time (s)
c = torch.tensor(contrast.values)  # Contrast, Positive = stimulus on side 1, negative = stim on side 2.
torch.unique(c)

######  So changes needed to existing script are:
# change output variables(resp, rt...) to be vectors = tensors, not matrices
# then inside the compute_trial_results script, maybe instead of cycling through the 7 cs levels,
# have a vectorized computation where c becomes a vector rather than a scalar? in the computation of A_l and A_r
# as well as alpha and beta themselves, which used to be scalars (.3) thus the same for each trial, but now
# they would be different on each trial, i.e, the made-up trajectory vectors


# Plot settings
plt.rcParams['figure.figsize'] = (9, 6)
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['xtick.bottom'] = False
plt.rcParams['ytick.left'] = False
pal = ["#FBB4AE", "#B3CDE3", "#CCEBC5", "#CFCCC4"]

# SDE model parameters
# mean of alpha,beta somewhere between 0 and 0.5 (from fitting)
mu, sigma, X0 = 0.5, 1, 0.3

# Simulation parameters
T, N = 1, 7000
dt = 1.0 / N
# these time steps will represent the 7000 trials
t = np.arange(dt, 1 + dt, dt)  # Start at dt because Y = X0 at t = 0

# Initiate plot object
plt.title('Sample Solution Paths for Geometric Brownian Motion')
plt.ylabel('Y(t)');
plt.xlabel('t')

# Simulate geometric brownian motion for 4 parameters (1 mouse):
# Create and plot sample paths
for i in range(len(pal)): # run it 4 times for 4 paths

    # Create Brownian Motion
    np.random.seed(i) # seed for each path (and mouse) randomly

    # to replicate the same profile it would be these:
    # np.random.seed(1)
    # np.random.seed(2)
    # np.random.seed(3)
    # np.random.seed(4)

    # drawing 128 random numbers from a N(0,1), and drawn altogether, no independently?
    dB = np.sqrt(dt) * np.random.randn(N) # this is where the whole random process comes in, now just gets accumulated
    B = np.cumsum(dB) # = Brownian motion, is what turns exact solution into vector

    # Compute exact solution
    Y = X0 * np.exp((mu - 0.5 * sigma ** 2) * t + sigma * B)

    if i ==0:
        beta = Y  # w2
    if i == 1:
        alpha = Y  # w1
    if i == 2:
        b_r = Y
    if i == 3:
        b_l = Y


plt.plot(t, beta, label="beta (w2)", color=pal[0])
plt.plot(t, alpha, label="alpha (w1)", color=pal[1])
plt.plot(t, b_r, label="Bias R", color=pal[2])
plt.plot(t, b_l, label="Bias L", color=pal[3])

    # Add line to plot
    #plt.plot(t, Y, label="Sample Path " + str(i + 1), color=pal[i])

# Add legend
plt.legend(loc=2)
plt.savefig('../Mouse_2_drift_rates.png')
plt.close()


# Parameters
dt = .01
T = 1000
N = 7000 # artificial trials
nl = .1
nr = .1
n = 1
z = 1
iti = 1


# Generate random contrasts
contrasts = np.linspace(-0.5,0.5,5) # Contrast levels
cs = np.random.choice(contrasts, N)
# alpha, beta and b_l, b_r are random smooth sequences


[corr, resp, rt, mint_l, mint_r] = compute_trial_data(alpha, beta, b_l, b_r, z, nl, nr, n, dt, T, N, cs, iti)


# trial 	resp 	RT	 cs	w1	w2	b1	b2

# resp is now vector of length 7000 (not matrix 500x9)
# cs is the contrasts of length 7000
# rt, corr, mint_l and mint_r are vectors of length 7000
# put them into data frame into columns for plotting

# then turn into long torches for learning model




# Psychometric curves / RT
# Mean response(1 = Right, 0 = Left) across 500 trials (per contrast level), so larger values in mresp means more right

## Now the response has to be filtered by contrast first.

mcorr = np.mean(corr, axis = 0)
mresp = np.mean(resp, axis  = 0)
mrt = np.mean(rt, axis = 0)

fig2, axs = plt.subplots(1, 2, sharex=True, sharey=False)  # normally: (3, 2, sharex=False, sharey=False)
fig2.subplots_adjust(hspace=.25, wspace=.25)
axs = axs.ravel()

axs[0].plot(cs, mresp)
axs[0].set_ylim(bottom=0, top=1)
axs[0].set_xlim(-1, 1)
axs[0].set_yticks((0,1)) # :)
axs[0].set_xlabel('Contrast', size = 14)
axs[0].set_ylabel('p(right)', size = 14)
axs[0].set_xticks((-1,0,1)) # :)

axs[1].plot(cs, mrt)
axs[1].set_ylim(bottom=0, top=4)
axs[1].set_yticks((0,4)) # :)
axs[1].set_xlabel('Contrast', size = 14)
axs[1].set_ylabel('RT', size = 14)
axs[1].set_xticks((-1,0,1)) # :)


fig2.set_size_inches(10, 5)
plt.savefig('../Race-model/psychometric_alpha_%.1f__beta_%.1f.png' % (alpha, beta))
plt.close()
