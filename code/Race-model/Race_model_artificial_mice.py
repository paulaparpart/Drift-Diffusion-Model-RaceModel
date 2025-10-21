"""
Produce Artificial Mouse data
2D Wiener process (independent race model) simulation
Paula Parpart

1. Simulate responses for particular alpha and beta parameters (drift proportionality constant) trajectories (vectors)
and  save data.

    # The output that we need, in order to fit the artificial mice with LearningRaceModel.py:
    # In sum, a vector for responses, reaction times, and contrasts shown


2. Optional: Psychometric curves of choices & RT plots that result from the simulated choices above


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

# no. of artificial mice profiles for model recovery
mice = 30


for m in range(0, mice):


    ## Create the mouse's drift trajectories: itself with Brownian motion simulation
    # we assume 7000 trials for each mouse

    # number of parameters: w1 = alpha, w2 = beta, bias 1, bias 2
    pal = ["#FBB4AE", "#B3CDE3", "#CCEBC5", "#CFCCC4"] # from a plot before

    # SDE model parameters
    # mean of alpha,beta somewhere between 0 and 0.5 (from fitting)
    mu, sigma, X0 = 0.5, 1, 0.3

    # Simulation parameters
    T, N = 1, 7000
    dt = 1.0 / N
    # these time steps will represent the 7000 trials
    t = np.arange(dt, 1 + dt, dt)  # Start at dt because Y = X0 at t = 0


    # Simulate geometric brownian motion for 4 parameters (1 mouse): w1 = alpha, w2 = beta, bias 1, bias 2
    # to get parameter trajectories for the drift rates and biases
    # Create and plot sample paths
    for i in range(len(pal)):

        np.random.seed() # seed for each path (and mouse) randomly

        # to replicate the same profile it would be these:
        # np.random.seed(1)
        # np.random.seed(2)
        # np.random.seed(3)
        # np.random.seed(4)

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


    # Start Race model simulation with the above parameters: to create resulting behaviour data for mouse m

    # Parameters
    dt = .01
    T = 1000
    N = 7000
    nl = .1
    nr = .1
    n = 1
    z = 1
    iti = 1

    # Generate a sequence of random contrasts
    contrasts = np.linspace(-0.5,0.5,5) # Contrast levels map onto actual mice data
    cs = np.random.choice(contrasts, N)
    # alpha, beta and b_l, b_r are random smooth sequences sampled above
    [corr, resp, rt, mint_l, mint_r] = compute_trial_data(alpha, beta, b_l, b_r, z, nl, nr, n, dt, T, N, cs, iti)


    # Store data:  trial  resp 	RT	 cs	 w1	 w2	  b1   b2
    data = np.zeros([N, 8])
    trial = np.linspace(1, N, N)
    data[:, 0] = trial
    data[:, 1] = resp
    data[:, 2] = rt
    data[:, 3] = cs
    data[:, 4] = alpha
    data[:, 5] = beta
    data[:, 6] = b_r
    data[:, 7] = b_l

    # transform to torch
    data = torch.from_numpy(data)

    ## save the data for mouse m
    torch.save(data, 'Mouse_%.0f_tensor.pt' % (m + 1))

    del data, corr, resp,rt, mint_r, mint_l, cs









# Psychometric curves / RT
# Mean response(1 = Right, 0 = Left) across 500 trials (per contrast level), so larger values in mresp means more right

## BEWARE REWRITE CODE: Now these means would need to be filtered by contrast first.

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
