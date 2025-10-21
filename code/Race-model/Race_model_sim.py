"""

2D Wiener process (independent race model) simulation


- rewriting Andrews race model Matlab code for our purpose
Replicate Andrews matlab race model simulation

"""

import os
# keep the names small and concise
path = "/Users/paulaparpart/PycharmProjects/TF_Test/RL/Race-model/"
os.chdir(path)
os.getcwd()

sys.path.append(".")
import torch
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch.optim as optim
import inspect
import pandas as pd
import scipy.optimize
from compute_trials_functions import compute_balanced_RR, compute_trial_results



## Replicate the basic Matlab script:  plot_behav_strat_neural_pred.m
## The bias we care about is this: Before psychometric curves become symmetric for L and R, there is often a leftward or rightward bias


dt = .01
T = 1000
N = 500

nl = .1
nr = .1
n = 1
z = 1
iti = 1

# vary
alpha = .3   # 4  6   #  .3 # right-bias
beta = .3   # 0  6    # .3  # left-bias


# Determine bias b that balances responses

RR, mcorr, mRT, mint_L, mint_R, b = compute_balanced_RR(alpha, beta, z, nl, nr, n, dt, T, N, iti)


# Reproduce flat curve for first 1000 trials
Nbig = 1000
cs = np.linspace(-1,1,9) # Contrast levels

[corr, resp, rt, mint_l, mint_r] = compute_trial_results(alpha, beta, b, z, nl, nr, n, dt, T, Nbig, cs, iti)



# Psychometric curves / RT
# Mean response(1 = Right, 0 = Left) across 500 trials (per contrast level), so larger values in mresp means more right
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
