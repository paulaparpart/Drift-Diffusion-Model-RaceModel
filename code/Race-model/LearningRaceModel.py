import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributions as td
torch.pi = torch.acos(torch.zeros(1)).item() * 2
import os

import pandas as pd

import matplotlib.cm as cm
import matplotlib.pyplot as plt

class LearningRaceModel:

    def __init__(self, learning_rate, momentum, mouseNum, max_epochs, zvar, wvar, bvar, verbose):

        self.learning_rate = learning_rate
        self.momentum = momentum

        self.verbose = verbose


        # Load dataset (hardcoded for now)
        mouse = pd.read_csv(r'../../dopaminepl-scratch/Dataset_2020-11-03/DAP_LearningGrating2AFC_mouseInfo.csv')
        sess = pd.read_csv(r'../../dopaminepl-scratch/Dataset_2020-11-03/DAP_LearningGrating2AFC_sessionInfo.csv')
        trial = pd.read_csv(r'../../dopaminepl-scratch/Dataset_2020-11-03/DAP_LearningGrating2AFC_trialInfo.csv')

        # Combine trial & session info
        df = trial.join(sess.set_index('expRef'), on='expRef', how='inner')

        # Unique mice names
        mouse_names = df['mouseName'].unique()

        self.mouse = mouse_names[mouseNum]
        print(self.mouse)

        # Pull out relevant vars. Depends on trials being in order, should double check.
        resp = df.choice[df.mouseName==self.mouse]=='Right choice' # Right choice is stim 1
        contrast = df.contrastRight[df.mouseName==self.mouse] - df.contrastLeft[df.mouseName==self.mouse]
        RT = df.choiceCompleteTime[df.mouseName==self.mouse] - df.goCueTime[df.mouseName==self.mouse]

        # Convert to torch

        self.P = len(resp) # Number of trials
        P = self.P
        self.r = torch.tensor(resp.values) # Response, True = 1, False = 2
        self.T = torch.tensor(RT.values) # Response Time (s)
        self.c = torch.tensor(contrast.values) # Contrast, Positive = stimulus on side 1, negative = stim on side 2.

        # Initialize race model parameters
        self.z = torch.ones(P,requires_grad=True) # Threshold
        self.w1 = torch.linspace(0.,.5,P,requires_grad=True) # Resp 1 input-output association strength
        self.w2 = torch.linspace(0.,.5,P,requires_grad=True) # Resp 2 input-output association strength
        self.b1 = torch.linspace(0.01,0.01,P,requires_grad=True) # Resp 1 constant bias strength
        self.b2 = torch.linspace(0.01,0.01,P,requires_grad=True) # Resp 2 constant bias strength
        self.sig_i = torch.tensor([1.],requires_grad=True) # Input noise variance
        self.sig_o = torch.tensor([1.],requires_grad=False) # Output noise variance. Sets units of parameters. Fix to 1.

        # Initialize learning model hyperparameters
        self.zvar = zvar
        self.wvar = wvar
        self.bvar = bvar

        self.max_epochs = max_epochs


        # Set up and save normal CDF
        self.std_n = td.Normal(torch.tensor([0.0]), torch.tensor([1.0]))

    def inv_gauss_pdf(self, thr,drift,v,x0,t):
        return (thr - x0)/torch.sqrt(2*torch.pi*v*t**3)*torch.exp(-(thr - x0 - drift*t)**2/(2*v*t))

    def inv_gauss_cdf(self, thr, drift, v, x0, t): 
        return self.std_n.cdf((drift*t-(thr-x0))/torch.sqrt(v*t)) + torch.exp(2.*(thr-x0)*drift/v)*self.std_n.cdf(-(drift*t+(thr-x0))/torch.sqrt(v*t))

    def NegLogLikelihood(self):

        # Calculate drift rates & noise variances for each integrator
        drift1 = self.w1*F.relu(self.c) + F.relu(self.b1) 
        drift2 = self.w2*F.relu(-self.c) + F.relu(-self.b1)

        v1 = self.w1**2*self.sig_i**2 + self.sig_o**2
        v2 = self.w2**2*self.sig_i**2 + self.sig_o**2

        # Drift & variance of chosen option
        drift_r = self.r*drift1 + (~self.r)*drift2
        v_r = self.r*v1 + (~self.r)*v2

        # Drift & var of unchosen option
        drift_rbar = (~self.r)*drift1 + self.r*drift2
        v_rbar = (~self.r)*v1 + self.r*v2

        # Calculate resulting negative log likelihood
        self.nll = -torch.log(self.inv_gauss_pdf(self.z,drift_r,v_r,0.,self.T))
        self.nll2 = -torch.log(1.-self.inv_gauss_cdf(self.z, drift_rbar, v_rbar, 0., self.T))
        
        return torch.sum(self.nll+self.nll2)

    def diff(self, M):
        return M[1:] - M[:-1]

    def TemporalTransitionNegLogLikelihood(self):

        tot = torch.norm(self.diff(self.z))**2/(2.*self.zvar)
        tot = tot + torch.norm(self.diff(self.w1))**2/(2.*self.wvar)
        tot = tot + torch.norm(self.diff(self.w2))**2/(2.*self.wvar)
        tot = tot + torch.norm(self.diff(self.b1))**2/(2.*self.bvar)
        tot = tot + torch.norm(self.diff(self.b2))**2/(2.*self.bvar)

        return tot

    def forward(self):
        return self.NegLogLikelihood() + self.TemporalTransitionNegLogLikelihood()

    def fit(self):


        optimizer = torch.optim.SGD([self.w1,self.w2,self.b1,self.sig_i,self.z], lr=self.learning_rate, momentum=self.momentum)

        self.loss_hist = np.zeros(self.max_epochs)
        for t in range(self.max_epochs):
            loss = self.forward()

            if self.verbose and (t<100 or t % 1000 == 0):
                print(t, loss.item())

            self.loss_hist[t] = loss.item()

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

