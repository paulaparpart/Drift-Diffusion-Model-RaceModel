import numpy as np





def compute_trial_data(alpha, beta, b_l, b_r, z, nl, nr, n, dt, T, N, cs, iti):

    # % Inputs
    # %
    # %   alpha   Right response contrast-to-drift-rate proportionality constant
    # %   beta    Left response contrast-to-drift-rate proportionality constant
    # %   b       Signed bias input to integrator, +ve is toward left resp
    # %   z       Response threshold
    # %   nl      Left input noise variance
    # %   nr      Right input noise variance
    # %   n       Output noise variance
    # %   dt      Simulation timestep
    # %   T       Max time steps to roll out
    # %   N       Number of trials
    # %   cs      Vector of contrast levels to simulate. Positive = left
    # %   iti     Length of inter-trial interval

    # % Outputs
    # %   corr    N x length(cs) binary matrix indicating correct trials
    # %   resp    N x length(cs) binary matrix indicating response direction
    # %   (1=left)
    # %   rt      N x length(cs) matrix indicating reaction time
    # %   mint_l  T x length(cs) matrix with mean left integrator trace (averaged over
    # %   trials)
    # %   mint_r  T x length(cs) matrix with mean right integrator trace (averaged over trials)


    # For testing

    #  b=0 # no bias for testing inside function, otherwise it comes from compute_balanced_RR
    #  dt = .01
    #  T = 1000
    #  N = 500
    #
    #  nl = .1
    #  nr = .1
    #  n = 1
    #  z = 1
    #  iti = 1
    #
    # cs = np.linspace(-1,1,9)
    # alpha = .3
    # beta = .3

    # c is now a different value for each trial, from the 'real' mouse
    # N is the actual number of trials coming in from the real mouse

    # cs is of length N = 7000
    c = cs


    # Brownian motion integrals: sigma * dB,
    # np.sqrt(nl) = c;  dB = sqrt(dt)*randn(N,T)
    Eta_l = np.sqrt(nl)*np.sqrt(dt)*np.random.normal(0, 1, (N,T))
    Eta_r = np.sqrt(nr)*np.sqrt(dt)*np.random.normal(0, 1, (N,T))

    # output noise
    # Eta = np.sqrt(n)*randn(N,T)*np.sqrt(dt); % Should really do a left/right version of this independently to both integrators. Going to induce a small correlation
    Out_l = np.sqrt(n)*np.random.normal(0, 1, (N,T))*np.sqrt(dt)
    Out_r = np.sqrt(n)*np.random.normal(0, 1, (N,T))*np.sqrt(dt)


    # Drift integrals: alpha/beta diminishes or pushes effect of the contrast
    # they used to repeat the same scalar 1000 times for each column in matrix: Can we now just create a matrix by replicating max(0, alpha * c)- vector?
    # A_r = alpha*max(0,alpha*c)* np.ones((N,T)) *dt  # if contrast is positive (on right side)
    # A_l = -beta*min(0,beta*c)*np.ones((N,T))*dt  # if contrast is negative (on the left side)

    A = alpha * np.maximum(np.zeros(N), alpha * c)
    A = np.reshape(A, (1, len(A))) # to stack as columns
    # matrix (N,T) = (7000, 1000)  * dt
    A_r = np.tile(A.transpose(), (1, T)) * dt

    A2 = -beta * np.minimum(np.zeros(N), beta * c)
    A2 = np.reshape(A2, (1, len(A2)))  # to stack as columns
    A_l = np.tile(A2.transpose(), (1, T)) * dt


    # now given as b = vector, and we turn back into a matrix
    #Bias_r = max(0, b) * np.ones((N, T)) * dt
    #Bias_l = -min(0, b) * np.ones((N, T)) * dt
    b_r = np.maximum(np.zeros(N), b_r)
    b_r = np.reshape(b_r, (1, len(b_r))) # matrix (N,T)
    Bias_r = np.tile(b_r.transpose(), (1, T)) * dt

    b_l = -np.minimum(np.zeros(N), b_l)
    b_l = np.reshape(b_l, (1, len(b_l)))  # matrix (N,T)
    Bias_l = np.tile(b_r.transpose(), (1, T)) * dt


    # accumulators values = drift Integral + Brownian motion + bias + output noise:
    # 7000 x 1000
    int_l = np.cumsum(A_l + Eta_l + Bias_l + Out_l,axis = 1)    # row-wise cumsum: per trial
    int_r = np.cumsum(A_r + Eta_r + Bias_r + Out_r,axis = 1)

    # counts time points until the int_l was crossing threshold, e.g., 4 time steps if it was fast
    hitting_l = np.sum(np.cumprod((int_l>z)==0, axis = 1),axis = 1) # cumulative product row-wise of boolean that indicates whether int was > threshold
    hitting_r = np.sum(np.cumprod((int_r>z)==0, axis = 1),axis = 1)

    # this sets the accumulators values to 1 for all those above the 1
    # threshold (probability mass in top stays in top for rest of trial)
    for i in range(N):
        int_l[i,hitting_l[i]:T]=z
        int_r[i,hitting_r[i]:T]=z

    # turns time steps into a reaction time with *0.01
    hitting_l = hitting_l*dt
    hitting_r = hitting_r*dt

    # puts the RT only for the fastest integrator
    hitting = np.minimum(hitting_l,hitting_r)
    # which response is predicted based on which one hit threshold sooner: 1 = Right; 0 = Left
    resp = (((hitting_l==hitting_r)*(np.random.rand(len(hitting_l))>.5)) *1) +  (~(hitting_l==hitting_r) *(hitting_r < hitting_l)*1)
    # is the response above correct? compare to ground truth
    corr = (resp == (c>0)) *1    # at the moment means Left, but change to Right

    # vector of 1000 len: so averaged across
    mint_l = np.mean(int_l, axis = 0)
    mint_r = np.mean(int_r, axis = 0)

    # this needs to be adjusted
    # corr[c == 0]  =
    for l in range(len(corr)):
        if c[l] == 0:
            corr[l] = (np.random.rand(1)>.5)*1
    # if c==0:  # randomly correct if contrast = 0, ground truth doesnt have preference
    #    corr= (np.random.rand(len(corr))>.5)*1

    # what is this?  Reward-rate
    RR = corr/(hitting + iti)  # this is done per trial out of the 500 trials: corr(yes/no)./(time it took + 1)

    # rename
    rt = hitting

        # all vectors of length 7000 now
    return corr, resp, rt, mint_l, mint_r
