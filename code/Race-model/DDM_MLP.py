
"""
    MLP simulation to learn mouse biases
    e.g., assymmetric psychometric functions (RR or LL)

    Input data: artificial mouse data
    decide on number of trials
    continuous contrast on either side (L/R) - decide on bins
    contrast in unity range
    generate ground truth y (from stimuli)

    The network:
    Input layer: Gaussian population code / continuous rate code / one-hot vector place code
    1 hidden layer
    1 output unit (sigmoid)
    noise - add some noise to input somehow

    Output graphs:
    1. Learning curve (accuracy/loss)
    2. Actions (% Rightward) as a function of contrast R and contrast L (accumulated and sorted across all trials)
    4. Actions (% Repeat) as function of last outcome (Win - Loose)

"""

import os
# keep the names small and concise
path = "/Users/paulaparpart/PycharmProjects/TF_Test/RL"
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

def logistic(var):
    p = 1 / (1 + np.exp(-var))  # elementwise devide in python: a/b
    return p

# Merge the split data into mini-batches:
def merge(list1, list2):
    merged_list = [[list1[i], list2[i]] for i in range(0, len(list1))]
    return merged_list


## Generate the data
#   decide on number of trials
#   continuous contrast on either side (L/R) - decide on bins: 100
#   contrast in unity range
#   generate ground truth y (from stimuli)

# trials
T = 800 * 100 # 80,000 in total
mice = 1
L_param = 0.01 # I used 0.001
trainsets = []
noise = 0.1  # could change and experiment with to test influence on biases by NN

# 100 mice for now: trainsets
for m in range(mice):
    # randomly draw contrasts on L and R side between 0-1 (rescaled from 0-100) with 0 being more likely
    CR = np.random.rand(T, 1)  # 800x1
    indices = np.random.choice(np.arange(T), replace=False, size=round(T/2))
    CR[indices] = 0
    CL = np.random.rand(T, 1)  # 800x1
    CL[CR > 0] = 0  # when CR is non-zero, make CL zero

    # Before noise: where did the contrast appear? 1 = Right, 0 = Left
    side = (CR > 0) * 1
    # stack horizontally (by columns), vstack is vertical = by rows
    stimuli = np.hstack((CL, CR, side))  # 800x3
    # have 1 row that stays raw, before adding the noise: add CL + CR
    raw = np.reshape(np.sum(stimuli[:, 0:2], axis=1), (T,1))
    stimuli = np.hstack((stimuli, raw))  # T x 4

    # GAUSSIAN NOISE: add some small Uniform(0,1) noise to the 2 inputs: before creating y from X
    stimuli[:,0:2] = stimuli[:,0:2] + (np.random.normal(0,1, size = (T, 2)) * 0.1)

    # UNIFORM NOISE:
    #stimuli[:, 0:2] = stimuli[:, 0:2] + (np.random.rand(T, 2) * 0.2)

    # ground truth outcome feedback: y
    # we dont know the ground truth function so we assume its L (0) when contrast is L and R (1) when contrast is R
    W = np.ones((2, 1))
    W[0, :] = -1  # Left side
    W[1, :] = 1  # Right side
    y = stimuli[:,0:2] @ W
    y[y < 0] = 0  # L
    y[y > 0] = 1  # R

    labels = y

    # for each mouse there is a train
    # inside each list element is a list with 2 arrays
    trainsets.append([stimuli, labels])


testsets = []
# testsets for the 100 mice: create separately to keep classes CL/CR == 0 equal (400 each)
for m in range(mice):
    # randomly draw contrasts on L and R side between 0-1 (rescaled from 0-100) with 0 being more likely
    CR = np.random.rand(T, 1)  # 800x1
    indices = np.random.choice(np.arange(T), replace=False, size=round(T/2))
    CR[indices] = 0
    CL = np.random.rand(T, 1)  # 800x1
    CL[CR > 0] = 0  # when CR is non-zero, make CL zero

    # Before noise: where did the contrast appear? 1 = Right, 0 = Left
    side = (CR > 0) * 1
    # stack horizontally (by columns), vstack is vertical = by rows
    stimuli = np.hstack((CL, CR, side))  # 800x2

    # have 1 row that stays raw, before adding the noise: add CL + CR
    raw = np.reshape(np.sum(stimuli[:, 0:2], axis=1), (T, 1))
    stimuli = np.hstack((stimuli, raw))  # T x 4

    # GAUSSIAN NOISE: add some small Uniform(0,1) noise to the 2 inputs: before creating y from X
    stimuli[:, 0:2] = stimuli[:, 0:2] + (np.random.normal(0, 1, size=(T, 2)) * 0.1)

    # UNIFORM NOISE:
    #stimuli[:, 0:2] = stimuli[:, 0:2] + (np.random.rand(T, 2) * 0.2)

    # ground truth outcome feedback: y
    # we dont know the ground truth function so we assume its L (0) when contrast is L and R (1) when contrast is R
    W = np.ones((2, 1))
    W[0, :] = -1  # Left side
    W[1, :] = 1  # Right side
    y = stimuli[:,0:2] @ W
    y[y < 0] = 0  # L
    y[y > 0] = 1  # R

    labels_test = y
    stimuli_test = stimuli

    testsets.append([stimuli_test, labels_test])



# Rate Code -  Feedforward Network
import torch.nn as nn
import torch.nn.functional as F

# My own weight initialization function:
def he_initialize(shape):
    """
    Kaiming He normalization: sqrt(2 / fan_in)
    better for ReLu activation fn
    """
    fan_in = shape[1]  # 2nd tensor dimension (size[l-1])
    # eg shape would be torch.Size([200, 220]
    w = torch.randn(shape) * np.sqrt(2. / fan_in)
    w.requires_grad = True
    return w

def init_weights(m):
    if type(m) == nn.Linear:
        # torch.nn.init.kaiming_normal_(m, mode='fan_in', nonlinearity='relu')
        m.weight.data = he_initialize(m.weight.data.size())


# 4 inputs  for CL and CR stimuli, last action, last outcome
# output unit: binary so 1 sigmoid

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # input to H1
        self.fc1 = torch.nn.Linear(4, 50) # bias = True is default
        # relu activation
        self.relu = torch.nn.ReLU()
        # H1 to output
        self.fc2 = torch.nn.Linear(50, 1) # bias = True is default
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # all the self. functions below have to be defined in _init_
        hidden1 = self.fc1(x)
        relu = self.relu(hidden1)
        output = self.fc2(relu)
        output = self.sigmoid(output)  # output is one probability

        return output


net = Net()

net.apply(init_weights)


# 3. Train the network
# Loss function and optimizer
# good for binary outputs: nn.BCE.loss() which does not include the sigmoid yet
criterion = nn.BCELoss()
#optimizer = optim.SGD(net.parameters(), lr=L_param, weight_decay=0)
optimizer = optim.Adam(net.parameters(), lr=L_param)
BATCH_SIZE = 1


for m in range(mice):

    # for now just 1 mouse
    # m = 0

    # reset the network parameters anew (otherwise it improves from mouse to mouse?)
    #net = Net()
    net.apply(init_weights)

    # for plotting below: per mouse m
    stimuli, ground_truth = trainsets[m]

    input_train, y_train = trainsets[m]
    input_train = torch.from_numpy(input_train[:,0:2])
    y_train = torch.from_numpy(y_train)
    # input_train: 800x2, y_train: 800x1
    # creates equal batches of size BATCH_SIZE out of input_train
    input_split = torch.split(input_train, BATCH_SIZE, dim=0)  # 800 chunks
    labels_split = torch.split(y_train, BATCH_SIZE, dim=0)  # 800 chunks
    # merge next to each other the chunks, so Trainloader[0] is 1 batch
    Trainloader = merge(input_split, labels_split)

    input_test, y_test = testsets[m]
    input_test = torch.from_numpy(input_test[:,0:2])
    y_test = torch.from_numpy(y_test)
    # input_train: 800x2, y_train: 800x1
    # creates equal batches of size BATCH_SIZE out of input_train
    inputT_split = torch.split(input_test, BATCH_SIZE, dim=0)  # 800 chunks
    labelsT_split = torch.split(y_test, BATCH_SIZE, dim=0)  # 800 chunks
    # merge next to each other the chunks, so Trainloader[0] is 1 batch
    Testloader = merge(inputT_split, labelsT_split)

    # initiliaze some variables for the mouse
    avg_train_loss = np.zeros(T)
    avg_valid_loss = np.zeros(T)
    #batch_loss = np.zeros([iter, len(Trainloader)])  # 10,000 x 18 matrix, change data type? or do i need to specify?
    #batch_loss_valid = np.zeros([iter, len(Testloader)])  # 10,000 x 18 matrix, change data type? or do i need to specify?
    batch_loss = np.zeros([len(Trainloader)])  # as long as trials now
    batch_loss_valid = np.zeros([len(Testloader)])


    #for epoch in range(iter):  # not sure how many epochs we need for network to learn

    action = np.zeros(len(Trainloader))
    accuracy = np.zeros(len(Trainloader))

    net = net.float()
    net.train()  # prep model for training

    for i, data in enumerate(Trainloader):  # cycle through trial i

        # trial_train is 1x2, y_train is 1 scalar (ground truth)
        # data = Trainloader[i]
        trial_train, y_train = data
        y_train = y_train.view(1)  # reshape to ([1])

        # add i.i.d. noise on each trial, anew
        # trial_train = trial_train + (np.random.rand(BATCH_SIZE, 2) * noise)

        # 1x4 input for net: CL, CR, LA, LO
        if i == 0:
            # second 2 rows are NA if that works
            trial_train = np.hstack((trial_train.split(1, dim=1), 0.5, 0.5))  # or could do 0 0
            trial_train = torch.from_numpy(trial_train)
        else:
            # split and joined as np array
            trial_train = np.hstack((trial_train.split(1, dim=1), action[i - 1], accuracy[i - 1]))
            trial_train = torch.from_numpy(trial_train)

        # zero the parameter gradients
        optimizer.zero_grad()
        y_hat = net(trial_train.float())

        # y_train is 0/1, y_hat is sigmoid probability
        # adjust y_train shape  in future
        loss = criterion(y_hat, y_train.float())

        # Action: round probs to 0/1
        action[i] = torch.round(y_hat)
        accuracy[i] = (round(torch.round(y_hat).item()) == round(y_train.item())) * 1

        # loss is just a scalar
        loss.backward()
        optimizer.step()

        # change from anything that has 'epoch' in it, to trial i
        batch_loss[i] = loss.item()  # average distance of 20 y's to y_hat, averaged per batch, across 500 datasets

    # After the 80k trial loop:
    # avg_train_loss[i] = np.mean(batch_loss[epoch, :])

    ground_truth = np.reshape(ground_truth, (T,))
    action = np.reshape(action, (T,))

    # LA = action shifted by 1, L0 = accuracy shifted by 1
    # trail i, CL, CR, y, action, acc, LA, LO
    joint = pd.DataFrame(
        dict(mouse=m, trial=range(1, len(Trainloader) + 1), CL=stimuli[:, 0], CR=stimuli[:, 1], raw=stimuli[:, 3],
             side=stimuli[:, 2],
             ground_truth=ground_truth, action=action, accuracy=accuracy, loss=batch_loss,
             LA=action, LO=accuracy))
    joint['LA'] = joint['LA'].shift(+1)
    joint['LO'] = joint['LO'].shift(+1)


    #joint.to_csv("mouse_%.0f_data.csv" % (m + 1), index=False)
    joint.to_csv("mouse_%.0f_data_SGD_nhid50_LR001.csv" % (m+1), index=False)

    del joint, Trainloader, Testloader, action, ground_truth, accuracy, batch_loss, loss, stimuli, input_train, input_test



# put all 80,000 learning trails for mouse m into Mdata
# Mdata.append([joint])

# save this data somehow, and then use below for plotting.
# import pickle
# with open('Mouse_data.data', 'wb') as filehandle:
#     # store the data as binary data stream
#     pickle.dump(Mdata, filehandle)

#     ######################
#     # validate the model: Evaluation on independent testsets (same distribution) #
#     ######################
#     actionT = np.zeros(len(Trainloader))
#     accuracyT = np.zeros(len(Trainloader))
#
#     net.eval()
#     # with 100.000 testset, these are 200 batch runs
#     net = net.float()
#     with torch.no_grad():
#         # cycle through test batches too
#         for l, data in enumerate(Testloader):
#
#             # data = Trainloader[i]
#             trial_test, y_test = data
#             y_test = y_test.view(1)  # reshape to ([1])
#
#             # add i.i.d. noise on each trial, anew
#             trial_test = trial_test + (np.random.rand(BATCH_SIZE, 2) * noise)
#
#             if l == 0:
#                 # second 2 rows are NA if that works
#                 trial_test = np.hstack((trial_test.split(1, dim=1), 0.5, 0.5))
#                 trial_test = torch.from_numpy(trial_test)
#             else:
#                 # split and joined as np array
#                 trial_test = np.hstack((trial_test.split(1, dim=1), actionT[l - 1], accuracyT[l - 1]))
#                 trial_test = torch.from_numpy(trial_test)
#
#             y_hat_test = net(trial_test.float())
#             loss2 = criterion(y_hat_test, y_test.float())
#
#             actionT[l] = torch.round(y_hat_test)
#             accuracyT[l] = (round(torch.round(y_hat_test).item()) == round(y_test.item())) * 1
#
#             # can look at loss over trials for epoch 1
#             batch_loss_valid[
#                 epoch, l] = loss2.item()  # average distance of 20 y's to y_hat, averaged per batch, across 500 datasets
#
#     # calculate average loss over an epoch: mean per epoch is still going to be in the 400s if the sum(Loss) is taken above for a batch (batch size = 500 makes it be around 400)
#     avg_valid_loss[epoch] = np.mean(batch_loss_valid[epoch, :])
#
#     # print the current train and val loss
#     epoch_len = len(str(iter))
#     print_msg = (f'[{epoch:>{epoch_len}}/{iter:>{epoch_len}}] ' +
#                  f'train_loss: {avg_train_loss[epoch]:.5f} ' +
#                  f'valid_loss: {avg_valid_loss[epoch]:.5f}')
#
#     print(print_msg)
#
# print('Finished Training and Testing')
#
    #
    # # x labels in trials
    # lab = np.linspace(0, 100, 11) * 800
    # # Plot 1) Visualize the training + testing error  - and save
    # plt.plot(range(1, len(avg_train_loss) + 1), avg_train_loss, 'g', label='Train loss (avg. batch)')
    # plt.plot(range(1, len(avg_valid_loss) + 1), avg_valid_loss, 'b', label='Validation loss (avg. batch)')
    # # every 10 epochs
    # plt.xticks(ticks=np.linspace(0, 100, 11), labels = lab.round())
    # plt.title('Training and Validation loss (single artificial mouse)')
    # plt.xlabel('Trials (100 epochs)')
    # plt.ylabel('Loss')
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    #
    # plt.savefig('../Loss_across_epochs_mouse_m%.0f_epochs%.0f.pdf' % (m, iter))
    # plt.close()
    #


### Data prep
m = 0
joint = pd.read_csv("mouse_%.0f_data_Gaussian.csv" % (m + 1), sep=",", header=0)

joint = pd.read_csv("mouse_%.0f_data_nhid50_LR001.csv" % (m + 1), sep=",", header=0)


# First 1000 trials only
joint = joint.iloc[0:1000, :]

# Individual mouse
# # Loss plots over trials:

plt.plot(range(1, joint.shape[0] + 1), joint['loss'], 'g', label='Train loss')
# plt.ylim(bottom=0)
plt.title('Loss across all trials' , fontsize=16)
plt.xlabel('Trials')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.savefig('../RL/Loss_over_1000Trials_mouse1.pdf')
plt.close()


# Accuracy plot:
btrials = 5 #1000
blocks = joint.shape[0]//btrials  # make sure division results in int, not float!
# just with changed data type
#joint['accuracy'] = joint['accuracy'].astype(np.int64)
# take df column, transfer to numpy, then operate as normal with reshape
block_acc =  np.reshape(joint['accuracy'].to_numpy(), (btrials, blocks), order = 'F') # 1000x80  # order = 'F' means the reshaping happens by putting entries below each other row-wise first, then columns
# same as this now: np.reshape(accuracy, (btrials, blocks))
block_means = np.mean(block_acc, axis = 0)
# Accuracy: Needs to be binned to get a prob.
plt.plot(block_means)
# plt.ylim(bottom=0)
plt.title('Accuracy', fontsize=16)
plt.xlabel('Blocks (1 block = 5 trials)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.savefig('../RL/Accuracy_over_blocks_5trials.pdf')
plt.close()



# Choices (colour) by contrast and block
btrials = 1000
blocks = T//btrials  # make sure division results in int, not float!

# all 1000x80 = 80 blocks
block_choices =  np.reshape(joint['action'].to_numpy(), (btrials, blocks), order = 'F') # 1000x80  # order = 'F' means the reshaping happens by putting entries below each other row-wise first, then columns
block_sides =  np.reshape(joint['side'].to_numpy(), (btrials, blocks), order = 'F') # 1000x80  # order = 'F' means the reshaping happens by putting entries below each other row-wise first, then columns
# one column for both stimuli CL + CR:
block_sides[block_sides == 0] = -1
block_contrasts =  np.reshape(joint['raw'].to_numpy(), (btrials, blocks), order = 'F') * block_sides

# I'm gonna binnit, into levels (since our contrast is 0-100)
bins = np.linspace(block_contrasts.min(), block_contrasts.max(), num = 20)
# compute the mean of bins (blocks)
pff = np.zeros([19, blocks])
for b in range(blocks):
    for u in range(19): # contrasts levels  np.logical_and(x>1, x<4)
        # boolean AND for selection of rows, in column 'b'
        # block_contrasts[(block_contrasts[:, b] > bins[u]) & (block_contrasts[:, b] < bins[u + 1]), b]
        pff[u,b]  = np.mean(block_choices[(block_contrasts[:,b] > bins[u]) & (block_contrasts[:,b] < bins[u+1]), b])

# pff is 19x80 = contrasts x blocks
# pff is ordered as -1 to +1 for contrasts, starting with left (-1) and then to right (+1)
plt.imshow(pff, origin = 'lower',  extent = [0, 80, -1, 1], aspect = 30)
plt.xlabel('Blocks')
plt.ylabel('Contrasts')
plt.show()

plt.savefig('../RL/Mean_choices_over_contrast_blocks.pdf')
plt.close()




# Coefficients (log regression) for contrast L/R and prev choices :

m = 0
#joint = pd.read_csv("mouse_%.0f_data_Gaussian.csv" % (m + 1), sep=",", header=0)
#joint = pd.read_csv("mouse_%.0f_data.csv" % (m + 1), sep=",", header=0)
joint = pd.read_csv("mouse_%.0f_data_nhid50_LR001.csv" % (m + 1), sep=",", header=0)



T = 80000
btrials = 1000
blocks = T//btrials  # make sure division results in int, not float!

# all 1000x80 = 80 blocks
block_prevchoices = np.reshape(joint['LA'].to_numpy(), (btrials, blocks), order = 'F')
block_prevrewards = np.reshape(joint['LO'].to_numpy(), (btrials, blocks), order = 'F')
block_prevchoices[block_prevchoices == 0] = -1  # make -1/+1 for GLM below
block_prevrewards[block_prevrewards == 0] = -1

block_choices =  np.reshape(joint['action'].to_numpy(), (btrials, blocks), order = 'F') # 1000x80  # order = 'F' means the reshaping happens by putting entries below each other row-wise first, then columns
block_sides =  np.reshape(joint['side'].to_numpy(), (btrials, blocks), order = 'F') # 1000x80  # order = 'F' means the reshaping happens by putting entries below each other row-wise first, then columns
# one column for both stimuli CL + CR:
block_sides[block_sides == 0] = -1
block_contrasts =  np.reshape(joint['raw'].to_numpy(), (btrials, blocks), order = 'F') * block_sides

# single contrasts with noise from above
block_contrastsL =  np.reshape(joint['CL'].to_numpy(), (btrials, blocks), order = 'F') * block_sides
block_contrastsR =  np.reshape(joint['CR'].to_numpy(), (btrials, blocks), order = 'F') * block_sides
# multiplying both with -1/+1 results in the coefficients both being positive? otherwise CL would be negative, as larger contrast values (e.g., 0.25) relate to smaller y values (0)
# while for CR, larger values relate to larger y values (1)

import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression, LinearRegression
# glm_binom = sm.GLM(block_choices[:,b], XX, family=sm.families.Binomial())
# res = glm_binom.fit()
# print(res.summary())
# betas = res.params

#betas = np.zeros([blocks, 3])
betas = np.zeros([blocks, 5])
intercept = np.ones([1000,1])

for b in range(blocks):

    # 'contrast','pchoice','preward','int'  = columns in an array that is: 1000x5
    #XX = np.column_stack((block_contrastsL[:, b], block_contrastsR[:, b]))
    #XX = np.column_stack((block_contrastsL[:,b],  block_contrastsR[:,b],  block_prevchoices[:,b],  block_prevrewards[:,b],  block_prevchoices[:,b]*block_prevrewards[:,b]))
    XX = np.column_stack((block_contrasts[:, b], block_prevchoices[:, b], block_prevrewards[:, b], block_prevchoices[:, b]*block_prevrewards[:, b]))

    XX[np.isnan(XX)] = 0
    #XX = sm.tools.tools.add_constant(XX) # intercept
    XX = np.hstack((intercept, XX))
    # XX a matrix:
    model = LogisticRegression(fit_intercept=False).fit(XX, block_choices[:,b]) # already contains intercept (1's)
    betas[b,:] = model.coef_
    # 1st col is intercepts
    del XX

# interpretation: when prev choice was positive (+1) = Right, more likely to be Rightward choice (1)
# than when prev choice was negative (-1) = Left

names = ['Intercept', 'Contrast L', 'Contrast R']

names = ['Intercept', 'Contrast', 'Prev choice', 'Prev reward', 'Interaction (WSLS)']

plt.plot(betas[:, 0:5])
plt.xlabel('Blocks')
#plt.xticks(ticks=bins)
plt.ylabel('Coefficients')
plt.grid(True)
plt.tight_layout()
plt.legend(labels = names[0:5])
plt.show()

plt.savefig('../RL/Mouse_coeffs_WSLS_m%.0f.pdf' % m)
plt.close()


plt.plot(betas[:, 0:3])
plt.xlabel('Blocks')
#plt.xticks(ticks=bins)
plt.ylabel('Coefficients')
plt.grid(True)
plt.tight_layout()
plt.legend(labels = names[0:3])
plt.show()

plt.savefig('../RL/Mouse_coeffs_ContrastLR_m%.0f.pdf' % m)
plt.close()




#  Psychometric curves:

m = 1
joint = pd.read_csv("mouse_%.0f_data.csv" % (m + 1), sep=",", header=0)


# ## Data prep
left = joint[(joint['side'] == -1)]
right = joint[(joint['side'] == 1)]
# sort values by contrast
left = left.sort_values(by='CL', ascending=False) # large to small
right = right.sort_values(by='CR', ascending=True)
# make CL contrast negative [-1,0] for plotting one 1 axis
left['CL'] = left['CL'] * (-1)
# delete the CR col in left, and CL col in right
left = left.drop(['CR'], axis=1)
right = right.drop(['CL'], axis=1)
# rename both contrasts into 'contrast'
left = left.rename(columns={'CL': 'contrast'})
right = right.rename(columns={'CR': 'contrast'})
#Left on top, right below, so that contrast increases from -1 to +1
psych = pd.concat([left, right], axis=0)
# Get Means for bins:   #bins = np.linspace(round(min(psych['contrast']),6), max(psych['contrast']), num = 20)
bins = np.linspace(-1, +1, num=20)  # keep them fixed, independent of mouse
psych['binned'] = pd.cut(psych['contrast'], bins = bins)
means = psych.groupby('binned')['action'].mean()
counted = psych.groupby('binned')['trial'].count()
ses = psych.groupby('binned')['action'].std()
errors = ses / np.sqrt(counted)

# contrast from -1 to 1 just for plotting, with 0 in middle
new_a = np.delete(bins, 0)
plt.plot(new_a, means, 'o', markersize=4, color = 'darkred')
#plt.plot(x, means)
plt.errorbar(new_a, means, yerr=errors, color = 'orange')
plt.title('Mouse #%.0f - after many trials' % (m+1), fontsize=16)
plt.xlabel('(L)    Contrast     (R)')
#plt.xticks(ticks=bins)
plt.ylabel('Rightward (%)')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.savefig('../RL/Mouse_psychometric_m%.0f.pdf' % m)
plt.close()



# Multiple mice psychometric curves:
bins = np.linspace(-1, +1, num=20)  # keep them fixed, independent of mouse
new_a = np.delete(bins, 0)
mean_all_contrasts = np.zeros(mice)

for m in range(mice):

    ### Data prep
    joint = pd.read_csv("mouse_%.0f_data.csv" % (m + 1), sep=",", header=0)
    joint = joint.iloc[0:1000, :]

    mean_all_contrasts[m] = np.mean(joint['action'])

    left = joint[(joint['side'] == 0)]
    right = joint[(joint['side'] == 1)]
    # sort values by contrast
    left = left.sort_values(by='CL', ascending=False) # large to small
    right = right.sort_values(by='CR', ascending=True)
    # make CL contrast negative [-1,0] for plotting one 1 axis
    left['CL'] = left['CL'] * (-1)
    # delete the CR col in left, and CL col in right
    left = left.drop(['CR'], axis=1)
    right = right.drop(['CL'], axis=1)
    # rename both contrasts into 'contrast'
    left = left.rename(columns={'CL': 'contrast'})
    right = right.rename(columns={'CR': 'contrast'})
    #Left on top, right below, so that contrast increases from -1 to +1
    psych = pd.concat([left, right], axis=0)
    # Get Means for bins:   #bins = np.linspace(round(min(psych['contrast']),6), max(psych['contrast']), num = 20)
    psych['binned'] = pd.cut(psych['contrast'], bins = bins)
    means = psych.groupby('binned')['action'].mean()
    counted = psych.groupby('binned')['trial'].count()
    ses = psych.groupby('binned')['action'].std()
    errors = ses / np.sqrt(counted)


    #plt.plot(new_a, means, 'o', markersize=4)
    #plt.plot(x, means)
    plt.errorbar(new_a, means, yerr=errors)
    plt.title('Multiple mice - after 1000 trials', fontsize=16)
    plt.xlabel('(L)    Contrast     (R)')
    #plt.xticks(ticks=bins)
    plt.ylabel('Rightward (%)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


plt.savefig('../RL/Multiple_mice_psychometric_1000trials.pdf')

plt.close()



mean_all_contrasts.min()
mean_all_contrasts.max()
mean_all_contrasts.mean()




## Compute the mean across all contrasts for each mouse, and then we can compare. After all trials, and after 800.






