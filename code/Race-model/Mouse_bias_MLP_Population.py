
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
    2 output units (softmax)
    noise - add some noise to input somehow


    Output graphs:
    1. Learning curve (accuracy/loss)
    2. Actions (% Rightward) as a function of contrast R and contrast L (accumulated and sorted across all trials)
    3. same as above for % Leftward
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
# trials
T = 800
mice = 100
L_param = 0.001
iter = 100  # or 1
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
    stimuli = np.hstack((CL, CR, side))  # 800x2


    # NO NOISE HERE YET
    # is like saying there was some uncertainty in perception of contrasts on either side.
    #stimuli[:,0:2] = stimuli[:,0:2] + (np.random.rand(T, 2) * noise)

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

    # add some small Uniform(0,1) noise to the 2 inputs: before creating y from X
    # is like saying there was some uncertainty in perception of contrasts on either side.
    stimuli[:,0:2] = stimuli[:,0:2] + (np.random.rand(T, 2) * noise)

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
        self.fc1 = torch.nn.Linear(202, 200, bias=False)
        # relu activation
        self.relu = torch.nn.ReLU()
        # H1 to output
        self.fc2 = torch.nn.Linear(200, 1, bias=False)
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
optimizer = optim.Adam(net.parameters(), lr=L_param)



BATCH_SIZE = 1

# for m in range(mice):

# inputs already contains noise
# for now just 1 mouse
m = 1

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
avg_train_loss = np.zeros(iter)
avg_valid_loss = np.zeros(iter)
batch_loss = np.zeros([iter, len(Trainloader)])  # 10,000 x 18 matrix, change data type? or do i need to specify?
batch_loss_valid = np.zeros([iter, len(Testloader)])  # 10,000 x 18 matrix, change data type? or do i need to specify?

# Learning data append with each epoch (100 list elements)
Ldata = []


for epoch in range(iter):  # not sure how many epochs we need for network to learn

    action = np.zeros(len(Trainloader))
    accuracy = np.zeros(len(Trainloader))

    net = net.float()
    net.train()  # prep model for training
    for i, data in enumerate(Trainloader):  # Trainloader contains trial-by-trail data for mouse m

        ## HERE: now take the contrast of the side the stimuli was on
        ## from trial_train and use that as the mean for a PDF of a Gaussain, that I sample 100 linearly spaced
        ## then somehow order so that it fits the ordered units?
        ## add noise to this sampled values (uniform)
        ## later to the mean of the Guassian

        # trial_train is 1x2, y_train is 1 scalar (ground truth)
        # data = Trainloader[i]
        trial_train, y_train = data
        y_train = y_train.view(1) # reshape to ([1])

        mu = trial_train[trial_train > 0]

        # key is the linearly spaced inputs: where they lie
        # If i assume ordered 100 units, then mean of 0.72 is correpsonding to the 72th unit?
        # and hence I want to evaluate more linear spaces (71) below the mean than above? (100 - x to give upper, and 100 - (100-x) to give lower?) and put inputs where?
        # equal distance steps from the mean, no matter if negative. e.g., 0.7, 0.65, 0.6 




        # 1x4 input for net: CL, CR, LA, LO
        if i == 0:
            # second 2 rows are NA if that works
            trial_train = np.hstack((trial_train.split(1, dim=1), 0.5, 0.5))
            trial_train = torch.from_numpy(trial_train)
        else:
            # split and joined as np array
            trial_train = np.hstack((trial_train.split(1, dim = 1), action[i-1], accuracy[i-1]))
            trial_train = torch.from_numpy(trial_train)

        # zero the parameter gradients
        optimizer.zero_grad()
        y_hat = net(trial_train.float())

        # y_train is 0/1, y_hat is sigmoid probability
        # adjust y_train shape in future
        loss = criterion(y_hat, y_train.float())

        # Action: round probs to 0/1
        action[i] = torch.round(y_hat)
        accuracy[i] = (round(torch.round(y_hat).item()) == round(y_train.item())) * 1

        # loss is just a scalar
        loss.backward()
        optimizer.step()

        # can look at loss over trials for epoch 1
        batch_loss[epoch, i] = loss.item()  # average distance of 20 y's to y_hat, averaged per batch, across 500 datasets

    # for loss per epoch curve
    avg_train_loss[epoch] = np.mean(batch_loss[epoch, :])



    # Store data after training loop for epoch = 0
    # in a 800x6 tensor, that can be put into epoch slice: 800x6xepochs

    # reshape for pandas
    # ground_truth = np.reshape(ground_truth, (800,))
    # action = np.reshape(action, (800,))
    #
    # # LA = action shifted by 1, L0 = accuracy shifted by 1
    # # trail i, CL, CR, y, action, acc, LA, LO
    # joint = pd.DataFrame(dict(trial=range(len(Trainloader)), CL = stimuli[:, 0], CR= stimuli[:, 1], side = stimuli[:,2],
    #                        ground_truth = ground_truth, action = action, accuracy = accuracy,
    #                          LA = action, LO = accuracy))
    # joint['LA'] = joint['LA'].shift(+1)
    # joint['LO'] = joint['LO'].shift(+1)
    # Ldata.append([joint])
    #
    #
    # # Psychometric curves:
    # ## Data prep
    # left = joint[(joint['side'] == 0)]
    # right = joint[(joint['side'] == 1)]
    # # sort values by contrast
    # left = left.sort_values(by='CL', ascending=False) # large to small
    # right = right.sort_values(by='CR', ascending=True)
    # # make CL contrast negative [-1,0] for plotting one 1 axis
    # left['CL'] = left['CL'] * (-1)
    # # delete the CR col in left, and CL col in right
    # left = left.drop(['CR'], axis=1)
    # right = right.drop(['CL'], axis=1)
    # # rename both contrasts into 'contrast'
    # left = left.rename(columns={'CL': 'contrast'})
    # right = right.rename(columns={'CR': 'contrast'})
    # #Left on top, right below, so that contrast increases from -1 to +1
    # psych = pd.concat([left, right], axis=0)
    # # Get Means for bins:
    # bins = np.linspace(round(min(psych['contrast']),6), max(psych['contrast']), num = 20)
    # psych['binned'] = pd.cut(psych['contrast'], bins = bins)
    # means = psych.groupby('binned')['action'].mean()
    # counted = psych.groupby('binned')['trial'].count()
    # ses = psych.groupby('binned')['action'].std()
    # errors = ses / np.sqrt(counted)
    #
    #
    # # contrast from -1 to 1 just for plotting, with 0 in middle
    # x = np.linspace(-1, +1, num=19)
    # plt.plot(x, means, 'o', markersize=4, color = 'darkred')
    # #plt.plot(x, means)
    # plt.errorbar(x, means, yerr=errors, color = 'orange')
    #
    # plt.title('Mouse #%.0f - epoch%.0f' % (m+1, epoch), fontsize=16)
    # plt.xlabel('(L)    Contrast     (R)')
    # plt.ylabel('Rightward (%)')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    #
    # plt.savefig('../Mouse_psychometric_m%.0f_epoch%.0f.pdf' % (m, epoch))
    # plt.close()
    # ground_truth = np.reshape(ground_truth, (800,))
    # action = np.reshape(action, (800,))
    #
    # # LA = action shifted by 1, L0 = accuracy shifted by 1
    # # trail i, CL, CR, y, action, acc, LA, LO
    # joint = pd.DataFrame(dict(trial=range(len(Trainloader)), CL = stimuli[:, 0], CR= stimuli[:, 1], side = stimuli[:,2],
    #                        ground_truth = ground_truth, action = action, accuracy = accuracy,
    #                          LA = action, LO = accuracy))
    # joint['LA'] = joint['LA'].shift(+1)
    # joint['LO'] = joint['LO'].shift(+1)
    # Ldata.append([joint])
    #
    #
    # # Psychometric curves:
    # ## Data prep
    # left = joint[(joint['side'] == 0)]
    # right = joint[(joint['side'] == 1)]
    # # sort values by contrast
    # left = left.sort_values(by='CL', ascending=False) # large to small
    # right = right.sort_values(by='CR', ascending=True)
    # # make CL contrast negative [-1,0] for plotting one 1 axis
    # left['CL'] = left['CL'] * (-1)
    # # delete the CR col in left, and CL col in right
    # left = left.drop(['CR'], axis=1)
    # right = right.drop(['CL'], axis=1)
    # # rename both contrasts into 'contrast'
    # left = left.rename(columns={'CL': 'contrast'})
    # right = right.rename(columns={'CR': 'contrast'})
    # #Left on top, right below, so that contrast increases from -1 to +1
    # psych = pd.concat([left, right], axis=0)
    # # Get Means for bins:
    # bins = np.linspace(round(min(psych['contrast']),6), max(psych['contrast']), num = 20)
    # psych['binned'] = pd.cut(psych['contrast'], bins = bins)
    # means = psych.groupby('binned')['action'].mean()
    # counted = psych.groupby('binned')['trial'].count()
    # ses = psych.groupby('binned')['action'].std()
    # errors = ses / np.sqrt(counted)
    #
    #
    # # contrast from -1 to 1 just for plotting, with 0 in middle
    # x = np.linspace(-1, +1, num=19)
    # plt.plot(x, means, 'o', markersize=4, color = 'darkred')
    # #plt.plot(x, means)
    # plt.errorbar(x, means, yerr=errors, color = 'orange')
    #
    # plt.title('Mouse #%.0f - epoch%.0f' % (m+1, epoch), fontsize=16)
    # plt.xlabel('(L)    Contrast     (R)')
    # plt.ylabel('Rightward (%)')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    #
    # plt.savefig('../Mouse_psychometric_m%.0f_epoch%.0f.pdf' % (m, epoch))
    # plt.close()


    ## Repeat L/R bias and WSLS



    # # Loss plots (per epoch) over trials:
    # plt.plot(range(1, len(Trainloader) + 1), batch_loss[epoch, :], 'g', label='Train loss')
    # # plt.ylim(bottom=0)
    # plt.title('Loss across 800 trials - Epoch 0', fontsize=16)
    # plt.xlabel('Trials')
    # plt.ylabel('Loss')
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    #
    # plt.savefig('../Loss_trials_mouse1_epoch0.pdf')
    # plt.close()
    #
    # # loss per batch
    # plt.plot(range(1, len(Trainloader) + 1), batch_loss[epoch, :], 'g', label='Train loss')
    # # plt.ylim(bottom=0)
    # plt.title('Loss across 800 trials - Epoch 100', fontsize=16)
    # plt.xlabel('Trials')
    # plt.ylabel('Loss')
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    #
    # plt.savefig('../Loss_trials_epoch100.pdf')
    # plt.close()
    #
    # # accuracy
    # plt.plot(range(1, len(Trainloader) + 1), accuracy, '+',  markersize=4, label='Accuracy')
    # # plt.ylim(bottom=0)
    # plt.title('Correct Actions across trials - Epoch 0', fontsize=16)
    # plt.xlabel('Trials')
    # plt.ylabel('Loss')
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    #
    # plt.savefig('../accuracy_trials_epoch0.pdf')
    # plt.close()



    ######################
    # validate the model: Evaluation on independent testsets (same distribution) #
    ######################
    actionT = np.zeros(len(Trainloader))
    accuracyT = np.zeros(len(Trainloader))

    net.eval()
    # with 100.000 testset, these are 200 batch runs
    net = net.float()
    with torch.no_grad():
        # cycle through test batches too
        for l, data in enumerate(Testloader):

            # data = Trainloader[i]
            trial_test, y_test = data
            y_test = y_test.view(1)  # reshape to ([1])

            if l == 0:
                # second 2 rows are NA if that works
                trial_test = np.hstack((trial_test.split(1, dim=1), 0.5, 0.5))
                trial_test = torch.from_numpy(trial_test)
            else:
                # split and joined as np array
                trial_test = np.hstack((trial_test.split(1, dim=1), actionT[l - 1], accuracyT[l - 1]))
                trial_test = torch.from_numpy(trial_test)

            y_hat_test = net(trial_test.float())
            loss2 = criterion(y_hat_test, y_test.float())

            actionT[l] = torch.round(y_hat_test)
            accuracyT[l] = (round(torch.round(y_hat_test).item()) == round(y_test.item())) * 1

            # can look at loss over trials for epoch 1
            batch_loss_valid[epoch, l] = loss2.item()  # average distance of 20 y's to y_hat, averaged per batch, across 500 datasets


    # calculate average loss over an epoch: mean per epoch is still going to be in the 400s if the sum(Loss) is taken above for a batch (batch size = 500 makes it be around 400)
    avg_valid_loss[epoch] = np.mean(batch_loss_valid[epoch, :])

    # print the current train and val loss
    epoch_len = len(str(iter))
    print_msg = (f'[{epoch:>{epoch_len}}/{iter:>{epoch_len}}] ' +
                 f'train_loss: {avg_train_loss[epoch]:.5f} ' +
                 f'valid_loss: {avg_valid_loss[epoch]:.5f}')

    print(print_msg)

print('Finished Training and Testing')



# Plot 1) Visualize the training + testing error  - and save
plt.plot(range(1, len(avg_train_loss) + 1), avg_train_loss, 'g', label='Train loss (avg. batch)')
plt.plot(range(1, len(avg_valid_loss) + 1), avg_valid_loss, 'b', label='Validation loss (avg. batch)')
#plt.ylim(bottom=0)
plt.title('Training and Validation loss (single artificial mouse)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.savefig('../Loss_across_epochs_mouse_m%.0f_epochs%.0f.pdf' % (m, iter))
plt.close()

