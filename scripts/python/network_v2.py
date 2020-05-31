# this is the same script as before but we use more data.

import numpy as np 
import matplotlib.pyplot as plt
import torch
import h5py
import torch.nn as nn
from torch.utils import data

# number of available gpu
ngpu = 1 
batch_size = 128
decayRate = 0.95
lr = 0.001

dataRoot = "../../data"

# this is a for loop to load different files
Mats = 0
Labels = 0

for ii in range(1,6):
    labelFiles = "labels-{}.h5".format(str(ii))
    matFiles = "matrices-{}.h5".format(str(ii))

    labelsh5 = h5py.File(dataRoot+"/"+labelFiles, 'r')
    labels = labelsh5['labels'][:].astype(np.int64)-1 
    # classes from 0 to C-1
    
    matsh5 = h5py.File(dataRoot+"/"+matFiles, 'r')
    mats = matsh5['matrices'][:]

    mats = mats.reshape((1550, -1, labels.shape[0]))    
    mats = np.transpose(mats, (2, 1, 0 ))

    if type(Mats) == int :
        Mats = mats
        Labels = labels
    else: 
        Mats = np.concatenate([Mats, mats], axis = 0)
        Labels = np.concatenate([Labels, labels], axis = 0)

savePath = "network_test.pt"

nSamples = Labels.shape[0]

nTrainSamples = int(0.95*nSamples)
nTestSamples = int(0.05*nSamples)

# #random shuffle
# idxShuffle = np.linspace(0, nSamples-1, nSamples).astype(np.int32)
# np.random.shuffle(idxShuffle)

# Labels = Labels[idxShuffle]
# Mats   = Mats[idxShuffle,:,:]

# dims of mats is (Nsamples, NChannels, Nsequence)

output = torch.tensor(Labels)
input  = torch.Tensor(Mats)

trainDataSet = data.TensorDataset(input[:nTrainSamples, :,:], output[:nTrainSamples]) 
testDataSet = data.TensorDataset(input[-nTestSamples:, :,:], output[-nTestSamples:]) 


dataloaderTrain = torch.utils.data.DataLoader(trainDataSet, batch_size=batch_size,
                                         shuffle=True)

dataloaderTest = torch.utils.data.DataLoader(testDataSet, batch_size=batch_size)

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


##############################################################
# We specify the networks (this are quite simple, we should be
# able to build some more complex)

## copy paste from the Zou 2019 model 
# here is the residue block
class _ResidueModule(torch.nn.Module):

    def __init__(self, channel_count):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv1d(channel_count, channel_count, 1),
            torch.nn.BatchNorm1d(channel_count),
            torch.nn.ReLU(),
            torch.nn.Conv1d(channel_count, channel_count, 1),
            torch.nn.BatchNorm1d(channel_count),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        return x + self.layers(x)


class _Model(torch.nn.Module):
    """A neural network model to predict phylogenetic trees."""

    def __init__(self):
        """Create a neural network model."""
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(80, 80, 1, groups=20),
            torch.nn.BatchNorm1d(80),
            torch.nn.ReLU(),
            torch.nn.Conv1d(80, 32, 1),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            _ResidueModule(32),
            _ResidueModule(32),
            torch.nn.AvgPool1d(2),
            _ResidueModule(32),
            _ResidueModule(32),
            torch.nn.AvgPool1d(2),
            _ResidueModule(32),
            _ResidueModule(32),
            torch.nn.AvgPool1d(2),
            _ResidueModule(32),
            _ResidueModule(32),
            torch.nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = torch.nn.Linear(32, 3)

    def forward(self, x):
        """Predict phylogenetic trees for the given sequences.

        Parameters
        ----------
        x : torch.Tensor
            One-hot encoded sequences.

        Returns
        -------
        torch.Tensor
            The predicted adjacency trees.
        """
        x = x.view(x.size()[0], 80, -1)
        x = self.conv(x).squeeze(dim=2)
        return self.classifier(x)

###############################################

# specify loss function
criterion = torch.nn.CrossEntropyLoss(reduction='sum')
# criterion = torch.nn.CrossEntropyLoss()

# define the model 
model = _Model().to(device)

# specify loss function
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
stepSizeEpochs = 10
step_size = stepSizeEpochs*(nSamples//batch_size)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                             step_size=stepSizeEpochs, gamma=decayRate)
# TODO: add scheduler here!! 


n_epochs = 1000

print("final learning rate = {:.8f}".format((lr*np.power(decayRate, n_epochs//stepSizeEpochs))))

for epoch in range(1, n_epochs+1):
    # monitor training loss
    train_loss = 0.0
    
    ###################
    # train the model #
    ###################
    for genes, quartets_batch in dataloaderTrain:
        #send to the device (either cpu or gpu)
        genes, quartets_batch = genes.to(device), quartets_batch.to(device)
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        quartetsNN = model(genes)
        # calculate the loss
        loss = criterion(quartetsNN, quartets_batch)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item()
    
     # changing the learning rate following the schedule (for next step)
    exp_lr_scheduler.step()        
    # print avg training statistics 
    train_loss = train_loss/len(dataloaderTrain)
    print('Epoch: {} \tTraining Loss: {:.6f} \t lr: {:.6f}'.format(
        epoch, 
        train_loss,
        optimizer.param_groups[0]['lr']
        )) 

    correct = 0
    total = 0
    with torch.no_grad():
        for genes, quartets_batch in dataloaderTest:
            #moving data to GPU
            genes, quartets_batch = genes.to(device), quartets_batch.to(device)

            quartetsNN = model(genes)

            _, predicted = torch.max(quartetsNN.data, 1)
            total += quartets_batch.size(0)
            correct += (predicted == quartets_batch).sum().item()

        print('Epoch: {} \t Test accuracy: {:.6f}'.format(
            epoch, 
            correct/total
            )) 


torch.save(model.state_dict(), savePath)


correct = 0
total = 0
with torch.no_grad():
    for genes, quartets_batch in dataloaderTrain:
        #moving data to GPU
        genes, quartets_batch = genes.to(device), quartets_batch.to(device)

        quartetsNN = model(genes)

        _, predicted = torch.max(quartetsNN.data, 1)
        total += quartets_batch.size(0)
        correct += (predicted == quartets_batch).sum().item()

    print('Epoch: {} \t Training accuracy: {:.6f}'.format(
        epoch, 
        correct/total
        )) 