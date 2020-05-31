import numpy as np 
import matplotlib.pyplot as plt
import torch
import h5py
import torch.nn as nn
from torch.utils import data

# number of available gpu
ngpu = 1 
batch_size = 32

dataRoot = "../../data"

labelFiles = "labels-1.h5"
matFiles = "matrices-1.h5"

labelsh5 = h5py.File(dataRoot+"/"+labelFiles, 'r')
labels = labelsh5['labels'][:].astype(np.int64)-1 
# classes from 0 to C-1

matsh5 = h5py.File(dataRoot+"/"+matFiles, 'r')
mats = matsh5['matrices'][:]

nSamples = labels.shape[0]

mats = mats.reshape((1550, nSamples, -1))    
mats = np.transpose(mats, (1, 0, 2))
# dims of mats is (Nsamples, NChannels, Nsequence)

output = torch.tensor(labels)
input  = torch.Tensor(mats)

my_dataset = data.TensorDataset(input, output) 

dataloader = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size,
                                         shuffle=True)

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


## we use a class to storage the different permutations:

class _Permutation():

    def __init__():

        self.permData = np.asarray(list(itertools.permutations(range(4))))

        self.permTaxon1 =  torch.tensor([ 0, 0, 1, 1, 
                                          2, 2, 0, 0, 
                                          2, 2, 1, 1, 
                                          1, 1, 2, 2, 
                                          0, 0, 2, 2, 
                                          1, 1, 0, 0 ], dtype = torch.long)

        self.permTaxon1 =  torch.tensor([ 1, 1, 0, 0, 
                                          2, 2, 1, 1, 
                                          2, 2, 0, 0, 
                                          0, 0, 2, 2, 
                                          1, 1, 2, 2, 
                                          0, 0, 1, 1 ], dtype = torch.long)

        self.permTaxon1 =  torch.tensor([ 2, 2, 0, 0, 
                                          1, 1, 2, 2, 
                                          1, 1, 0, 0, 
                                          0, 0, 1, 1, 
                                          2, 2, 1, 1, 
                                          0, 0, 2, 2 ], dtype = torch.long)

    def __call__(sample, label):
        # this is the function to perform the permutations 
        taxa = np.reshape(sample, (-1, 4,20)) 



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
# criterion = torch.nn.CrossEntropyLoss(reduction='sum')
criterion = torch.nn.CrossEntropyLoss()

# define the model 
model = _Model().to(device)

# specify loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# TODO: add scheduler here!! 


n_epochs = 300

for epoch in range(1, n_epochs+1):
    # monitor training loss
    train_loss = 0.0
    
    ###################
    # train the model #
    ###################
    for genes, quartets_batch in dataloader:
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
            
    # print avg training statistics 
    train_loss = train_loss/len(dataloader)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch, 
        train_loss
        ))

# to do: how to save the data... can we do a 


