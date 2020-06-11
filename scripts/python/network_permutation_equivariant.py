import numpy as np 
# import matplotlib.pyplot as plt
import torch
import h5py
import torch.nn as nn
from torch.utils import data
import itertools

# number of available gpu
ngpu = 1 
batch_size = 16

###################################################
## Loading the data

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
mats = np.transpose(mats, (1, 2, 0))
# dims of mats is (Nsamples, NChannels, Nsequence)

nTrainSamples = 4500
nTestSamples = 500

outputTrain = torch.tensor(labels[0:nTrainSamples])
inputTrain  = torch.Tensor(mats[0:nTrainSamples, :, :])

datasetTrain = data.TensorDataset(inputTrain, outputTrain) 

outputTest = torch.tensor(labels[-nTestSamples:-1])
inputTest  = torch.Tensor(mats[-nTestSamples:-1, :, :])

datasetTest = data.TensorDataset(inputTest, outputTest) 


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

class _ResidueModuleDense(torch.nn.Module):

    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in = size_in
        self.size_out = size_out
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(size_in, size_out),
            torch.nn.BatchNorm1d(size_out),
            torch.nn.ReLU(),
            torch.nn.Linear(size_out, size_out),
            torch.nn.BatchNorm1d(size_out),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        if self.size_out == self.size_in :
            return x + self.layers(x)
        else :
            return self.layers(x)
        # TODO: add 


class _DescriptorModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            _ResidueModule(20),
            torch.nn.Conv1d(20, 20, 3),
            torch.nn.BatchNorm1d(20),
            torch.nn.ReLU(),
            _ResidueModule(20),
            torch.nn.Conv1d(20, 20, 3),
            torch.nn.BatchNorm1d(20),
            torch.nn.ReLU(),
            _ResidueModule(20),
            torch.nn.Conv1d(20, 10, 3),
            torch.nn.BatchNorm1d(10),
            torch.nn.ReLU(),
            _ResidueModule(10),
            _ResidueModule(10),
            torch.nn.AvgPool1d(2),
            _ResidueModule(10),
            _ResidueModule(10),
            torch.nn.AvgPool1d(2),
            _ResidueModule(10),
            _ResidueModule(10),
            torch.nn.AvgPool1d(2),
            _ResidueModule(10),
            _ResidueModule(10),
            torch.nn.AvgPool1d(2),
            _ResidueModule(10),
            _ResidueModule(10)
        )

    def forward(self, x):
        return self.layers(x)


class _MergeModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            _ResidueModule(10),
            _ResidueModule(10),
            torch.nn.AvgPool1d(2),
            _ResidueModule(10),
            _ResidueModule(10),
            torch.nn.AvgPool1d(2),
            _ResidueModule(10),
            _ResidueModule(10),
            torch.nn.AvgPool1d(2),
            _ResidueModule(10),
            _ResidueModule(10),
            torch.nn.AvgPool1d(2),
            _ResidueModule(10),
            _ResidueModule(10)
        )
        self.merge1 = torch.nn.Linear(60, 60)
        self.bn1 = nn.BatchNorm1d(60)
        self.merge2 = torch.nn.Linear(60, 60)
        self.bn2 = nn.BatchNorm1d(60)


    def forward(self, x):
        y = self.layers(x)
        y = y.view(y.size()[0], 60)
        z = self.merge1(y)
        z = self.bn1(z)
        z = torch.nn.ReLU()(z)
        z = self.merge2(z)
        z = self.bn2(z)
        z = torch.nn.ReLU()(z)
        return y + z

class _MergeModule2(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            _ResidueModuleDense(60, 30),
            _ResidueModuleDense(30, 15),
            _ResidueModuleDense(15, 15),
            _ResidueModuleDense(15, 10),
            _ResidueModuleDense(10, 10),
            _ResidueModuleDense(10, 5),
            _ResidueModuleDense(5, 5),
           torch.nn.Linear(5, 1),
        )

    def forward(self, x):
        return self.layers(x)



class _PermutationModule(torch.nn.Module):

    def __init__(self, descriptorModule, 
                       mergeModulelv1, 
                       mergeModulelv2):
        super().__init__()
        self._DescriptorModule = descriptorModule
        self._MergeModuleLv1 = mergeModulelv1
        self._MergeModuleLv2 = mergeModulelv2

    def forward(self, x):
        # we split the input with the proper channels
        x = x.view(x.size()[0],4,20,-1)   

        d0 =  self._DescriptorModule(x[:,0,:,:]) 
        d1 =  self._DescriptorModule(x[:,1,:,:])     
        d2 =  self._DescriptorModule(x[:,2,:,:])     
        d3 =  self._DescriptorModule(x[:,3,:,:])   

        # we compute by hand the different paths
        d01 = d0 + d1
        F1 = self._MergeModuleLv1(d01)

        d23 = d2 + d3
        F2 = self._MergeModuleLv1(d23)

        F12 = F1 + F2
        G1 = self._MergeModuleLv2(F12)


        d03 = d0 + d3
        F3 = self._MergeModuleLv1(d03)

        d12 = d1 + d2
        F4 = self._MergeModuleLv1(d12)

        F34 = F3 + F4
        G2 = self._MergeModuleLv2(F34)


        d13 = d1 + d3
        F5 = self._MergeModuleLv1(d13)

        d02 = d0 + d2
        F6 = self._MergeModuleLv1(d02)

        F56 = F5 + F6
        G3 = self._MergeModuleLv2(F56)

        G = torch.cat([G1, G2, G3], -1) # concatenation at the end

        return G


# building the data sets (no need for special collate function)
dataloaderTrain = torch.utils.data.DataLoader(datasetTrain, 
                                              batch_size=batch_size,
                                              shuffle=True)

dataloaderTest = torch.utils.data.DataLoader(datasetTest, 
                                             batch_size=100,
                                             shuffle=True)

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# defining the models
D = _DescriptorModule()
M1 = _MergeModule()
M2 = _MergeModule2()

# model using the permutations
model = _PermutationModule(D, M1, M2).to(device)

# specify loss function
criterion = torch.nn.CrossEntropyLoss(reduction='sum')

# specify optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


n_epochs = 500

print("Starting Training Loop")

min_accuracy = 0

for epoch in range(1, n_epochs+1):
    # monitor training loss
    train_loss = 0.0
    model.train()
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
            
    # print avg training statistics 
    train_loss = train_loss/len(dataloaderTrain)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch, 
        train_loss
        ), flush=True)

    # we compute the test loss every 10 epochs 
    if epoch % 10 == 0 :

        model.eval()
        correct, total = 0, 0

        for genes, quartets_batch in dataloaderTest:
            #send to the device (either cpu or gpu)
            genes, quartets_batch = genes.to(device), quartets_batch.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            quartetsNN = model(genes)
            # calculate the loss
            _, predicted = torch.max(quartetsNN, 1)
            
            total += quartets_batch.size(0)
            correct += (predicted == quartets_batch).sum().item()

        accuracyTest = correct/len(dataloaderTest)

        print('Epoch: {} \tTest accuracy: {:.6f}'.format(epoch, 
                                                         accuracyTest))

        if accuracyTest > min_accuracy:
            min_accuracy = accuracyTest
            torch.save(model.state_dict(), "saved_model.pth")


