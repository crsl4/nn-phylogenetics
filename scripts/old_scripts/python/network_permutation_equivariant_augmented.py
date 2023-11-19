## In this script we use to train both the permutation 
# equivariant network and the data augmentation during 
# training

import numpy as np 
# import matplotlib.pyplot as plt
import torch
import h5py
import torch.nn as nn
from torch.utils import data
import itertools

# number of available gpu
ngpu = 1 
batch_size = 2

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

#################################################
# We define the augmented data


class _Permutation():

    def __init__(self):

        self.permData = np.asarray(list(itertools.permutations(range(4))))
        # hard coded transformation of taxons
        self.permTaxon0 =  torch.tensor([ 0, 0, 1, 1, 
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

        self.permTaxon2 =  torch.tensor([ 2, 2, 0, 0, 
                                          1, 1, 2, 2, 
                                          1, 1, 0, 0, 
                                          0, 0, 1, 1, 
                                          2, 2, 1, 1, 
                                          0, 0, 2, 2 ], dtype = torch.long)

    def __call__(self, sample, label):
        # this is the function to perform the permutations 
        taxa = torch.reshape(sample, (4, 20, -1)) 
        taxaout = torch.stack([taxa[idx,:,:] for idx in self.permData]) 
        taxaout = torch.reshape(taxaout, (24, 80, -1))

        if label == 0:
            return (taxaout, self.permTaxon0)
        elif label == 1:
            return (taxaout, self.permTaxon1)
        elif label == 2:
            return (taxaout, self.permTaxon2)

class _Collate():

    def __init__(self):
        self.perm = _Permutation()

    def __call__(self, dataList):
        
        GenData = []
        LabelData = []

        sizeBatch = len(dataList)

        for genes, labels in dataList:
            (genesPerm, labelsPerm) = self.perm(genes, labels)  
            GenData.append(genesPerm)
            LabelData.append(labelsPerm)

        if sizeBatch == 1:
            return (GenData[0], LabelData[0])

        else:
            Gen2 = torch.stack(GenData)
            # noe the sizes are hardcoded, this needs to change
            Gen3 = Gen2.view(24*sizeBatch, 80,1550)

            Labels = torch.stack(LabelData)
            Labels2 = Labels.view(-1) 

            return (Gen3, Labels2)


#########################################################333
## we define the pieces of the neural network

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

        # Quartet 1 (12|34)
        d01 = d0 + d1
        F1 = self._MergeModuleLv1(d01)

        d23 = d2 + d3
        F2 = self._MergeModuleLv1(d23)

        F12 = F1 + F2
        G1 = self._MergeModuleLv2(F12)

        #Quartet 2 (13|24)
        d02 = d0 + d2
        F6 = self._MergeModuleLv1(d02)

        d13 = d1 + d3
        F5 = self._MergeModuleLv1(d13)

        F56 = F5 + F6
        G2 = self._MergeModuleLv2(F56)

        # Quartet 3 (14|23)
        d03 = d0 + d3
        F3 = self._MergeModuleLv1(d03)

        d12 = d1 + d2
        F4 = self._MergeModuleLv1(d12)

        F34 = F3 + F4
        G3 = self._MergeModuleLv2(F34)


        # putting all the quartest together
        G = torch.cat([G1, G2, G3], -1) # concatenation at the end

        return G


collate_fc =  _Collate()

# building the data sets (no need for special collate function)
dataloaderTrain = torch.utils.data.DataLoader(datasetTrain, 
                                              batch_size=batch_size,
                                              shuffle=True, 
                                              collate_fn = collate_fc)

dataloaderTest = torch.utils.data.DataLoader(datasetTest, 
                                             batch_size=batch_size,
                                             shuffle=True)

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# defining the models
D  = _DescriptorModule()
M1 = _MergeModule()
M2 = _MergeModule2()

# model using the permutations
model = _PermutationModule(D, M1, M2).to(device)

# specify loss function
criterion = torch.nn.CrossEntropyLoss(reduction='sum')

# specify optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# loading the other model
# model.load_state_dict(torch.load("saved_permutation_model_best_dataset_1.pth"))
# model.eval()

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

        accuracyTest = correct/total

        print('Epoch: {} \tTest accuracy: {:.6f}'.format(epoch, 
                                                         accuracyTest))

        if accuracyTest > min_accuracy:
            min_accuracy = accuracyTest
            torch.save(model.state_dict(), "saved_permutation_model_augmented_best_dataset_1.pth")


