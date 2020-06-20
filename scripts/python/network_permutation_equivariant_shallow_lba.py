import numpy as np 
# import matplotlib.pyplot as plt
import torch
import h5py
import torch.nn as nn
from torch.utils import data
import itertools
  
import sys
nameScript = sys.argv[0].split('/')[-1]

# we are going to give all the arguments using a Json file
print("=================================================")
print("Executing " + nameScript, flush = True)
print("=================================================")


# number of available gpu
ngpu = 1 
batch_size = 16
lr = 0.001
###################################################
## Loading the data

dataRoot = "../../data"

Mats = 0
Labels = 0

for ii in range(1,2):
    labelFiles = "labels-lba-{}.h5".format(str(ii))
    matFiles = "matrices-lba-{}.h5".format(str(ii))

    print("Loading Data in " + matFiles, flush = True)


    labelsh5 = h5py.File(dataRoot+"/"+labelFiles, 'r')
    labels = labelsh5['labels'][:].astype(np.int64)-1 
    # classes from 0 to C-1
    
    matsh5 = h5py.File(dataRoot+"/"+matFiles, 'r')
    mats = matsh5['matrices'][:]

    nSamples = labels.shape[0]

    mats = mats.reshape((1550, nSamples, -1))    
    mats = np.transpose(mats, (1, 2, 0))
    # dims of mats is (Nsamples, NChannels, Nsequence)

    if type(Mats) == int :
        # we initialize the Mats
        Mats = mats
        Labels = labels
    else: 
        Mats = np.concatenate([Mats, mats], axis = 0)
        Labels = np.concatenate([Labels, labels], axis = 0)

print("Number of samples-{}".format(Labels.shape[0]))

nTrainSamples = 4500
nTestSamples = 500

outputTrain = torch.tensor(Labels[0:nTrainSamples])
inputTrain  = torch.Tensor(Mats[0:nTrainSamples, :, :])

datasetTrain = data.TensorDataset(inputTrain, outputTrain) 

outputTest = torch.tensor(Labels[-nTestSamples:-1])
inputTest  = torch.Tensor(Mats[-nTestSamples:-1, :, :])

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
        elif self.size_out ==  self.size_in/2:
            return  0.5*torch.sum(x.view(x.size()[0],-1,2), 2) + \
                    self.layers(x)
        else:    
            return self.layers(x)
        # TODO: add 


class _DescriptorModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            _ResidueModule(20),
            _ResidueModule(20),
            torch.nn.AvgPool1d(2),
            _ResidueModule(20),
            _ResidueModule(20),
            torch.nn.AvgPool1d(2),
            _ResidueModule(20),
            _ResidueModule(20),
        )

    def forward(self, x):
        return self.layers(x)


class _MergeModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            _ResidueModule(20),
            _ResidueModule(20),
            torch.nn.AvgPool1d(2),
            _ResidueModule(20),
            _ResidueModule(20),
            torch.nn.AvgPool1d(2),
            _ResidueModule(20),
            _ResidueModule(20),
        )

    def forward(self, x):
        # x  x.view(x.size()[0], 60)
        return  self.layers(x)


class _MergeModule2(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            _ResidueModule(20),
            _ResidueModule(20),
            torch.nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = torch.nn.Linear(20, 1)
    def forward(self, x):
        y = self.layers(x).squeeze(dim=2)
        return self.classifier(y)



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


# building the data sets (no need for special collate function)
dataloaderTrain = torch.utils.data.DataLoader(datasetTrain, 
                                              batch_size=batch_size,
                                              shuffle=True, 
                                              num_workers=4,
                                              pin_memory=True )

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
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


#model.load_state_dict(torch.load("saved_permutation_model_best_dataset_1.pth"))
# model.eval()

n_epochs = 3000

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
            torch.save(model.state_dict(), 
                "saved_permutation_model_shallow_lr_{}_batch_{}lba_best.pth".format(str(lr), 
                                                                                    str(batch_size)))


torch.save(model.state_dict(), "saved_permutation_model_shallow_lr_{}_batch_{}lba_last.pth".format(str(lr), 
                                                                                    str(batch_size)))