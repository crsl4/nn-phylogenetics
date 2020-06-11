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


D = _DescriptorModule()

dataloaderTest = torch.utils.data.DataLoader(datasetTest, 
                                             batch_size=100,
                                             shuffle=True)

# # to do: how to save the data... can we do a 
dataiter = iter(dataloaderTest)
genes, labels = dataiter.next()

x = genes.view(genes.size()[0],4,20,-1)   

descriptArray = []

for ii in range(0,4):
	descriptArray.append(D(x[:,ii,:,:]))





