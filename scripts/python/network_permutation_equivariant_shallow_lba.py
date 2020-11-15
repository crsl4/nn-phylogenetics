import numpy as np 
# import matplotlib.pyplot as plt
import torch
import h5py
import torch.nn as nn
from torch.utils import data
import itertools
import json
import sys
from os import path

from modules import _ResidueModule
from modules import _ResidueModuleDense

nameScript = sys.argv[0].split('/')[-1]

# we are going to give all the arguments using a Json file
nameJson = sys.argv[1]
print("=================================================")
print("Executing " + nameScript + " following " + nameJson, flush = True)
print("=================================================")

# opening Json file 
jsonFile = open(nameJson) 
dataJson = json.load(jsonFile)   

# loading the input data from the json file
ngpu = dataJson["ngpu"]                  # number of GPUS
lr = dataJson["lr"]                      # learning rate
batch_size = dataJson["batchSize"]       # batch size

dataRoot = dataJson["dataRoot"]          # data folder
modelRoot = dataJson["modelRoot"]        # folder to save the data

label_file = dataJson["labelFile"]        # file with labels
mat_file = dataJson["matFile"]            # file with sequences

n_train_samples = dataJson["nTrainSamples"]
n_test_samples = dataJson["nTestSamples"]

nEpochs  = dataJson["nEpochs"]           # number of epochs

gamma = dataJson["gamma"]               # decrease for the lr scheduler
lr_steps = dataJson["lrSteps"]          # number of steps for the scheduler

if "summaryFile" in dataJson:
    summary_file = dataJson["summaryFile"]   # file in which we 
                                             # summarize the end result
else :
    summary_file = "summary_file.txt"

print("=================================================\n")

print("Learning Rate {} ".format(lr))
print("Batch Size {} \n".format(batch_size))

print("=================================================")

print("Loading Sequence Data in " + mat_file, flush = True)
print("Loading Label Data in " + label_file, flush = True)


labelsh5 = h5py.File(dataRoot+"/"+label_file, 'r')
labels = labelsh5['labels'][:].astype(np.int64)-1 
# classes from 0 to C-1
    
matsh5 = h5py.File(dataRoot+"/"+mat_file, 'r')
mats = matsh5['matrices'][:]

nSamples = labels.shape[0]

mats = mats.reshape((1550, nSamples, -1))    
mats = np.transpose(mats, (1, 2, 0))
# dims of mats is (Nsamples, NChannels, Nsequence)

print("Total number of samples: {}".format(labels.shape[0]))
print("Number of training samples: {}".format(n_train_samples))
print("Number of testing samples: {}".format(n_test_samples))

assert n_train_samples + n_test_samples <=  nSamples

outputTrain = torch.tensor(labels[0:n_train_samples])
inputTrain  = torch.Tensor(mats[0:n_train_samples, :, :])

datasetTrain = data.TensorDataset(inputTrain, outputTrain) 

outputTest = torch.tensor(labels[-n_test_samples:-1])
inputTest  = torch.Tensor(mats[-n_test_samples:-1, :, :])

datasetTest = data.TensorDataset(inputTest, outputTest) 

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
                                             num_workers=4,
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

# specidy scheduler
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                             step_size=10, gamma=0.9)

# model.load_state_dict(torch.load("best_models/saved_permutation_model_shallow_augmented_best_batch_16.pth"))
# model.eval()

print("Starting Training Loop")

maxAccuracy = 0

for epoch in range(1, nEpochs+1):
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
    print('Epoch: {} \tLearning rate: {:.6f} \tTraining Loss: {:.6f}'.format(
        epoch, 
        optimizer.param_groups[0]['lr'],
        train_loss
        ), flush=True)

    # advance the step in the scheduler
    exp_lr_scheduler.step() 

    # we compute the test accuracy every 10 epochs 
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

        if accuracyTest > maxAccuracy:
            maxAccuracy = accuracyTest
            torch.save(model.state_dict(), modelRoot + "/" +
                "saved_{}_{}_lr_{}_batch_{}_lba_best.pth".format(nameScript.split(".")[0],
                                                                 nameJson.split(".")[0],
                                                                str(lr), 
                                                                 str(batch_size)))


torch.save(model.state_dict(), modelRoot + "/" +
           "saved_{}_{}_lr_{}_batch_{}_lba_last.pth".format(nameScript.split(".")[0],
                                                            nameJson.split(".")[0],
                                                            str(lr), 
                                                            str(batch_size)))

if not path.exists("guru99.txt"):
    with open(summary_file, 'w') as f:
        f.write("{} \t {} \t {} \t {} \t {}".format("Script name",
                                    " Json file",
                                    "lerning rate", 
                                    "batch size", 
                                    "max testing accuracy", 
                                    "train loss"))

# we write the last data to a file
with open(summary_file, 'w') as f:
    f.write("{} \t {} \t {} \t {} \t {}".format(nameScript.split(".")[0],
                                    nameJson.split(".")[0],
                                    str(lr), 
                                    str(batch_size), 
                                    str(maxAccuracy), 
                                    str(train_loss)))
## testing and saving data to centralized file 

