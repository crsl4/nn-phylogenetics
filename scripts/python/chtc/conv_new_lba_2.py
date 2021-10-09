#cript containing the modular version of the code
# so far we have only implemented the non_linear_embedding layer
# for simplicity we can just use dense layers for the merge operations


import numpy as np 
# import matplotlib.pyplot as plt
import torch
import h5py
import torch.nn as nn
import torch.nn.functional as F

from torch.utils import data
import itertools
import json
import sys
from os import path
import time


sys.path.insert(0, '../')

from modules import _ResidueModule
from modules import _ResidueModuleDense

from modules import _NonLinearScoreConv
from modules import _NonLinearMergeConv
from modules import _NonLinearEmbeddingConv

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

chnl_dim = dataJson["channel_dimension"]
embd_dim = dataJson["embedding_dimension"]


if "random_seed" in dataJson:
    torch.manual_seed(dataJson["random_seed"])
else:
    torch.manual_seed(0)


if "summaryFile" in dataJson:
    summary_file = dataJson["summaryFile"]   # file in which we 
                                             # summarize the end result
else :
    summary_file = "summary_file.txt"

if "kernel_size" in dataJson:
    kernel_size = dataJson["kernel_size"]   # file in which we 
                                             # summarize the end result
else :
    kernel_size = 5

if "kernel_size_emb" in dataJson:
    kernel_size_emb = dataJson["kernel_size_emb"]   # file in which we 
                                             # summarize the end result
else :
    kernel_size_emb = 3

print("=================================================\n")

print("Learning Rate {} ".format(lr))
print("Batch Size {} \n".format(batch_size))

print("=================================================")

print("Loading Sequence Data in " + mat_file, flush = True)
print("Loading Label Data in " + label_file, flush = True)


# function to count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# we read the labels as list of strings
with open(dataRoot+'/'+label_file, 'r') as f: 
    label_char = f.readlines() 

# we read the sequence as a list of strings
with open(dataRoot+'/'+mat_file, 'r') as f: 
    seq_string = f.readlines() 

# extracting the number of samples
n_samples = len(label_char)

# extracting the sequence lenght
seq_lenght = len(seq_string[0])-1
# note: each element has a '\n' character at the end

# function to convert string to numbers 
def convert_string_to_numbers(str, dict):
    ''' str: is the string to convert, 
        dict: dictionary with the relative ordering of each char'''

    # create a map iterator using a lambda function
    numbers = map(lambda x: dict[x], str)

    return np.fromiter(numbers, dtype=np.int64)

# We need to extract the dictionary with the relative positions
# fo each aminoacid

# first we need to extract all the different chars 
strL = "" 
for c in seq_string[0][:-1]: 
    if not c in strL: 
        strL += c

# we sort them 
strL = sorted(strL)

# we give them a relative order
dict_amino = {}
for ii, c in enumerate(strL):
    dict_amino[c] = ii 

# looping over the labels and create array
# here each element of the label_char has 
# the form "1\n", so we only take the first one

print("allocating the space for the data", flush=True)
labels = np.fromiter(map(lambda x: int(x[0])-1, 
                         label_char), dtype=np.int64)

mats = np.zeros((len(seq_string), seq_lenght), dtype = np.int64)


# this is pretty slow (optimize in numba)
for ii, seq in enumerate(seq_string):
    # note each line has a \n character at the end so we remove it
    mats[ii,:] = convert_string_to_numbers(seq[:-1], dict_amino).reshape((1,seq_lenght))


mats = mats.reshape((n_samples, -1, seq_lenght))    
# dims of mats is (N_samples, n_sequences, seq_length)

print("Total number of samples: {}".format(labels.shape[0]))
print("Number of training samples: {}".format(n_train_samples))
print("Number of testing samples: {}".format(n_test_samples))

assert n_train_samples + n_test_samples <=  n_samples


# We define our custom Sequence DataSet, which provides the 
# the one-hot encoding on the fly.
class SequenceDataSet(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, sequences, labels, n_char = 20, transform=None):
        """
        Args:  sequences: pytorch tensor with the sequences
               labels:    pytorch tensor with the labels
        """
        self.sequences = sequences
        self.labels = labels
        # number of characters
        self.n_char = n_char

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sequence_out = self.sequences[idx,:,:]
        label = self.labels[idx]
        seq_stack = []

        for ii in range(4):
            temp = torch.nn.functional.one_hot(sequence_out[ii,:], 
                                                      self.n_char)
            # we need to transpose it. Perhaps is better to 
            # transpose everything at the end.
            temp = torch.transpose(temp, 0, 1)
            seq_stack.append(temp)

        seq_stack = torch.stack(seq_stack, dim=0)

        sample = (seq_stack, label)

        return sample


# we perform the training/validation splitting
outputTrain = torch.from_numpy(labels[0:n_train_samples])
inputTrain  = torch.from_numpy(mats[0:n_train_samples, :, :])

datasetTrain = SequenceDataSet(inputTrain, outputTrain) 

outputTest = torch.from_numpy(labels[-n_test_samples:-1])
inputTest  = torch.from_numpy(mats[-n_test_samples:-1, :, :])

datasetTest = SequenceDataSet(inputTest, outputTest) 

print("freeing space")
## freeing space 
del seq_string
del labels
del mats


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
        # d01 = d0 + d1
        d01 = self._MergeModuleLv1(d0, d1)

        # d23 = d2 + d3
        d23 = self._MergeModuleLv1(d2, d3)

        G1 = self._MergeModuleLv2(d01, d23)

        #Quartet 2 (13|24)
        # d02 = d0 + d2
        d02 = self._MergeModuleLv1(d0, d2)

        # d13 = d1 + d3
        d13 = self._MergeModuleLv1(d1, d3)

        # F56 = F5 + F6
        G2 = self._MergeModuleLv2(d02, d13)

        # Quartet 3 (14|23)
        # d03 = d0 + d3
        d03 = self._MergeModuleLv1(d0, d3)

        # d12 = d1 + d2
        d12 = self._MergeModuleLv1(d1, d2)

        # F34 = F3 + F4
        G3 = self._MergeModuleLv2(d03, d12)

        # putting all the quartest together
        G = torch.cat([G1, G2, G3], -1) # concatenation at the end

        return G


# building the data sets (no need for special collate function)
dataloaderTrain = torch.utils.data.DataLoader(datasetTrain, 
                                              batch_size=batch_size,
                                              shuffle=True, 
                                              num_workers=2)

dataloaderTest = torch.utils.data.DataLoader(datasetTest, 
                                             batch_size=batch_size,
                                             num_workers=2)

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# defining the models
# this is harwired for now
jit = False
if jit: 
    D  = torch.jit.script(_NonLinearEmbeddingConv(1550, 20, chnl_dim, embd_dim, 
                             kernel_size=kernel_size_emb, dropout_bool=True)).to(device)
    # non-linear merge is just a bunch of dense ResNets 
    M1 = torch.jit.script(_NonLinearMergeConv(chnl_dim, kernel_size, 6, 
                             dropout_bool=True, act_fn=F.relu)).to(device)

    M2 =torch.jit.script(_NonLinearScoreConv(chnl_dim, kernel_size, 6, 
                             dropout_bool=True, act_fn = F.relu)).to(device)

    # model using the permutations
    model = torch.jit.script(_PermutationModule(D, M1, M2)).to(device)

else:
    D  = _NonLinearEmbeddingConv(1550, 20, chnl_dim, embd_dim, 
                                 kernel_size=kernel_size_emb, dropout_bool=True)
    # non-linear merge is just a bunch of dense ResNets 
    M1 = _NonLinearMergeConv(chnl_dim, kernel_size, 6, 
                             dropout_bool=True, act_fn=F.relu)
    M2 = _NonLinearScoreConv(chnl_dim, kernel_size, 6, 
                             dropout_bool=True, act_fn = F.relu)
    # model using the permutations
    model = _PermutationModule(D, M1, M2).to(device)


print("number of parameters for the model is %d"%count_parameters(model))

# specify loss function
criterion = torch.nn.CrossEntropyLoss(reduction='mean')

# specify optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# specidy scheduler
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                             step_size=lr_steps, gamma=gamma)

# model.load_state_dict(torch.load("best_models/saved_permutation_model_shallow_augmented_best_batch_16.pth"))
# model.eval()

print("Starting Training Loop")

maxAccuracy = 0

for epoch in range(1, nEpochs+1):
    # monitor training loss
    train_loss = 0.0
    # monitor time 
    start = time.time()
    model.train()
    ###################
    # train the model #
    ###################

    for genes, quartets_batch in dataloaderTrain:
        #send to the device (either cpu or gpu)
        genes, quartets_batch = genes.to(device).float(), quartets_batch.to(device)
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

    end = time.time()

    train_loss = train_loss/len(dataloaderTrain)
    print('Epoch: {} \tLearning rate: {:.6f} \tTraining Loss: {:.6f} \tTime Elapsed: {:.6f}[s]'.format(
        epoch, 
        optimizer.param_groups[0]['lr'],
        train_loss,
        end - start
        ), flush=True)

    # advance the step in the scheduler
    exp_lr_scheduler.step() 

    # we compute the test accuracy every 10 epochs 
    if epoch % 10 == 0 :

        model.eval()
        correct, total = 0, 0

        for genes, quartets_batch in dataloaderTest:
            #send to the device (either cpu or gpu)
            genes, quartets_batch = genes.to(device).float(), quartets_batch.to(device)
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
            torch.save(model.state_dict(), modelRoot+
                "saved_{}_{}_lr_{}_batch_{}_lba_best.pth".format(nameScript.split(".")[0],
                                                                 nameJson.split(".")[0],
                                                                str(lr), 
                                                                 str(batch_size)))


torch.save(model.state_dict(), modelRoot  +
           "saved_{}_{}_lr_{}_batch_{}_lba_last.pth".format(nameScript.split(".")[0],
                                                            nameJson.split(".")[0],
                                                            str(lr), 
                                                            str(batch_size)))

if not path.exists(summary_file):
    with open(summary_file, 'w') as f:
        f.write("{} \t {} \t {} \t {} \t {} \t {} \t {} \t {}\n".format("Script name",
                                    " Json file",
                                    "lerning rate", 
                                    "batch size", 
                                    "max testing accuracy", 
                                    "train loss", 
                                    "N epoch", 
                                    "chnl_dim",
                                    "embd_dim"))

# we write the last data to a file
with open(summary_file, 'a') as f:
    f.write("{} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \n".format(nameScript.split(".")[0],
                                    nameJson.split(".")[0],
                                    str(lr), 
                                    str(batch_size), 
                                    str(maxAccuracy), 
                                    str(train_loss),
                                    str(nEpochs), 
                                    str(chnl_dim),
                                    str(embd_dim)))
## testing and saving data to centralized file 
