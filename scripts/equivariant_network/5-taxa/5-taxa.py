#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import torch

torch.manual_seed(0)

import h5py
import torch.nn as nn
from torch.utils import data
import itertools
import json
import sys
import time
import os

from os import path
sys.path.insert(0, '../')
import gc

gc.collect()


# In[17]:


# first argument, get the last name in the path (file name of this file)
nameScript = "5-taxa.py"
# second argument, Json file for storing all arguments
nameJson = "5-taxa.json"

print("=================================================")
print("Training for 5-taxa")
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

data_root = dataJson["dataRoot"]         # data folder
model_root = dataJson["modelRoot"]       # folder to save the data

label_files = dataJson["labelFile"]      # file with labels
sequence_files = dataJson["matFile"]     # file with sequences

n_epochs = dataJson["nEpochs"]           # number of epochs

if "summaryFile" in dataJson:
    summary_file = dataJson["summaryFile"]
else :
    summary_file = "summary_file.txt"

print("=================================================")
print("Learning Rate {} ".format(lr))
print("Batch Size {} ".format(batch_size))
print("=================================================")


# In[18]:


print("Loading Sequence Data in " + sequence_files, flush = True)
print("Loading Label Data in " + label_files, flush = True)

# we read the labels as list of strings
with open(data_root+label_files, 'r') as f:
    label_char = f.readlines()

# we read the sequence as a list of strings
with open(data_root+sequence_files, 'r') as f:
    seq_string = f.readlines()

n_samples = len(label_char)

# extracting the sequence lenghth
seq_length = len(seq_string[0])-1


# In[19]:


# function to convert string to numbers
def convert_string_to_numbers(str, dict):
    ''' str: is the string to convert,
        dict: dictionary with the relative ordering of each char'''

    # create a map iterator using a lambda function
    # lambda x -> return dict[x]
    # This return the value for each key in dict based on str
    numbers = map(lambda x: dict[x], str)
    # return an array of int64 numbers
    return np.fromiter(numbers, dtype=np.int64)


# In[20]:


# We need to extract the dictionary with the relative positions 
# for each aminoacid

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


# In[21]:


# looping over the labels and create array. Here each element of the
# label_char has the form "1\n", so we only take the first one
labels = np.fromiter(map(lambda x: int(x[0])-1,
                         label_char), dtype= np.int64)

# setting up the empty data matrix
mats = np.zeros((len(seq_string), seq_length), dtype = np.int64)

# this is pretty slow (optimize in numba)
for ii, seq in enumerate(seq_string):
    # note each line has a \n character at the end so we remove it
    mats[ii,:] = convert_string_to_numbers(seq[:-1],\
                 dict_amino).reshape((1,seq_length))

mats = mats.reshape((n_samples, -1, seq_length))    
# dims of mats is (N_samples, n_sequences, seq_length)

# we need to assign the truncation lenght
trunc_length = 1550

print("Total number of samples: {}".format(labels.shape[0]))


# In[22]:


# We specify the networks

# This is the same as 4-taxa
class ResNetModule(torch.nn.Module):
    '''Dense Residual network acting on each site, thus
    implemtented via a Conv1 with window size equals to one
    '''

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


# In[23]:


# The descriptors will be different
class DescriptorModule(torch.nn.Module):
    ''' Class implementing the Descriptor module, in this case we implement
    in a unified manner, D_1, D_2, ... , D_15.
    '''

    def __init__(self, length_dict, embedding_dim, trunc_length = 1550):
        super().__init__()

        self.embedding_layer = nn.Embedding(length_dict, embedding_dim)
        self._res_module_1 = ResNetModule(embedding_dim)
        self._res_module_2 = ResNetModule(embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, x):
        # (none, 5, 1550)

        # we use an embedding layer first
        x = self.embedding_layer(x).permute([0, 1, 3, 2])
        # (none, 5, 1550, chn_dim) without permute
        # (none, 5, chn_dim, 1550) with permutation

        # Embedding all 5 species
        # we apply \phi
        d0 =  self._res_module_1(x[:,0,:,:])
        d1 =  self._res_module_1(x[:,1,:,:])
        d2 =  self._res_module_1(x[:,2,:,:])
        d3 =  self._res_module_1(x[:,3,:,:])
        d4 =  self._res_module_1(x[:,4,:,:])
        
        # Embedding all 10 pairs
        # d01 = d0 + d1
        d01 = self._res_module_2(d0 + d1)
        # d02 = d0 + d2
        d02 = self._res_module_2(d0 + d2)
        # d03 = d0 + d3
        d03 = self._res_module_2(d0 + d3)
        # d04 = d0 + d4
        d04 = self._res_module_2(d0 + d4)
        # d12 = d1 + d2
        d12 = self._res_module_2(d1 + d2)
        # d13 = d1 + d3
        d13 = self._res_module_2(d1 + d3)
        # d14 = d1 + d4
        d14 = self._res_module_2(d1 + d4)
        # d23 = d2 + d3
        d23 = self._res_module_2(d2 + d3)
        # d24 = d2 + d4
        d24 = self._res_module_2(d2 + d4)
        # d34 = d3 + d4
        d34 = self._res_module_2(d3 + d4)
        
        # Combine into 15 descriptors
        # 12|34|5
        D_1 = d01 + d23
        # 12|35|4
        D_2 = d01 + d24
        # 12|45|3
        D_3 = d01 + d34
        # 13|24|5
        D_4 = d02 + d13
        # 13|25|4
        D_5 = d02 + d14
        # 13|45|2
        D_6 = d02 + d34
        # 14|23|5
        D_7 = d03 + d12
        # 14|25|3
        D_8 = d03 + d14
        # 14|35|2
        D_9 = d03 + d24
        # 15|23|4
        D_10 = d04 + d12
        # 15|24|3
        D_11 = d04 + d13
        # 15|34|2
        D_12 = d04 + d23
        # 23|45|1
        D_13 = d12 + d34
        # 24|35|1
        D_14 = d13 + d24
        # 25|34|1
        D_15 = d14 + d23


        x = torch.cat([torch.unsqueeze(D_1,1),
                       torch.unsqueeze(D_2,1),
                       torch.unsqueeze(D_3,1),
                       torch.unsqueeze(D_4,1),
                       torch.unsqueeze(D_5,1),
                       torch.unsqueeze(D_6,1),
                       torch.unsqueeze(D_7,1),
                       torch.unsqueeze(D_8,1),
                       torch.unsqueeze(D_9,1),
                       torch.unsqueeze(D_10,1),
                       torch.unsqueeze(D_11,1),
                       torch.unsqueeze(D_12,1),
                       torch.unsqueeze(D_13,1),
                       torch.unsqueeze(D_14,1),
                       torch.unsqueeze(D_15,1)],dim = 1)
        # (none, 15, embedding_dim, 1550)

        return x


# In[24]:


# small change: change 3*batch to 15, as we have 15 descriptors now
class _Model(torch.nn.Module):
    """A neural network model to predict phylogenetic trees."""

    def __init__(self, embedding_dim = 160, hidden_dim = 40, 
                      num_layers = 6, output_size = 40, 
                      dropout = 0.0):
        """Create a neural network model."""
        super().__init__()


        self.descriptor_model = DescriptorModule(20, embedding_dim)
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.embedding_dim = embedding_dim

        # we define the required elements for \Psi
        self.classifier = torch.nn.Linear(self.output_size, 1)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, 
                           num_layers, dropout=dropout,
                           batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, self.output_size)

        # flatenning the parameters (required for the lstm)
        self.rnn.flatten_parameters()


    def forward(self, x):
        """Function that infers the phylogenetic trees for the input sequences.
        Input: x the raw sequences
        Output: the scores for each topology
        """
        # extracting the device and the batch size
        device = x.device
        batch_size = x.size()[0]
        
        # this is the structure preserving embedding
        g =  self.descriptor_model(x)
        
        # we reshape the output tensor
        X =  g.view(15*batch_size, self.embedding_dim, -1)

        # (none*15, 1550, hidden_dim)
        r_output, hidden = self.rnn(X.permute([0, 2, 1]))

        # extracting only the last in the sequence
        # (none*15, hidden_dim)
        r_output_last = r_output[:, -1, :] 

        # not sure if this helps
        out = r_output_last.contiguous().view(-1, self.hidden_dim)
        
        # (none*15, out_put_dimensions)
        output = self.fc(out)

        X_combined = self.classifier(output) 
        # (15*none, 1)

        X_combined = X_combined.view(batch_size, 15)

        return X_combined


# In[25]:


# checking the device
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


# In[26]:


# specify loss function
criterion = torch.nn.CrossEntropyLoss(reduction='mean')

# define the model
model = _Model(dropout = 0.2).to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print("number of parameters is %d"%count_parameters(model))


# In[27]:


# specify loss function
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# specify scheduler
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.9)


# In[28]:


print("Starting Training Loop")

max_accuracy, max_train_accuracy = 0, 0
get_epoch, get_acc = [], []
trainEpoch, trainLoss = [], []
test_epoch, test_loss_array = [], []
inputTrain = torch.from_numpy(mats[0:9500, :, :trunc_length])
outputTrain = torch.from_numpy(labels[0:9500])

inputTest = torch.from_numpy(mats[9500:10000, :, :trunc_length])
outputTest = torch.from_numpy(labels[9500:10000])

datasetTrain = data.TensorDataset(inputTrain, outputTrain)
datasetTest = data.TensorDataset(inputTest, outputTest)

for epoch in range(1, n_epochs + 1):
    # monitor training loss
    train_loss = 0.0

    # monitor the time each epoch takes
    start = time.time()

    model.train()
    train_total, train_correct = 0, 0
    
    dataloaderTrain = torch.utils.data.DataLoader(datasetTrain,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  pin_memory=True)
    for genes, quintets_batch in dataloaderTrain:
        # send to the device (either cpu or gpu)
        genes, quintets_batch = genes.to(device), quintets_batch.to(device)

        # clear the gradients of all optimized variables
        optimizer.zero_grad()

        # forward pass: compute predicted outputs by passing inputs to the model
        quintetsNN = model(genes)
        _, predicted = torch.max(quintetsNN, 1)

        #calculate training accuracy
        train_total += quintets_batch.size(0)
        train_correct += (predicted == quintets_batch).sum().item()

        # calculate the loss
        loss = criterion(quintetsNN, quintets_batch)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item()
        
    trainEpoch.append(epoch)
    trainLoss.append(train_loss)
    
    end = time.time()
    
    # print avg training statistics
    train_accuracy_test = train_correct / train_total
    print("correct train predictions = %d \ttotal predictions = %d \tratio = %.3f"%(\
         train_correct, train_total, train_accuracy_test))

    train_loss = train_loss/len(dataloaderTrain)

    print('Epoch: {} \tLearning rate: {:.6f} \tTraining Loss: {:.6f} \tTime Elapsed: {:.6f}[s]'.format(
        epoch,
        optimizer.param_groups[0]['lr'],
        train_loss,
        end - start
        ), flush=True)
    if train_accuracy_test > max_train_accuracy:
      max_train_accuracy = train_accuracy_test

    # advance the step in the scheduler
    exp_lr_scheduler.step()
    
    # we compute the test accuracy every 10 epochs
    if epoch % 5 == 0 :
        #get the testing data by  collecting 50 samples per lba dataset
        test_loss, testlabel, testsequence = 0, [], []
        dataloaderTest = torch.utils.data.DataLoader(datasetTest,
                                                     batch_size=batch_size,
                                                     shuffle=True)
        model.eval()
        correct, total = 0, 0

        # we measure the time it takes to evaluat the test examples
        start = time.time()
        
        for genes, quintets_batch in dataloaderTest:

            #send to the device (either cpu or gpu)
            genes, quintets_batch = genes.to(device), quintets_batch.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model

            quintetsNN = model(genes)
            # calculate the loss
            _, predicted = torch.max(quintetsNN, 1)

            # calculate the loss
            loss = criterion(quintetsNN, quintets_batch)

            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            test_loss += loss.item()
            total += quintets_batch.size(0)
            correct += (predicted == quintets_batch).sum().item()

        test_epoch.append(epoch)
        test_loss_array.append(test_loss)

        end = time.time()
        accuracy_test = correct/total
        get_epoch.append(epoch)
        get_acc.append(accuracy_test)

        print('Epoch: {} \tTest accuracy: {:.6f}  \tTime Elapsed: {:.6f}[s]'.format(epoch,
                                                         accuracy_test,
                                                         end - start))

        # if test accuracy is good then
        if accuracy_test > max_accuracy:
            max_accuracy = accuracy_test
            torch.save(model.state_dict(), model_root  +
                "saved_{}_{}_lr_{}_batch_{}_lba_best.pth".format(nameScript.split(".")[0],
                                                                 nameJson.split(".")[0],
                                                                 str(lr),
                                                                 str(batch_size)))

        model.eval()
        correct, total = 0, 0

        for genes, quintets_batch in dataloaderTrain:
            #send to the device (either cpu or gpu)
            genes, quintets_batch = genes.to(device), quintets_batch.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            quintetsNN = model(genes)
            # calculate the loss
            _, predicted = torch.max(quintetsNN, 1)

            total += quintets_batch.size(0)
            correct += (predicted == quintets_batch).sum().item()

        accuracyTrain = correct/total

        print('Epoch: {} \tTrain accuracy: {:.6f}'.format(epoch,
                                                         accuracyTrain))


# In[ ]:


# we save the last model
torch.save(model.state_dict(), model_root  +
           "saved_{}_{}_lr_{}_batch_{}_lba_last.pth".format(nameScript.split(".")[0],
                                                            nameJson.split(".")[0],
                                                            str(lr),
                                                            str(batch_size)))

# and we add the statistics
if not path.exists(summary_file):
    with open(summary_file, 'w') as f:
        f.write("{} \t {} \t {} \t {} \t {} \t {} \t {} \t {}\n".format("Script name",
                                    " Json file",
                                    "label file",
                                    "lerning rate",
                                    "batch size",
                                    "max testing accuracy",
                                    "train loss",
                                    "N epoch",
                                    "chnl_dim",
                                    "embd_dim"))

info_output = np.c_[get_epoch,get_acc]
np.savetxt("EpochAccuracy100.csv", info_output, delimiter=",")

test_output = np.c_[test_epoch,test_loss_array]
np.savetxt("test_loss_array100.csv", test_output, delimiter=",")

train_output = np.c_[trainEpoch,trainLoss]
np.savetxt("trainLoss1300.csv", train_output, delimiter=",")

# we write the last data to a file
with open(summary_file, 'a') as f:
    f.write("{} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \n".format(nameScript.split(".")[0],
                                    nameJson.split(".")[0],
                                    label_files,
                                    str(lr),
                                    str(batch_size),
                                    str(max_accuracy),
                                    str(train_loss),
                                    str(n_epochs)))


# In[ ]:




