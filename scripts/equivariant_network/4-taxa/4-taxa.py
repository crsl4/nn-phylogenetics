import numpy as np
import torch

# we fix the random seed for reproducibility
torch.manual_seed(0)
import torch.nn as nn
from torch.utils import data
import itertools
import json
import sys
import time

from os import path
sys.path.insert(0, '../')

import gc

gc.collect()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# function to convert string to numbers
def convert_string_to_numbers(str, dict):
    ''' str: is the string to convert,
        dict: dictionary with the relative ordering of each char'''

    # create a map iterator using a lambda function
    numbers = map(lambda x: dict[x], str)

    return np.fromiter(numbers, dtype=np.int64)

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


class DescriptorModule(torch.nn.Module):
    ''' Class implementing the Descriptor module, in this case we implement
    in a unified manner, D_I, D_{II}, and D_{III}.
    '''

    def __init__(self, length_dict, embedding_dim, trunc_length = 1550):
        super().__init__()

        self.embedding_layer = nn.Embedding(length_dict, embedding_dim)
        self._res_module_1 = ResNetModule(embedding_dim)
        self._res_module_2 = ResNetModule(embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, x):
        # (none, 4, 1550)

        # we use an embedding layer first
        x = self.embedding_layer(x).permute([0, 1, 3, 2])
        # (none, 4, 1550, chn_dim) without permute
        # (none, 4, chn_dim, 1550) with permutation

        # we apply \phi
        d0 =  self._res_module_1(x[:,0,:,:])
        d1 =  self._res_module_1(x[:,1,:,:])
        d2 =  self._res_module_1(x[:,2,:,:])
        d3 =  self._res_module_1(x[:,3,:,:])

        # Quartet 1 (12|34)
        # d01 = d0 + d1
        d01 = self._res_module_2(d0 + d1)

        # d23 = d2 + d3
        d23 = self._res_module_2(d2 + d3)

        # the first descriptor
        D_1 = d01 + d23

        #Quartet 2 (13|24)
        # d02 = d0 + d2
        d02 = self._res_module_2(d0 + d2)

        # d13 = d1 + d3
        d13 = self._res_module_2(d1 + d3)

        # the second descriptor
        D_2 = d02 + d13

        # Quartet 3 (14|23)
        # d03 = d0 + d3
        d03 = self._res_module_2(d0 + d3)

        # d12 = d1 + d2
        d12 = self._res_module_2(d1 + d2)

        # the third descriptor
        D_3 = d03 + d12

        x = torch.cat([torch.unsqueeze(D_1,1),
                       torch.unsqueeze(D_2,1),
                       torch.unsqueeze(D_3,1)], dim = 1)
        # (none, 3, embedding_dim, 1550)

        return x


class _Model(torch.nn.Module):
    """A neural network model to predict phylogenetic trees."""

    def __init__(self, embedding_dim = 80, hidden_dim = 20,
                      num_layers = 3, output_size = 20,
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
        X =  g.view(3*batch_size, self.embedding_dim, -1)

        # (none*3, 1550, hidden_dim)
        r_output, hidden = self.rnn(X.permute([0, 2, 1]))

        # extracting only the last in the sequence
        # (none*3, hidden_dim)
        r_output_last = r_output[:, -1, :]

        # not sure if this helps
        out = r_output_last.contiguous().view(-1, self.hidden_dim)

        # (none*3, out_put_dimensions)
        output = self.fc(out)

        X_combined = self.classifier(output)
        # (3*none, 1)

        X_combined = X_combined.view(batch_size, 3)

        return X_combined

# get name of the script
# nameScript = sys.argv[0].split('/')[-1]
nameScript = "4-taxa.py"
# get json file name of the script
nameJson = "4-taxa.json"
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

# checking if the summary file exist in the json file
# we will save several statistics in this file
if "summaryFile" in dataJson:
    summary_file = dataJson["summaryFile"] 
else :
    summary_file = "summary_file.txt"

print("=================================================\n")
print("Learning Rate {} ".format(lr))
print("Batch Size {} \n".format(batch_size))
print("=================================================")

label_char = []
seq_string = []
# we read the labels as list of strings
for i in range(len(label_files)):
    with open(data_root+label_files[i], 'r') as f:
        labels = f.readlines()
    
    with open(data_root+sequence_files[i], 'r') as f:
        seq = f.readlines()

    
    label_char += labels[0:10000]
    seq_string += seq[0:40000]

n_samples = len(label_char)
seq_length = len(seq_string[0])-1

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

# looping over the labels and create array. Here each element of the 
# label_char has the form "1\n", so we only take the first one
labels = np.fromiter(map(lambda x: int(x[0])-1,
                         label_char), dtype= np.int64)

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

# specify loss function
criterion = torch.nn.CrossEntropyLoss(reduction='mean')

# define the model
model = _Model(dropout = 0.2).to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("number of parameters is %d"%count_parameters(model))

# specify loss function
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# specify scheduler
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.9)


print("Starting Training Loop")

# Set the training and the testing set
train_indexes = np.array([np.arange(0,9000), np.arange(10000,19000), np.arange(20000,29000)])
train_extractions_seq = mats[train_indexes]
train_extractions_lab = labels[train_indexes]
inputTrain = torch.from_numpy(np.concatenate((train_extractions_seq[0], train_extractions_seq[1], train_extractions_seq[2]), axis=0))
outputTrain = torch.from_numpy(np.concatenate((train_extractions_lab[0], train_extractions_lab[1], train_extractions_lab[2]), axis=0))

test_indexes = np.array([np.arange(9000,10000), np.arange(19000,20000), np.arange(29000,30000)])
test_extractions_seq = mats[test_indexes]
test_extractions_lab = labels[test_indexes]
inputTest = torch.from_numpy(np.concatenate((test_extractions_seq[0], test_extractions_seq[1], test_extractions_seq[2]), axis=0))
outputTest = torch.from_numpy(np.concatenate((test_extractions_lab[0], test_extractions_lab[1], test_extractions_lab[2]), axis=0))

def test_acc (dataloaderTest, epoch, max_accuracy):
    test_loss, correct, total = 0, 0, 0

    # we measure the time it takes to evaluat the test examples
    start = time.time()

    for genes, quartets_batch in dataloaderTest:

        #send to the device (either cpu or gpu)
        genes, quartets_batch = genes.to(device), quartets_batch.to(device)
        # forward pass: compute predicted outputs by passing inputs to the model

        quartetsNN = model(genes)
        # calculate the loss
        _, predicted = torch.max(quartetsNN, 1)

        # calculate the loss
        loss = criterion(quartetsNN, quartets_batch)

        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        test_loss += loss.item()
        total += quartets_batch.size(0)
        correct += (predicted == quartets_batch).sum().item()

    end = time.time()
    accuracy_test = correct/total
    print('Epoch: {} \tTest accuracy: {:.6f}  \tTime Elapsed: {:.6f}[s]'.format(epoch,
                                                         accuracy_test,
                                                         end - start))
    if accuracy_test > max_accuracy:
        max_accuracy = accuracy_test

    return test_loss, accuracy_test, max_accuracy

max_train_accuracy = 0
get_epoch = []
get_test1_acc, get_test2_acc, get_test3_acc = [],[],[]
trainEpoch, trainLoss = [], []
test_epoch = []
get_test1_loss, get_test2_loss, get_test3_loss = [],[],[]
max_accuracy_1, max_accuracy_2, max_accuracy_3 = 0,0,0

for epoch in range(1, n_epochs + 1):
    # monitor training loss
    train_loss = 0.0

    # monitor the time each epoch takes
    start = time.time()

    model.train()
    train_total, train_correct = 0, 0

    # load the dataloader
    datasetTrain = data.TensorDataset(inputTrain, outputTrain)
    dataloaderTrain = torch.utils.data.DataLoader(datasetTrain,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  pin_memory=True)

    # Train the model
    for genes, quartets_batch in dataloaderTrain:

        # send to the device (either cpu or gpu)
        genes, quartets_batch = genes.to(device), quartets_batch.to(device)

        # clear the gradients of all optimized variables
        optimizer.zero_grad()

        # forward pass: compute predicted outputs by passing inputs to the model
        quartetsNN = model(genes)
        _, predicted = torch.max(quartetsNN, 1)

        #calculate training accuracy
        train_total += quartets_batch.size(0)
        train_correct += (predicted == quartets_batch).sum().item()

        # calculate the loss
        loss = criterion(quartetsNN, quartets_batch)
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
    print("correct predictions = %d \ttotal predictions = %d \tratio = %.3f"%(\
         train_correct, train_total, train_accuracy_test))

    train_loss = train_loss/len(dataloaderTrain)

    print('Epoch: {} \tLearning rate: {:.6f} \tTraining Loss: {:.6f} \tTime Elapsed: {:.6f}[s]'.format(
        epoch,
        optimizer.param_groups[0]['lr'],
        train_loss,
        end - start
        ), flush=True)
    if train_accuracy_test >max_train_accuracy:
      max_train_accuracy = train_accuracy_test

    # advance the step in the scheduler
    exp_lr_scheduler.step()

    if epoch % 10 == 0 :
        test_epoch.append(epoch)
        get_epoch.append(epoch)
        # first case:
        datasetTest = data.TensorDataset(inputTest[0:1000], outputTest[0:1000])
        dataloaderTest = torch.utils.data.DataLoader(datasetTest,
                                                     batch_size=batch_size,
                                                     shuffle=True)
        model.eval()
        test_loss_1, test_acc_1, max_accuracy_1 = test_acc(dataloaderTest, epoch, max_accuracy_1)
        get_test1_loss.append(test_loss_1)
        get_test1_acc.append(test_acc_1)

        # second case:
        datasetTest = data.TensorDataset(inputTest[1000:2000], outputTest[1000:2000])
        dataloaderTest = torch.utils.data.DataLoader(datasetTest,
                                                     batch_size=batch_size,
                                                     shuffle=True)
        model.eval()
        test_loss_2, test_acc_2, max_accuracy_2 = test_acc(dataloaderTest, epoch, max_accuracy_2)
        get_test2_loss.append(test_loss_2)
        get_test2_acc.append(test_acc_2)

        # third case:
        datasetTest = data.TensorDataset(inputTest[2000:3000], outputTest[2000:3000])
        dataloaderTest = torch.utils.data.DataLoader(datasetTest,
                                                     batch_size=batch_size,
                                                     shuffle=True)
        model.eval()
        test_loss_3, test_acc_3, max_accuracy_3 = test_acc(dataloaderTest, epoch, max_accuracy_3)
        get_test3_loss.append(test_loss_3)
        get_test3_acc.append(test_acc_3)

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

info_output = np.c_[get_epoch,get_test1_acc, get_test2_acc, get_test3_acc]
np.savetxt("EpochAccuracy200.csv", info_output, delimiter=",") 
test_output = np.c_[test_epoch, get_test1_loss, get_test2_loss, get_test3_loss]
np.savetxt("test_loss_array200.csv", test_output, delimiter=",") 
train_output = np.c_[trainEpoch,trainLoss]
np.savetxt("trainLoss200.csv", train_output, delimiter=",")

with open(summary_file, 'a') as f:
    f.write("{} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \n".format(nameScript.split(".")[0],
                                    nameJson.split(".")[0],
                                    label_files,
                                    str(lr), 
                                    str(batch_size), 
                                    str(max_accuracy_1), 
                                    str(max_accuracy_2), 
                                    str(max_accuracy_3), 
                                    str(n_epochs)))
