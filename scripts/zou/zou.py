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
import gc

gc.collect()

nameScript = "zou.py"
nameJson = "zou.json"
test_index = int(sys.argv[1])
print("=================================================")
print("Executing " + nameScript + " following " + nameJson + " on case " + str(test_index), flush = True)
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

#if "summaryFile" in dataJson:
#    summary_file = dataJson["summaryFile"]   # file in which we 
#                                             # summarize the end result
#else :
summary_file = "summary_file_"+ str(test_index) + ".txt"

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

    
    label_char += labels[0:9000]
    seq_string += seq[0:36000]

    if i == test_index:
        test_label_char = labels[9000:10000]
        test_seq_string = seq[36000:40000]

n_samples = len(label_char)
seq_length = len(seq_string[0])-1

# function to convert string to numbers
def convert_string_to_numbers(str, dict):
    ''' str: is the string to convert,
        dict: dictionary with the relative ordering of each char'''

    # create a map iterator using a lambda function
    numbers = map(lambda x: dict[x], str)

    return np.fromiter(numbers, dtype=int)

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
labels = np.fromiter(map(lambda x: int(x[0])-1,
                         label_char), dtype= int)
test_labels = np.fromiter(map(lambda x: int(x[0])-1,
                         test_label_char), dtype= int)


mats = np.zeros((len(seq_string), seq_length), dtype = int)
test_mats = np.zeros((len(test_seq_string), seq_length), dtype = int)



# this is pretty slow (optimize in numba)
for ii, seq in enumerate(seq_string):
    # note each line has a \n character at the end so we remove it
    mats[ii,:] = convert_string_to_numbers(seq[:-1], dict_amino).reshape((1,seq_length))

# this is pretty slow (optimize in numba)
for ii, seq in enumerate(test_seq_string):
    # note each line has a \n character at the end so we remove it
    test_mats[ii,:] = convert_string_to_numbers(seq[:-1], dict_amino).reshape((1,seq_length))

N_amino = (np.unique(mats)).shape[0]
sequences = np.eye(N_amino)[mats]
sequences = sequences.reshape((n_samples, -1, seq_length))

N_test_amino = (np.unique(test_mats)).shape[0]
test_sequences = np.eye(N_test_amino)[test_mats]
test_sequences = test_sequences.reshape((len(test_label_char), -1, seq_length))

print("Number of training samples: {}".format(len(sequences)))
print("Number of testing samples: {}".format(len(test_sequences)))

outputTrain = torch.tensor(labels)
inputTrain = torch.tensor(sequences)

datasetTrain = data.TensorDataset(inputTrain, outputTrain)

outputTest = torch.tensor(test_labels)
inputTest = torch.tensor(test_sequences)

datasetTest = data.TensorDataset(inputTest, outputTest)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class _Permutation():
    """Class to store the different permutations"""
    def __init__(self):

        self.permData = np.asarray(list(itertools.permutations(range(4))))
        # hard coded transformation of taxons
        self.permTaxon0 =  torch.tensor([ 0, 0, 1, 2,
                                          1, 2, 0, 0,
                                          1, 2, 1, 2,
                                          2, 1, 2, 1,
                                          0, 0, 2, 1,
                                          2, 1, 0, 0], dtype = torch.long)

        self.permTaxon1 =  torch.tensor([ 1, 2, 0, 0,
                                          2, 1, 2, 1,
                                          2, 1, 0, 0,
                                          0, 0, 1, 2,
                                          1, 2, 1, 2,
                                          0, 0, 2, 1 ], dtype = torch.long)

        self.permTaxon2 =  torch.tensor([ 2, 1, 2, 1,
                                          0, 0, 1, 2,
                                          0, 0, 2, 1,
                                          1, 2, 0, 0,
                                          2, 1, 0, 0,
                                          1, 2, 1, 2 ], dtype = torch.long)

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
    """Class to collate the permuted data"""
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
            Gen3 = Gen2.view(24 * sizeBatch, 80, 1550)

            Labels = torch.stack(LabelData)
            Labels2 = Labels.view(-1)

            return (Gen3, Labels2)

##############################################################
#           Building the networks                           #
##############################################################


## This is from Zou et al. Extracted from the repository
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

# cunstom 
collate_fc = _Collate()

dataloaderTrain = torch.utils.data.DataLoader(datasetTrain,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              collate_fn=collate_fc)

dataloaderTest = torch.utils.data.DataLoader(datasetTest,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             collate_fn=collate_fc)

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# specify loss function
criterion = torch.nn.CrossEntropyLoss(reduction='sum')
# criterion = torch.nn.CrossEntropyLoss()

# define the model
model = _Model().to(device)

print("number of parameters is %d"%count_parameters(model))

# specify loss function
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# specify scheduler
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.9)

perm = _Permutation()


print("Starting Training Loop")

maxAccuracy = 0
"""
for epoch in range(1, n_epochs + 1):
    # monitor training loss
    train_loss = 0.0
    model.train()
    ###################
    # train the model #
    ###################
    for genes, quartets_batch in dataloaderTrain:
        # send to the device (either cpu or gpu)
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
    train_loss = train_loss / len(dataloaderTrain)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch,
        train_loss
    ), flush=True)
    
    # advance the step in the scheduler
    exp_lr_scheduler.step() 
    
    # we compute the test loss every 10 epochs
    if epoch % 10 == 0:
        

        model.eval()
        correct, total = 0, 0
        perm0 =  torch.tensor([ 0, 0, 1, 2,
                                1, 2, 0, 0,
                                1, 2, 1, 2,
                                2, 1, 2, 1,
                                0, 0, 2, 1,
                                2, 1, 0, 0], dtype = torch.long)

        perm1 =  torch.tensor([ 1, 2, 0, 0,
                                2, 1, 2, 1,
                                2, 1, 0, 0,
                                0, 0, 1, 2,
                                1, 2, 1, 2,
                                0, 0, 2, 1 ], dtype = torch.long)

        perm2 =  torch.tensor([ 2, 1, 2, 1,
                                0, 0, 1, 2,
                                0, 0, 2, 1,
                                1, 2, 0, 0,
                                2, 1, 0, 0,
                                1, 2, 1, 2 ], dtype = torch.long)

        for genes, quartets_batch in dataloaderTest:
            # send to the device (either cpu or gpu)
            genes, quartets_batch = genes.to(device).float(), quartets_batch.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            quartetsNN = model(genes)
            # calculate the loss
            _, predicted = torch.max(quartetsNN, 1)
            intruth = quartets_batch.reshape(-1,24)
            inpred = predicted.reshape(-1,24)
            for i in range(0,int(len(quartets_batch)/24)): 
              truelabel = intruth[i,0]  
              predictarray = []
              for j in range(0,24):
                if(inpred[i,j]==perm0[j]):
                  predictarray.append(0)
                elif(inpred[i,j]==perm1[j]):
                  predictarray.append(1)
                elif(inpred[i,j]==perm2[j]):
                  predictarray.append(2)  
                else:
                  predictarray.append(-1)
              predictlabel =  max(set(predictarray), key = predictarray.count)  
              if(truelabel == predictlabel):
                correct +=1          
              total += 1

        accuracyTest = correct / total


        print('Epoch: {} \tTest accuracy: {:.6f}'.format(epoch,
                                                         accuracyTest))
        if accuracyTest > maxAccuracy:
            maxAccuracy = accuracyTest


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
        
# we write the last data to a file
with open(summary_file, 'a') as f:
    f.write("{} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \n".format(nameScript.split(".")[0],
                                    nameJson.split(".")[0],
                                    label_files[test_index], 
                                    str(lr), 
                                    str(batch_size), 
                                    str(maxAccuracy), 
                                    str(train_loss),
                                    str(n_epochs)))
"""
