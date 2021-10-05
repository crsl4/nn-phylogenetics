# Script containing the modular version of the code
# so far we have only implemented the non_linear_embedding layer
# for simplicity we can just use dense layers for the merge operations


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

import torch.nn.functional as F

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
encoded_dim = dataJson["encoded dimension"]


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


# we read the labels as list of strings
with open(dataRoot+"/"+label_file, 'r') as f: 
    label_char = f.readlines() 

# we read the sequence as a list of strings
with open(dataRoot+"/"+mat_file, 'r') as f: 
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

    return np.fromiter(numbers, dtype=np.int32)

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
                         label_char), dtype=np.int)

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

# we need to truncate a bit the lenght of the sequences
trunc_length = 1548

# we perform the training/validation splitting
outputTrain = torch.from_numpy(labels[0:n_train_samples])
inputTrain  = torch.from_numpy(mats[0:n_train_samples, :, :trunc_length])

outputTest = torch.from_numpy(labels[-n_test_samples:-1])
inputTest  = torch.from_numpy(mats[-n_test_samples:-1, :,:trunc_length])

# # creating the dataset objects
# datasetTrain = data.TensorDataset(inputTrain, outputTrain) 
# datasetTest = data.TensorDataset(inputTest, outputTest) 

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

class SequenceEncoderDataSet(torch.utils.data.Dataset):
    """Data set to generate the matrices with the hot encoding on the fly"""

    def __init__(self, sequences, n_char = 20, transform=None):
        """
        Args:  sequences: pytorch tensor with the sequences
               labels:    pytorch tensor with the labels
        """
        self.sequences = sequences
        # number of characters
        self.n_char = n_char

    def __len__(self):
        return self.sequences.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sequence_out = self.sequences[idx,:,:]
        seq_stack = []

        for ii in range(4):
            temp = torch.nn.functional.one_hot(sequence_out[ii,:], 
                                                      self.n_char)
            # we need to transpose it. Perhaps is better to 
            # transpose everything at the end.
            temp = torch.transpose(temp, 0, 1)
            seq_stack.append(temp)

        seq_stack = torch.stack(seq_stack, dim=0)

        sample = (seq_stack, sequence_out)

        return sample
    
    
# we use this dataset that perform the one_hot encoding on the fly 
dataset_train_auto = SequenceEncoderDataSet(inputTrain) 
dataset_test_auto = SequenceEncoderDataSet(inputTest) 

### Todo: Build the autoencoder here

# #define the NN architecture, we use a slightly different architecture
# #define the NN architecture, we use a slightly different architecture
class Encoder(nn.Module):
    def __init__(self, input_shape, encoded_dim, embed_dim = 4, batch_norm = True, 
                 act_fn = torch.tanh, norm_first= True, drop_out_bool = False):
        super(Encoder, self).__init__()
        ## encoder layers ##
        
        self.batch_norm = batch_norm
        self.act_fn = act_fn
        
        # todo: apply the normalization either 
        # before of after the activation 
        self.norm_first = norm_first
        self.drop_out_bool = drop_out_bool
        
        # we perform a simple embedding a this point
        # there is no comunication in the location of the sequence
        # only in channel dimension
        self.conv1 = nn.Conv1d(20, embed_dim, 1, padding=0)  
        
        # conv layer (start with embed_dim continue to 2*embed_dim) 3 kernels
        self.conv2 = nn.Conv1d(embed_dim, 2*embed_dim, 3, padding=1)  
        
        # conv layer (depth from 2*embed_dim --> 4*embed_dim), 3x3 kernels
        self.conv3 = nn.Conv1d(2*embed_dim, 4*embed_dim, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool1d(2, 2)
        
        self.batch_norm0 = nn.BatchNorm1d(embed_dim)
        self.batch_norm1 = nn.BatchNorm1d(2*embed_dim)
        self.batch_norm2 = nn.BatchNorm1d(4*embed_dim)
        
        # to use later
        self.drop_out = torch.nn.Dropout(0.2)

        ## dense layers
        self.dense = nn.Linear((input_shape//4)*(4*embed_dim), encoded_dim)


    def forward(self, x):
        ## encode ##
        
        # we perform a simple embedding a this point
        x = self.act_fn(self.conv1(x))
        if self.batch_norm:
            x = self.batch_norm0(x)
        
        x = self.act_fn(self.conv2(x))
        # performing batch normalization
        if self.batch_norm:
            x = self.batch_norm1(x)
        # activation function
        
        if self.drop_out_bool:
            x = self.drop_out(x)

        # pooling
        x = self.pool(x)
        
        # add second hidden layer
        x = self.act_fn(self.conv3(x))
        # performing batch normalization
        if self.batch_norm:
            x = self.batch_norm2(x)

        if self.drop_out_bool:
            x = self.drop_out(x)
        
        x = self.pool(x)  
        
        x = self.dense(x.view(x.shape[0], -1))
        
        # compressed representation
        return x

class Decoder(nn.Module):
    def __init__(self, input_shape, encoded_dim, embed_dim = 4, batch_norm = True, 
                 act_fn = torch.tanh, norm_first= True, drop_out_bool = False):
        super(Decoder, self).__init__()

        # TODO: add flag for using the batch normalization 
        # either after or before
        ## decoder layers ##
        self.batch_norm = batch_norm
        self.act_fn = act_fn
        self.embed_dim = embed_dim
        self.drop_out_bool = drop_out_bool
        

        self.dense = nn.Linear(encoded_dim, (input_shape//4)*4*embed_dim)

        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose1d(4*embed_dim, 2*embed_dim, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose1d(2*embed_dim, embed_dim, 2, stride=2)
        
        self.conv3 = nn.Conv1d(embed_dim, 20, 1, padding=0)  
        
        self.batch_norm1 = nn.BatchNorm1d(4*embed_dim)
        self.batch_norm2 = nn.BatchNorm1d(2*embed_dim)
        self.batch_norm3 = nn.BatchNorm1d(embed_dim)

        self.drop_out = torch.nn.Dropout(0.2)


    def forward(self, x):
        ## decode ##

        x = self.act_fn(self.dense(x))
        x = x.view(x.shape[0], 4*self.embed_dim, -1)
        
        if self.batch_norm:
            x = self.batch_norm1(x)
        
        if self.drop_out_bool:
            x = self.drop_out(x)


        x = self.act_fn(self.t_conv1(x))
        
        if self.batch_norm:
            x = self.batch_norm2(x)
        
        if self.drop_out_bool:
            x = self.drop_out(x)

        x = self.act_fn(self.t_conv2(x))
        
        if self.batch_norm:
            x = self.batch_norm3(x)
        
        if self.drop_out_bool:
            x = self.drop_out(x)

        x = self.conv3(x)

        # output layer (with sigmoid for scaling from 0 to 1)
        return torch.sigmoid(x)
    

class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(AutoEncoder, self).__init__()

        ## decoder layers ##
    
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        
        ## encode ##
        x = self.encoder(x)
        ## decode ##
        x = self.decoder(x)
                
        return x

    
# building the simple encoder-decoder, and using a torch script to 
# accelerate the computation

print("we activate the  drop out layers")

encoder = torch.jit.script(Encoder(trunc_length, encoded_dim, 
                                   embed_dim= embd_dim, batch_norm = True,
                                   act_fn = F.elu, norm_first= True, drop_out_bool = False))
decoder = torch.jit.script(Decoder(trunc_length, encoded_dim, 
                                   embed_dim= embd_dim, batch_norm = True,
                                   act_fn = F.elu, norm_first= True, drop_out_bool = False))

autoencoder = torch.jit.script(AutoEncoder(encoder, decoder))

    
# building the data sets (no need for special collate function)
dataloader_train_auto = torch.utils.data.DataLoader(dataset_train_auto, 
                                              batch_size=batch_size,
                                              shuffle=True, 
                                              num_workers=8)#,
                                              #pin_memory=False )

dataloader_test_auto = torch.utils.data.DataLoader(dataset_test_auto, 
                                             batch_size=batch_size,
                                             num_workers=8,
                                             shuffle=True)


# specify loss function
# criterion = torch.nn.CrossEntropyLoss(reduction='sum')

criterion = torch.nn.MSELoss()

# specify optimizer
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)

# specidy scheduler
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                             step_size=lr_steps, gamma=gamma)

# model.load_state_dict(torch.load("best_models/saved_permutation_model_shallow_augmented_best_batch_16.pth"))
# model.eval()

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

autoencoder = autoencoder.to(device)

print("Starting Training Loop")

maxAccuracy = 0

############################################################################
################### Training the auto encoder first ########################
############################################################################

# TODO: add different stages for the training: start with small batch, and then 
# increase it, and then decrease the learning rate. 

for epoch in range(1, nEpochs+1):
    # monitor training loss
    train_loss = 0.0
    autoencoder.train()
    ###################
    # train the autoencoder #
    ###################
    for genes_one_hot, genes in dataloader_train_auto:
        #send to the device (either cpu or gpu)
        genes_one_hot, genes = genes_one_hot.float().to(device), genes.to(device)
        # reshape them 
        genes_one_hot = genes_one_hot.view(genes_one_hot.shape[0]*4, 20, -1)
        genes = genes.view(genes.shape[0]*4, -1)
        
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the autoencoder
        genes_auto = autoencoder(genes_one_hot)
        # calculate the loss
#         loss = criterion(genes_auto, genes)
        
        # we use the MSE error 
        loss = criterion(genes_auto,genes_one_hot)
        
        # backward pass: compute gradient of the loss with respect to autoencoder parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item()

    # print avg training statistics 
    train_loss = train_loss/len(dataloader_train_auto)
    print('Epoch: {} \tLearning rate: {:.6f} \tTraining Loss: {:.6f}'.format(
        epoch, 
        optimizer.param_groups[0]['lr'],
        train_loss
        ), flush=True)

    # advance the step in the scheduler
    exp_lr_scheduler.step() 
    
    # we compute the test accuracy every 10 epochs 
    if epoch % 10 == 0 :

        autoencoder.eval()
        correct, total = 0, 0

        for genes_one_hot, genes in dataloader_test_auto:
            #send to the device (either cpu or gpu)
            genes_one_hot, genes = genes_one_hot.float().to(device), genes.to(device)
            # reshape them 
            genes_one_hot = genes_one_hot.view(genes_one_hot.shape[0]*4, 20, -1)
            genes = genes.view(genes.shape[0]*4, -1)
        
            # forward pass: compute predicted outputs by passing inputs to the autoencoder
            genes_auto = autoencoder(genes_one_hot)
            # calculate the loss
            _, predicted = torch.max(genes_auto, 1)
            
            total += genes.size(0)*genes.size(1)
            correct += (predicted == genes).sum().item()

        accuracyTest = correct/total

        print('Epoch: {} \tTest accuracy: {:.6f}'.format(epoch, 
                                                         accuracyTest))

        if accuracyTest > maxAccuracy:
            maxAccuracy = accuracyTest
            torch.save(autoencoder.state_dict(), modelRoot + "/" +
                "autoencoder_saved_{}_{}_lr_{}_batch_{}_lba_best.pth".format(nameScript.split(".")[0],
                                                                 nameJson.split(".")[0],
                                                                str(lr), 
                                                                 str(batch_size)))


# here we compute the accuracy of the converged network in the training data 
autoencoder.eval()
correct, total = 0, 0

for genes_one_hot, genes in dataloader_train_auto:
    #send to the device (either cpu or gpu)
    genes_one_hot, genes = genes_one_hot.float().to(device), genes.to(device)
    # reshape them 
    genes_one_hot = genes_one_hot.view(genes_one_hot.shape[0]*4, 20, -1)
    genes = genes.view(genes.shape[0]*4, -1)

    # forward pass: compute predicted outputs by passing inputs to the autoencoder
    genes_auto = autoencoder(genes_one_hot)
    # calculate the loss
    _, predicted = torch.max(genes_auto, 1)
    
    total += genes.size(0)*genes.size(1)
    correct += (predicted == genes).sum().item()

accuracyTest = correct/total

print('Ttrain accuracy: {:.6f}'.format(accuracyTest))



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
        # input will be (batch_size, num_sites, 1, length_seq)
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


device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# defining the data sets 
# this is harwired for now

# also defining the data set to train the full model 
datasetTrain = SequenceDataSet(inputTrain, outputTrain) 
datasetTest = SequenceDataSet(inputTest, outputTest) 

# building the data sets (no need for special collate function)
dataloaderTrain = torch.utils.data.DataLoader(datasetTrain, 
                                              batch_size=batch_size,
                                              shuffle=True, 
                                              num_workers=8)
                                              # pin_memory=True )

dataloaderTest = torch.utils.data.DataLoader(datasetTest, 
                                             batch_size=batch_size,
                                             num_workers=8,
                                             shuffle=True)
# we add an embedding layer so we don't need to do 
# the one hot encoding


class _AutoEncoderEmbedding(torch.nn.Module):
    # we use first an embedding for the 

    def __init__(self, encoder, encoded_dim, chnl_dim):
        super().__init__()
        
        self.embedding_layer = encoder
        self.encoded_dim = encoded_dim
        self.chnl_dim = chnl_dim

    def forward(self, x):

        x = self.embedding_layer(x)

        # return as a 1D vector 
        return x.view(x.shape[0],self.chnl_dim, self.encoded_dim//self.chnl_dim )

# we will use the pretrained encoder for the embedding
D = torch.jit.script(_AutoEncoderEmbedding(encoder, encoded_dim, chnl_dim))

# non-linear merge is just a bunch of dense ResNets 
M1 = torch.jit.script(_NonLinearMergeConv(chnl_dim, 3, 6, dropout_bool=True))
M2 = torch.jit.script(_NonLinearScoreConv(chnl_dim, 3, 6, dropout_bool=True))

# model using the permutations
model = torch.jit.script(_PermutationModule(D, M1, M2)).to(device)

## TODO: add the torch script to make the algorithm faster (hopefully)

# specify loss function
criterion = torch.nn.CrossEntropyLoss(reduction='sum')

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
    model.train()
    ###################
    # train the model #
    ###################
    for genes, quartets_batch in dataloaderTrain:
        #send to the device (either cpu or gpu)
        genes, quartets_batch = genes.float().to(device), quartets_batch.to(device)
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
            genes, quartets_batch = genes.float().to(device), quartets_batch.to(device)
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
    f.write("{} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \n".format(nameScript.split(".")[0],
                                    nameJson.split(".")[0],
                                    label_file,
                                    str(lr), 
                                    str(batch_size), 
                                    str(maxAccuracy), 
                                    str(train_loss),
                                    str(nEpochs), 
                                    str(chnl_dim),
                                    str(embd_dim)))
## testing and saving data to centralized file 


