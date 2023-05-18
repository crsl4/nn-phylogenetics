import numpy as np
import torch

torch.manual_seed(0)

import torch.nn as nn
from torch_geometric.data import Data
import itertools
import json
import sys
import time
import os
from os import path
sys.path.insert(0, '../')
import gc
gc.collect()

# get name of the script
nameScript = sys.argv[0].split('/')[-1]
# get json file name of the script
nameJson = sys.argv[1]

print("------------------------------------------------------------------------")
print("Training the Garph Auto Encoder for 5-taxa dataset")
print("------------------------------------------------------------------------")
print("Executing " + nameScript + " following " + nameJson, flush = True)

# opening Json file 
jsonFile = open(nameJson) 
dataJson = json.load(jsonFile)

# loading the input data from the json file
ngpu = dataJson["ngpu"]                  # number of GPUS
lr = dataJson["lr"]                      # learning rate
# TODO: batch size
# TODO: number of epoch

data_root = dataJson["dataRoot"]         # data folder
model_root = dataJson["modelRoot"]       # folder to save the data

label_files = dataJson["labelFile"]      # file with labels
sequence_files = dataJson["matFile"]     # file with sequences


if "summaryFile" in dataJson:
    summary_file = dataJson["summaryFile"]
else :
    summary_file = "summary_file.txt"


print("------------------------------------------------------------------------")
print("Loading Sequence Data in " + sequence_files, flush = True)
print("Loading Label Data in " + label_files, flush = True)

# we read the labels as list of strings
with open(data_root+label_files, 'r') as f:
    label_char = f.readlines()

# we read the sequence as a list of strings
with open(data_root+sequence_files, 'r') as f:
    seq_string = f.readlines()

n_samples = len(label_char)
seq_length = len(seq_string[0])-1
print("Number of samples:{}; Sequence length of each sample:{}"
        .format(n_samples, seq_length))
print("------------------------------------------------------------------------")

# function to convert string to numbers
def convert_string_to_numbers(str, dict):
    ''' str: string to convert
        dict dictionary with the relative ordering of each char'''
            # create a map iterator using a lambda function
    # lambda x -> return dict[x]
    # This return the value for each key in dict based on str
    numbers = map(lambda x: dict[x], str)
    # return an array of int64 numbers
    return np.fromiter(numbers, dtype=np.int64)

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

# looping over the labels and create array. Here each element of the
# label_char has the form "1\n", so we only take the first one
labels = np.fromiter(map(lambda x: int(x[0])-1,
                         label_char), dtype= np.int64)


# deal with the edge set
def create_edge_set(label):
    ''' the edge set only depend on the label of the original dataset '''

