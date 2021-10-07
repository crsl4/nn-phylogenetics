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
    