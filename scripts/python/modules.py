import numpy as np 
import torch

# class modules
# In this file we collect all the modules so we don't need to re-write them 
# at every file 



class _ResidueModule(torch.nn.Module):
  '''One-dimensional convolutional residual 
   we only provide the channel count 
   the stride and the filter width are fixed to zero. 
   In particular here we just have channel mixing '''

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
  '''Dense residual
  input : size_in:    dimension of the input vector
           size_out:   dimension fo the ouput vector
  the stride and the filter width are fixed to zero. 
  In particular here we just have channel mixing '''

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
      # if the dimension are the same, return a resnet block
      return x + self.layers(x)
    elif self.size_out ==  self.size_in/2:
      # if the size_out is half of size_in, use an 
      # average pooling of size two
      return  0.5*torch.sum(x.view(x.size()[0],-1,2 ), 2) \
             + self.layers(x)
    else: 
      # if there is not relation do not add anythin   
      return self.layers(x)
    # TODO: add 
