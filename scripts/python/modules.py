
import numpy as np
import torch
import torch.nn as nn

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
        if self.size_out == self.size_in:
            # if the dimension are the same, return a resnet block
            return x + self.layers(x)
        elif self.size_out == self.size_in / 2:
            # if the size_out is half of size_in, use an
            # average pooling of size two
            return 0.5 * torch.sum(x.view(x.size()[0], -1, 2), 2)\
            + self.layers(x)
        else:
            # if there is not relation do not add anythin
            return self.layers(x)
    # TODO: add


class _NonLinearEmbedding(torch.nn.Module):

    def __init__(self, input_dim, input_channel, chnl_dim, emb_dim):
        super().__init__()
        
        self.num_levels = np.int(np.floor(np.log((input_dim*chnl_dim)/emb_dim)/np.log(2)))

        blocks = []
        blocks.append(torch.nn.Conv1d(input_channel, chnl_dim, 1))
        blocks.append(torch.nn.BatchNorm1d(chnl_dim)) 
        for ii in range(self.num_levels):
            blocks.append(_ResidueModule(chnl_dim))
            blocks.append(torch.nn.AvgPool1d(2))
        
        self.seq = nn.Sequential(*blocks)

        dims = np.int(input_dim//2**self.num_levels)


        self.dense = _ResidueModuleDense(dims*chnl_dim,emb_dim)
        self.dense2 = _ResidueModuleDense(emb_dim,emb_dim)

    def forward(self, x):

        x = self.seq(x)
        x = x.view(x.shape[0], -1)
        x = self.dense(x)
        x = self.dense2(x)

        # return as a 1D vector 
        return x.view(x.shape[0], -1)


class _NonLinearMerge(torch.nn.Module):

    def __init__(self, emb_dim, depth):
        super().__init__()
        
        blocks = []
        for ii in range(depth):
            blocks.append(_ResidueModuleDense(emb_dim,emb_dim))
        
        self.layers = nn.Sequential(*blocks)


    def forward(self, x):
        return self.layers(x)


class _NonLinearMergeEmbed(torch.nn.Module):

    def __init__(self, emb_dim, mid_dim, depth):
        super().__init__()
        
        self.dense1 = torch.nn.Linear(emb_dim, mid_dim)

        blocks_mid = []
        for ii in range(depth//2):
            blocks_mid.append(_ResidueModuleDense(emb_dim,emb_dim))
        
        self.layers_mid = nn.Sequential(*blocks_mid)

        blocks_embed = []
        for ii in range(depth//2):
            blocks_embed.append(_ResidueModuleDense(emb_dim,emb_dim))
        
        self.layers_embed = nn.Sequential(*blocks_embed)

        self.dense2 = torch.nn.Linear(mid_dim, emb_dim)


    def forward(self, x, y):
        # applying the \Phi
        x = self.layers_mid(self.dense1(x))
        y = self.layers_mid(self.dense1(y))

        z = torch.add(x,y)

        return self.dense2(self.layers_embed(z))


class _NonLinearScoreEmbed(torch.nn.Module):

    def __init__(self, emb_dim, mid_dim, depth):
        super().__init__()
        
        self.dense1 = torch.nn.Linear(emb_dim, mid_dim)

        blocks_mid = []
        for ii in range(depth):
            blocks_mid.append(_ResidueModuleDense(emb_dim,emb_dim))
        
        self.layers_mid = nn.Sequential(*blocks_mid)

        blocks_score = []

        self.levels = np.int(np.log2(emb_dim))-1

        for ii in range(self.levels):
            in_dim = emb_dim//(2**ii)
            out_dim = emb_dim//(2**(ii+1))
            blocks_score.append(_ResidueModuleDense(in_dim,out_dim))
        
        self.layers_score = nn.Sequential(*blocks_score)

        self.dense2 = torch.nn.Linear(emb_dim//(2**(self.levels)), 1)


    def forward(self, x, y):
        # applying the \Phi
        x = self.layers_mid(self.dense1(x))
        y = self.layers_mid(self.dense1(y))

        z = torch.add(x,y)

        return self.dense2(self.layers_score(z))

## We suppose that the 

class _NonLinearScore(torch.nn.Module):

    def __init__(self, emb_dim):
        super().__init__()
        
        levels = np.int(np.log2(emb_dim))
        blocks = []
        for ii in range(depth):
            blocks.append(_ResidueModuleDense(emb_dim,emb_dim))
        
        self.layers = nn.Sequential(*blocks)


    def forward(self, x):
        return self.layers(x)

# a couple of ideas for the merge module. 


## here are the ResNet modules used for the Unet
class ConvCircBlock(nn.Module):
    def __init__(self, in_layer, out_layer, 
                 kernel_size, stride, dilation, bias=False):
        super(ConvCircBlock, self).__init__()

        self.padding = kernel_size//2

        # in this case we want to use circular padding
        self.conv1 = nn.Conv1d(in_layer, out_layer, kernel_size=kernel_size, 
                               stride=stride, dilation=dilation, 
                               padding=0, bias=bias)
        self.bn = nn.BatchNorm1d(out_layer)
        self.relu = nn.ReLU()
    
    def forward(self,x):

        # adding the padding
        x = torch.cat([x[:,:,-self.padding:], 
                       x[:,:,:], 
                       x[:,:,:self.padding]], axis = 2)
        x = self.conv1(x)
        x = self.bn(x)
        out = self.relu(x)
        
        return out

class ResNetBlock(nn.Module):
    '''ResNet block using the circular convolutional block above'''
    def __init__(self, in_out_layer, kernel_size, dilation, bias=False):
        super(ResNetBlock, self).__init__()
        
        self.cbr1 = ConvCircBlock(in_out_layer,in_out_layer, kernel_size, 1, dilation, bias)
        self.cbr2 = ConvCircBlock(in_out_layer,in_out_layer, kernel_size, 1, dilation, bias)
        # self.seblock = ScalarBlock(out_layer, out_layer)
    
    def forward(self,x):

        x_re = self.cbr1(x)
        x_re = self.cbr2(x_re)
        # x_re = self.seblock(x_re)
        x_out = torch.add(x, x_re)
        return x_out      

    

class UnetModule(nn.Module):
    def __init__(self ,input_dim, layer_n, kernel_size, depth, num_levels, bias=False):
        super(UnetModule, self).__init__()
        self.input_dim = input_dim
        self.layer_n = layer_n
        self.kernel_size = kernel_size
        self.depth = depth
        self.bias = bias

        self.num_levels = num_levels
        
        self.downsample = nn.AvgPool1d(input_dim, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.chnls_lvls = self.layer_n*np.power(2, 
                                np.linspace(0, self.num_levels, 
                                            self.num_levels+1,dtype=np.int))
        # self.chnls_lvls = np.concatenate([self.chnls_lvls, self.chnls_lvls[-1:]])

        self.layers = {}

        self.layers[0] = self.down_layer(self.input_dim, self.layer_n, 
                                           self.kernel_size,1, self.depth, self.bias )

        for ii in range(self.num_levels):
            layer_last = self.chnls_lvls[ii]
            layer_next = self.chnls_lvls[ii+1]

            self.layers[ii+1]= self.down_layer(layer_last, layer_next, 
                                               self.kernel_size, 1, self.depth ,self.bias)

        # defining up-sampling layers
        self.layers_up = {}
        self.layers_up[0] =self.down_layer(2*self.layer_n, self.input_dim, 
                                           self.kernel_size,1, self.depth, self.bias)

        for ii in range(self.num_levels-1):
            layer_last = self.chnls_lvls[ii]
            layer_next = self.chnls_lvls[ii+1]

            self.layers_up[ii+1]= self.down_layer(2*layer_next, layer_last, 
                                               self.kernel_size, 1, self.depth, self.bias)

        # the last one is special
        ii = self.num_levels-1
        layer_last = self.chnls_lvls[ii]
        layer_next = self.chnls_lvls[ii+1]

        self.layers_up[ii+1]= self.down_layer(layer_next, layer_last, 
                                              self.kernel_size, 1, self.depth, self.bias)


        self.conv_block_init = ConvCircBlock(1, self.input_dim, self.kernel_size, 1, 1, self.bias)
        self.conv_block_end = ConvCircBlock(self.input_dim, 1, self.kernel_size, 1, 1, self.bias)

        
    def down_layer(self, input_layer, out_layer, kernel, stride, depth, bias = False):
        block = []
        block.append(ConvCircBlock(input_layer, out_layer, kernel, stride, 1, bias))
        for i in range(depth):
            block.append(ResNetBlock(out_layer, kernel, 1, bias))
        return nn.Sequential(*block)
            
    def forward(self, x):
        
        # first one
        x = self.conv_block_init(x)
        # [None, nx, self.input_dim]

        xtemp = {}
        xtemp[0] = self.layers[0](x)

        for ii in range(self.num_levels):
            xtemp[ii+1] = self.layers[ii+1](self.downsample(xtemp[ii]))
            # [None, nx//2**(ii+1),self.chnls_lvls[ii+1]]

        xtot = self.layers_up[self.num_levels](xtemp[self.num_levels])  
        # [None, nx//2**(end),self.chnls_lvls[end]]

        for ii in range(self.num_levels)[::-1]:

            xmerge = torch.cat([self.upsample(xtot),xtemp[ii]], axis = 1) # channel merge
            xtot = self.layers_up[ii](xmerge)


        out = self.conv_block_end(xtot)

        #out = nn.functional.softmax(out,dim=2)
        
        return out



class ScalarBlock(nn.Module):
    def __init__(self,in_layer, mid_layer, out_layer):
        super(ScalarBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_layer, mid_layer, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(mid_layer, in_layer, kernel_size=1, padding=0)
        self.fc = nn.Linear(1,mid_layer)
        self.fc2 = nn.Linear(mid_layer,out_layer)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):

        x_se = nn.functional.adaptive_avg_pool1d(x,1)
        x_se = self.conv1(x_se)
        x_se = self.relu(x_se)
        x_se = self.conv2(x_se)
        x_se = self.sigmoid(x_se)
        
        x_out = torch.add(x, x_se)

        return x_out

