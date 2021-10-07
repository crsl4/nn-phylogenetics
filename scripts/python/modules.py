
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



#################################
# these are the modules acting on vectors:
# it seems that there is geometric information that needs to be 
# be kept

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


class _NonLinearEmbeddingFourier(torch.nn.Module):

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


# perhaps we need to use conv nets for merges too. 
# using dense networks seems to make the problems hard to optimize
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

    def __init__(self, emb_dim, depth):
        super().__init__()
        
        levels = np.int(np.log2(emb_dim))
        blocks = []
        for ii in range(depth):
            blocks.append(_ResidueModuleDense(emb_dim,emb_dim))
        
        self.layers = nn.Sequential(*blocks)

        self.score = torch.nn.Linear(emb_dim, 1, bias = False)

    def forward(self, x, y):
        x = self.layers(x)
        y = self.layers(y)

        z = torch.add(x,y)

        return self.score(z)
# a couple of ideas for the merge module. 


############################################################################
############################################################################
############################################################################
# here are the version of the modules that keep the geometric information 
# inside each sequence


class _NonLinearEmbeddingConv(torch.nn.Module):

    def __init__(self, input_dim, input_channel, chnl_dim, emb_dim):
        super().__init__()
        
        self.chnl_dim = chnl_dim
        self.num_levels = np.int(np.floor(np.log2((input_dim)/emb_dim)))

        blocks = []
        blocks.append(torch.nn.Conv1d(input_channel, chnl_dim, 1))
        blocks.append(torch.nn.BatchNorm1d(chnl_dim)) 
        for ii in range(self.num_levels):
            blocks.append(_ResidueModule(chnl_dim))
            blocks.append(_ResidueModule(chnl_dim))
            blocks.append(torch.nn.AvgPool1d(2))
        
        self.seq = nn.Sequential(*blocks)

        self.out_dim = input_dim//(2**self.num_levels)
        self.dense = nn.Linear(self.out_dim*chnl_dim, emb_dim*chnl_dim)

    def forward(self, x):
        x = self.seq(x)
        x = self.dense(x.view(x.shape[0], self.out_dim*self.chnl_dim))

        # return as a 1D vector 
        return x.view(x.shape[0], self.chnl_dim, -1)


class _NonLinearMergeConv(torch.nn.Module):
    """Class for merging two different branches"""

    def __init__(self, chnl_dim, kernel_size, depth,
                 dropout_bool=False, dropout_prob=0.2):
        super().__init__()
        
        blocks_mid = []
        for ii in range(depth//2):
            blocks_mid.append(ResNetBlock(chnl_dim, kernel_size, 1))
            if dropout_bool:
                blocks_mid.append(nn.Dropout(dropout_prob))
        
        self.layers_mid = nn.Sequential(*blocks_mid)

        blocks_embed = []
        for ii in range(depth//2):
            blocks_embed.append(ResNetBlock(chnl_dim, kernel_size, 1))
            if dropout_bool:
                blocks_embed.append(nn.Dropout(dropout_prob))
        
        self.layers_embed = nn.Sequential(*blocks_embed)


    def forward(self, x, y):
        # applying the \Phi
        x = self.layers_mid(x)
        y = self.layers_mid(y)

        z = torch.add(x,y)

        return self.layers_embed(z)


# class _NonLinearScoreEmbed(torch.nn.Module):

#     def __init__(self, emb_dim, mid_dim, depth):
#         super().__init__()
        
#         self.dense1 = torch.nn.Linear(emb_dim, mid_dim)

#         blocks_mid = []
#         for ii in range(depth):
#             blocks_mid.append(_ResidueModuleDense(emb_dim,emb_dim))
        
#         self.layers_mid = nn.Sequential(*blocks_mid)

#         blocks_score = []

#         self.levels = np.int(np.log2(emb_dim))-1

#         for ii in range(self.levels):
#             in_dim = emb_dim//(2**ii)
#             out_dim = emb_dim//(2**(ii+1))
#             blocks_score.append(_ResidueModuleDense(in_dim,out_dim))
        
#         self.layers_score = nn.Sequential(*blocks_score)

#         self.dense2 = torch.nn.Linear(emb_dim//(2**(self.levels)), 1)


#     def forward(self, x, y):
#         # applying the \Phi
#         x = self.layers_mid(self.dense1(x))
#         y = self.layers_mid(self.dense1(y))

#         z = torch.add(x,y)

#         return self.dense2(self.layers_score(z))

# ## We suppose that the 

class _NonLinearScoreConv(torch.nn.Module):

    def __init__(self, chnl_dim, kernel_size, depth,
                 dropout_bool = False, dropout_prob = 0.2):
        super().__init__()
         

        if dropout_bool:
            print("using dropout layers")

        blocks_merge = []
        for ii in range(depth//2):
            blocks_merge.append(ResNetBlock(chnl_dim, kernel_size, 1))
            if dropout_bool:
                blocks_merge.append(nn.Dropout(dropout_prob))
        
        self.layers_merge = nn.Sequential(*blocks_merge)

        blocks_score = []
        for ii in range(depth//2):
            blocks_score.append(ResNetBlock(chnl_dim, kernel_size, 1))
            if dropout_bool:
                blocks_score.append(nn.Dropout(dropout_prob))

        blocks_score.append(torch.nn.AdaptiveAvgPool1d(1))
        
        self.layers_score = nn.Sequential(*blocks_score)

        self.score = torch.nn.Linear(chnl_dim, 1, bias = False)

    def forward(self, x, y):

        # first we apply the merge blocks
        x = self.layers_merge(x)
        y = self.layers_merge(y)

        # we add the 2 parts together
        z = torch.add(x,y)

        # we applyt the scoring layers
        z = self.layers_score(z).squeeze(dim=2)

        return self.score(z)
# a couple of ideas for the merge module. 


## here are the ResNet modules used for the Unet
class ConvCircBlock(nn.Module):
    """class that computes a 1D convolution by a periodic padding"""
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
                       x[:,:,:self.padding]], dim = 2)
        # todo: use the torch.nn.functional.pad function

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




################################################################
#  Autoencoder layers
################################################################

## TODO: these need to be properly tested! 

class Encoder(nn.Module):
    def __init__(self, input_shape, encoded_dim, kernel_size = 3, num_layers = 3, embed_dim = 4, batch_norm = True, 
                 act_fn=torch.tanh, norm_first= True, dropout_bool = False, dropout_prob = 0.2):
        super(Encoder, self).__init__()
        ## encoder layers ##
        
        # check that the input is divisible by the propery power of the 
        # number of layers
        assert input_shape%(2**num_layers) == 0

        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.act_fn = act_fn
        
        # todo: apply the normalization either 
        # before of after the activation 
        self.norm_first = norm_first
        self.dropout_bool = dropout_bool

        layers_conv = []
        batch_norm_layers = []

        self.pool = nn.MaxPool1d(2, 2)

        # adding the embedding layer 
        self.emb = nn.Conv1d(20, embed_dim, 1, padding=0) 
        self.batch_norm_emb = nn.BatchNorm1d(embed_dim)

        for ii in range(num_layers):
            layers_conv.append(nn.Conv1d(embed_dim*(2**ii), 
                                         embed_dim*(2**(ii+1)), 
                                         kernel_size, 
                                         padding=kernel_size//2))
            batch_norm_layers.append(nn.BatchNorm1d(embed_dim*(2**(ii+1))))


        self.layers_conv = nn.ModuleList(layers_conv)
        self.batch_norm_layers = nn.ModuleList(batch_norm_layers)

        # to use later
        self.drop_out = torch.nn.Dropout(dropout_prob)

        ## dense layers
        self.dense = nn.Linear((input_shape*embed_dim), encoded_dim)


    def forward(self, x):
        ## encode ##
        
        # we perform a simple embedding a this point
        x = self.act_fn(self.emb(x))
        
        if self.batch_norm:
            x = self.batch_norm_emb(x)


        for conv_l, norm_l in zip(self.layers_conv,\
                              self.batch_norm_layers):
            x = self.act_fn(conv_l(x))
            if self.batch_norm:
                x = norm_l(x)
            if self.dropout_bool:
                x = self.drop_out(x)

            # pooling the information
            x = self.pool(x) 
 
        x = self.dense(x.view(x.shape[0], -1))
        
        # compressed representation
        return x

class Decoder(nn.Module):
    def __init__(self, input_shape, encoded_dim, num_layers=3, 
                embed_dim = 4, batch_norm = True, 
                 act_fn = torch.tanh, norm_first= True, 
                 dropout_bool = False, dropout_prob = 0.2):
        super(Decoder, self).__init__()

        # TODO: add flag for using the batch normalization 
        # either after or before
        ## decoder layers ##
        
        assert input_shape%(2**num_layers) == 0

        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.act_fn = act_fn
        self.embed_dim = embed_dim
        
        # todo: apply the normalization either 
        # before of after the activation 
        self.norm_first = norm_first
        self.dropout_bool = dropout_bool

        ## dense layers
        self.dense = nn.Linear(encoded_dim, (input_shape*embed_dim))
        self.batch_norm_dense = nn.BatchNorm1d(input_shape*embed_dim)

        layers_deconv = []
        batch_norm_layers = []

        for ii in range(num_layers):
            layers_deconv.append(
                nn.ConvTranspose1d(embed_dim*(2**(num_layers-ii)), 
                                   embed_dim*(2**(num_layers-ii-1)), 
                                   2, stride=2))
            batch_norm_layers.append(
                nn.BatchNorm1d(embed_dim*(2**(num_layers-ii-1))))


        self.layers_deconv = nn.ModuleList(layers_deconv)
        self.batch_norm_layers = nn.ModuleList(batch_norm_layers)

        # to use later
        self.drop_out = torch.nn.Dropout(dropout_prob)

        self.emb = nn.Conv1d(embed_dim, 20, 1, padding=0) 
        

    def forward(self, x):
        ## decode ##
        # size of x  (batch_size, self.encoded_dim)

        x = self.act_fn(self.dense(x))
        # (batch_size, self.input * self.embd_dim)

        if self.batch_norm:
            x = self.batch_norm_dense(x)
        # (batch_size, self.input * self.embd_dim)

        x = x.view(x.shape[0], (2**self.num_layers)*self.embed_dim, -1)
        # (batch_size, (2**num_layers)*embed_dim, self.input/((2**num_layers)*embed_dim))

        for deconv_l, norm_l in zip(self.layers_deconv,\
                                    self.batch_norm_layers):

            x = self.act_fn(deconv_l(x))
            if self.batch_norm:
                x = norm_l(x)
            if self.dropout_bool:
                x = self.drop_out(x)

        x = self.emb(x)

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




################################################################
#  1d fourier layer
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """


        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * 
                                     torch.rand(in_channels, 
                                                out_channels, 
                                                self.modes1, 
                                                dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        ## Multiply relevant Fourier modes
        # create the output, in which the high frequency modes are already zeroeth
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)

        # compute the multiplication and add to the lowest frequency modes
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))

        return x

class SpectralTwoScalesResNet1d(nn.Module):

    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        self.fourier_layer = SpectralConv1d(in_channels, out_channels, modes1)
        self.conv1d = nn.Conv1d(self.width, self.width, 1)

    def forward(x):

        x1 = self.fourier_layer(x)
        x2 = self.conv1d(x)
        x = x1 + x2
        x = F.gelu(x)

        return x