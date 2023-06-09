import numpy as np
import random
import ete3
import torch
import mlflow.pytorch
torch.manual_seed(0)
import torch.nn as nn
from torch_geometric.data import Data
import itertools
import json
import sys
import time
from tqdm import tqdm
import os
from os import path
sys.path.insert(0, '../')
import gc
import torch_geometric.transforms as T
from torch_geometric.nn.conv import TransformerConv
from torch_geometric.nn import VGAE
from torch_geometric.loader import DataLoader
from torch_geometric.utils import batched_negative_sampling
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn import BatchNorm
from torch.nn import BatchNorm1d
from torch.nn import Linear
from ete3 import Tree
gc.collect()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Functions
# The internal node will be in the order of 5-7-6, the single tip will always be connected to node 7


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


# function to create a graph for each 
def construct_single_graph(idx, t):
    ''' idx: the current graph index w.r.t the label
        t: the tree object from ete'''
    # transform the character of amino acid in to numbers for all 5 sequences in this graph
    transformed_x = []
    for i in range(5):
        # get the index of the sequence from the original dataset
        seq_idx = 5*idx + i
        transformed_x.append(convert_string_to_numbers(seq_string[seq_idx][:-1], dict_amino))
        
    # initialize the sequence of 3 internal nodes
    vec_len = len(transformed_x[0])
    internal_node_5 = np.full(vec_len, -1, dtype=np.int64)
    internal_node_6 = np.full(vec_len, -1, dtype=np.int64)
    internal_node_7 = np.full(vec_len, -1, dtype=np.int64)
    
    # Work out the branch distance from the Newick format
    leaf_pair = 0               # The amount of leaf pair so far, max=2
    prev_leaf = False           # Whether the previous leaf in the preorder is a leaf node
    prev_dist = 0               # The distance of the branch coming out of the preivous node in preorder
    dist_array = [0]*8          # The distance for outgoing branch for each node, node 7 will always be 0
    prev_index = -1             # The index of the last leaf node in the preorder
    tot_in_node = 0             # All distance of internal nodes that are unassigned so far
    pending = False             # Some condition for assigning branch length that I don't remember
    preorder=[]                 # The preorder of all leaf nodes
    
    # Traverse through all nodes in preorder, work out the branch distance 
    # There are only 2 possible rooted tree format from ETE, 
    # so 2 if statements that work out all different scenarios
    for node in t.traverse("preorder"):
        if not node.name=='':
            index = int(node.name) - 1
            preorder.append(index)
            dist_array[index] = node.dist
            prev_index = index
            if leaf_pair >= 2:
                tot_in_node += node.dist
                dist_array[index] = tot_in_node
                break
            else:
                if prev_leaf==False:
                    prev_leaf=True
                else:
                    leaf_pair += 1
                    prev_leaf=False
                    if prev_dist != 0:
                        dist_array[leaf_pair+4] = prev_dist
                    else:
                        pending = True
                    tot_in_node-=prev_dist
        else:
            prev_dist = node.dist
            tot_in_node+=node.dist
            if pending==True:
                pending = False
                prev_dist = 0
                tot_in_node -= node.dist
                dist_array[leaf_pair+4] = node.dist
            if prev_leaf==True:
                dist_array[prev_index] += node.dist
                prev_dist = 0
                tot_in_node -= node.dist
    # Set up the adjency Matrix in COO format
    # We find the smaller node number of each side.
    # In this case, the tip with the larger node number is on the left side, thus connect to node 5
    if min(preorder[0], preorder[1]) > min(preorder[2], preorder[3]):
        # change edge value of edge 5 and 6
        # I think this is due to the conditions from the previous part, but I don't remember the details
        # It works though!
        tmp = dist_array[5]
        dist_array[5] = dist_array[6]
        dist_array[6] = tmp
        # Assign edge origin/destination and value
        edge_index = torch.tensor([[preorder[2],5],[5,preorder[2]],[preorder[3],5],[5,preorder[3]],
                                       [5,7],[7,5],[preorder[4],7],[7,preorder[4]],
                                       [6,7],[7,6],[preorder[0],6],[6,preorder[0]],
                                       [preorder[1],6],[6,preorder[1]]], dtype=torch.long)
        edge_attr = [dist_array[preorder[2]], dist_array[preorder[2]], 
                 dist_array[preorder[3]], dist_array[preorder[3]], 
                 dist_array[5], dist_array[5],
                 dist_array[preorder[4]], dist_array[preorder[4]],
                 dist_array[6], dist_array[6],
                 dist_array[preorder[0]], dist_array[preorder[0]],
                 dist_array[preorder[1]], dist_array[preorder[1]]]
        # Assign the value for internal node 5 and 6, based on the 2 leaf node they are connected with
        for j in range(0,vec_len):
            internal_node_5[j] = random.choice([transformed_x[preorder[2]][j],transformed_x[preorder[3]][j]])
            internal_node_6[j] = random.choice([transformed_x[preorder[0]][j],transformed_x[preorder[1]][j]])
    # Same thing, but now the smaller node number is on the left, thus connected with node 5
    else:
        edge_index = torch.tensor([[preorder[0],5],[5,preorder[0]],[preorder[1],5],[5,preorder[1]],
                                       [5,7],[7,5],[preorder[4],7],[7,preorder[4]],
                                       [6,7],[7,6],[preorder[2],6],[6,preorder[2]],
                                       [preorder[3],6],[6,preorder[3]]], dtype=torch.long)
        edge_attr = [dist_array[preorder[0]], dist_array[preorder[0]], 
                 dist_array[preorder[1]], dist_array[preorder[1]], 
                 dist_array[5], dist_array[5],
                 dist_array[preorder[4]], dist_array[preorder[4]],
                 dist_array[6], dist_array[6],
                 dist_array[preorder[2]], dist_array[preorder[2]],
                 dist_array[preorder[3]], dist_array[preorder[3]]]
        for j in range(0,vec_len):
            internal_node_5[j] = random.choice([transformed_x[preorder[0]][j],transformed_x[preorder[1]][j]])
            internal_node_6[j] = random.choice([transformed_x[preorder[2]][j],transformed_x[preorder[3]][j]])
    # Assign value for internal node 7, based on internal node 5&6, and leaf node 4
    for j in range(0,vec_len):
        internal_node_7[j] = random.choice([internal_node_5[j], internal_node_6[j], 
                                           transformed_x[preorder[4]][j]])
    # append all node feature into an array
    transformed_x.append(internal_node_5)
    transformed_x.append(internal_node_6)
    transformed_x.append(internal_node_7)
    concat_x = np.array( transformed_x )
    # create the node feature vector
    x = torch.tensor(concat_x, dtype=torch.float)
    # Now we create the graph object as Data
    data = Data(x=x, edge_index=edge_index.t().contiguous(), edge_attr = torch.FloatTensor(edge_attr))
    return data


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# select the correct adjacency matrix of one graph
def select_graph_original(graph_id, batch_targets, batch_index):
    # create a true/false mask to select the graph we want from the all dense adj matrix of the whole batch
    graph_mask = torch.eq(batch_index, graph_id)
    graph_targets = batch_targets[graph_mask][:, graph_mask]
    triu_indices = torch.triu_indices(graph_targets.shape[0], graph_targets.shape[0], offset=1)
    triu_mask = torch.squeeze(to_dense_adj(triu_indices)).bool()
    return graph_targets[triu_mask]


# select the predicted adjacency matrix of one graph
def select_graph_prediction(triu_logit, graph_size, start):
    graph_triu_logit = torch.squeeze(triu_logit[start:start + graph_size])
    return graph_triu_logit


def kl_loss(mu=None, logstd=None):
    MAX_LOGSTD = 10
    logstd =  logstd.clamp(max=MAX_LOGSTD)
    kl_div = -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))

    # Limit numeric errors
    kl_div = kl_div.clamp(max=1000)
    return kl_div


def loss_func(triu_logits, edge_index, mu, logvar, batch_index, kl_beta):
    # Convert target edge index to dense adjacency matrix
    # This is the actual adj matrix for the whole batch, converted directly from edge index
    batch_targets = torch.squeeze(to_dense_adj(edge_index))
    batch_recon_loss = []         # the loss for each graph in this batch
    batch_node_counter = 0        # track which node are we in the current batch
    for graph_id in torch.unique(batch_index):
        # get the actual upper triangular adjacency matrix for this graph
        graph_true_triu = select_graph_original(graph_id, batch_targets, batch_index)
        # get the prediction of adjac matrix for this graph
        graph_predict_triu =  select_graph_prediction(triu_logits, graph_true_triu.shape[0], batch_node_counter)
        # update node counter
        batch_node_counter = batch_node_counter + graph_true_triu.shape[0]
        # Calculate edge-weighted binary cross entropy
        weight = graph_true_triu.shape[0]/sum(graph_true_triu)
        bce = torch.nn.BCEWithLogitsLoss(pos_weight=weight).to(device)
        graph_recon_loss = bce(graph_predict_triu.view(-1), graph_true_triu.view(-1))
        batch_recon_loss.append(graph_recon_loss)
        
    # Take average of all losses
    num_graphs = torch.unique(batch_index).shape[0]
    batch_recon_loss = sum(batch_recon_loss) / num_graphs
    # KL Divergence
    kl_divergence = kl_loss(mu, logvar)
    return batch_recon_loss + kl_beta * kl_divergence, kl_divergence


def check_triu_graph_reconstruction(graph_predictions_triu, graph_targets_triu, num_nodes=None):
    # Apply sigmoid to get binary prediction values
    preds = (torch.sigmoid(graph_predictions_triu.view(-1)) > 0.5).int()
    # Reshape the ground truth
    labels = graph_targets_triu.view(-1)
     # Check if the predictions and the groundtruth match
    if labels.shape[0] == sum(torch.eq(preds, labels)):
        return True
    return False


# Get the accuracy for each epoch (both train and test)
def reconstruction_accuracy(triu_logits, edge_index, batch_index):
    batch_targets = torch.squeeze(to_dense_adj(edge_index))
    batch_targets_triu = []
    # Iterate over batch and collect each of the trius
    batch_node_counter = 0
    num_recon = 0
    for graph_id in torch.unique(batch_index):
        # Get triu parts for this graph
        graph_targets_triu = select_graph_original(graph_id, 
                                                batch_targets, 
                                                batch_index)
        graph_predictions_triu = select_graph_prediction(triu_logits, 
                                                        graph_targets_triu.shape[0], 
                                                        batch_node_counter)
        # Update counter to the index of the next graph
        batch_node_counter = batch_node_counter + graph_targets_triu.shape[0]
        # Check if graph is successfully reconstructed
        num_nodes = sum(torch.eq(batch_index, graph_id))
        recon = check_triu_graph_reconstruction(graph_predictions_triu, 
                                                graph_targets_triu, 
                                                num_nodes) 
        num_recon = num_recon + int(recon)
        batch_targets_triu.append(graph_targets_triu)
    
    batch_targets_triu = torch.cat(batch_targets_triu)
    triu_discrete = torch.squeeze(torch.tensor(torch.sigmoid(triu_logits) > 0.5, dtype=torch.int32))
    acc = torch.true_divide(torch.sum(batch_targets_triu==triu_discrete), batch_targets_triu.shape[0]) 
        
    return acc.detach().cpu().numpy(), num_recon    


def run_one_epoch(data_loader, type, epoch, kl_beta):
    all_losses = []
    all_accs = []
    all_kldivs = []
    
    reconstructed_tree = 0
    
    for i, batch in enumerate(tqdm(data_loader)):
        try:
            batch.to(device)
            optimizer.zero_grad()
            triu_logits, mu, logvar = model(batch.x.float(), batch.edge_index, batch.batch)
            loss, kl_loss = loss_func(triu_logits, batch.edge_index, mu, logvar, batch.batch, kl_beta)
            if type == "Train":
                loss.backward()  
                optimizer.step()  
            # Calculate metrics
            acc, num_recon = reconstruction_accuracy(triu_logits, batch.edge_index, batch.batch)
            reconstructed_tree = reconstructed_tree + num_recon
            if type == "Train":
                reconstructed_perc = reconstructed_tree/9000
            else:
                reconstructed_perc = reconstructed_tree/1000
            
            all_losses.append(loss.detach().cpu().numpy())
            all_accs.append(acc)
            all_kldivs.append(kl_loss.detach().cpu().numpy())
        except IndexError as error:
            print("Error: ", error)
    
    print(f"{type} epoch {epoch} loss: ", np.array(all_losses).mean())
    print(f"{type} epoch {epoch} accuracy: ", np.array(all_accs).mean())
    print(f"Reconstructed {reconstructed_perc}.")
    mlflow.log_metric(key=f"{type} Epoch Loss", value=float(np.array(all_losses).mean()), step=epoch)
    mlflow.log_metric(key=f"{type} Epoch Accuracy", value=float(np.array(all_accs).mean()), step=epoch)
    mlflow.log_metric(key=f"{type} Percentage Reconstructed", value=float(reconstructed_perc), step=epoch)
    mlflow.log_metric(key=f"{type} KL Divergence", value=float(np.array(all_kldivs).mean()), step=epoch)
    #mlflow.log_model(model, "model")
    


# Model

class GVAE(nn.Module):
    def __init__(self, feature_size, embedding_size, edge_dim):
        super(GVAE, self).__init__()
        self.latent_embedding_size = int(embedding_size/2)
        decoder_size = embedding_size*4
        
        # Encoder Layers
        # 3 layers with batch normalization
        self.conv1 = TransformerConv(feature_size,
                                    embedding_size*4,
                                    heads=4,
                                    concat=False,
                                    beta=True)
                                            #edge_dim=edge_dim)
        self.bn1 = BatchNorm(embedding_size*4)
        self.conv2 = TransformerConv(embedding_size*4,
                                    embedding_size*2,
                                    heads=4,
                                    concat=False,
                                    beta=True)
                                            #edge_dim=edge_dim)
        self.bn2 = BatchNorm(embedding_size*2)
        self.conv3 = TransformerConv(embedding_size*2,
                                    embedding_size,
                                    heads=4,
                                    concat=False,
                                    beta=True)
                                            #edge_dim=edge_dim)
        self.bn3 = BatchNorm(embedding_size)
        
        # Latent transform
        self.mu_transform = TransformerConv(embedding_size, 
                                            self.latent_embedding_size,
                                            heads=4,
                                            concat=False,
                                            beta=True)
                                            #edge_dim=edge_dim)
        self.logvar_transform = TransformerConv(embedding_size, 
                                            self.latent_embedding_size,
                                            heads=4,
                                            concat=False,
                                            beta=True)
                                            #edge_dim=edge_dim)
        
        # Decoder
        self.decoder_dense_1 = Linear(self.latent_embedding_size*2, decoder_size)
        self.decoder_bn_1 = BatchNorm1d(decoder_size)
        self.decoder_dense_2 = Linear(self.latent_embedding_size*2, decoder_size)
        self.decoder_bn_2 = BatchNorm1d(decoder_size)
        self.decoder_dense_3 = Linear(self.latent_embedding_size*2, decoder_size)
        self.decoder_bn_3 = BatchNorm1d(decoder_size)
        self.decoder_dense_4 = Linear(decoder_size, 1)
    
    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.bn1(x)
        x = self.conv2(x, edge_index).relu()
        x = self.bn2(x)
        x = self.conv3(x, edge_index).relu()
        x = self.bn3(x)
        # latent variable
        mu = self.mu_transform(x, edge_index)
        logvar = self.logvar_transform(x, edge_index)
        return mu, logvar
    
    def decode(self, z, batch_index):
        # z: the 5 latent vectors for all graph
        # batch
        inputs = []
        
        # for each graph in this batch
        for graph_id in torch.unique(batch_index):
            # Select the latent vectors for the graphs in this batch
            graph_mask = torch.eq(batch_index, graph_id)
            graph_z = z[graph_mask]
            # Get upper triangle adjacency matrix
            # the diagonal is not included (the node is always connected to itself)
            # should be something like this:
            # [0,0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,3,3,3,3,4,4,4,5,5,6],
            # [1,2,3,4,5,6,7,2,3,4,5,6,7,3,4,5,6,7,4,5,6,7,5,6,7,6,7,7]
            # Note that each column is a pair of possible connection. Now we have all possible connection
            edge_indices = torch.triu_indices(graph_z.shape[0], graph_z.shape[0], offset=1)
            # We want to put actual latent vectors in the place.
            # i.e, replace 0 in the previous array with latent vector for node 0.
            dim = self.latent_embedding_size
            # create the shape for source and target
            # each should be a 28*dim (length of the edge_indices[0]) array with same values for each row indicating the node number
            source_indices = torch.reshape(edge_indices[0].repeat_interleave(dim), (edge_indices.shape[1], dim))
            target_indices = torch.reshape(edge_indices[1].repeat_interleave(dim), (edge_indices.shape[1], dim))
            # Get the latent vectors
            # should fill the previous arrays with actual latent vectors such as 
            # [-1, 0, 3.2, ..., -2.1, 0] (latent vector for node 0)
            # [-1, 0, 3.2, ..., -2.1, 0]
            #        ......              (28 rows)
            # [0.3, 4.2, ..., 8, 2.5, 5] (latent vector for node 6)
            sources_latent = torch.gather(graph_z, 0, source_indices.to(device))
            target_latent = torch.gather(graph_z, 0, target_indices.to(device))
            # Should be 28(pairs of node) * (2*dim)
            graph_inputs = torch.cat([sources_latent, target_latent], axis=1)
            inputs.append(graph_inputs)
        
        # now we concat all graphs in this batch
        inputs = torch.cat(inputs)
        # feed into the decoding layers
        x = self.decoder_dense_1(inputs).relu()
        x = self.decoder_bn_1(x)
        x = self.decoder_dense_2(inputs).relu()
        x = self.decoder_bn_2(x)
        x = self.decoder_dense_3(inputs).relu()
        x = self.decoder_bn_3(x)
        edge_logits = self.decoder_dense_4(x)
        
        # We transform the logits later for probabilities
        return edge_logits
    
    # transform mu and logvar into the latent vectors
    def reparam(self, mu, logvar):
        if self.training:
            # transform logvar
            std = torch.exp(logvar)
            # generate same amount of random numbers from N(0, 1)
            eps = torch.randn_like(std)
            # get the sampled value
            return eps.mul(std).add_(mu)
        else:
            return mu
        
    def forward(self, x, edge_index, batch_index):
        # GNN layers
        mu, logvar = self.encode(x, edge_index)
        # latent vectors
        z = self.reparam(mu, logvar)
        # Decode layers
        logit = self.decode(z, batch_index)
        
        return logit, mu, logvar
            


# File inputs


# get name of the script
# nameScript = sys.argv[0].split('/')[-1]
nameScript = "gae_model.py"
# get json file name of the script
nameJson = "gae.json"
# nameJson = sys.argv[1]
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
embedSize = dataJson["embedSize"]        # Embedding size
nEpochs = dataJson["nEpochs"]            # Number of Epochs
batchSize = dataJson["batchSize"]        # batchSize
kl_beta = dataJson["klBeta"]

data_root = dataJson["dataRoot"]         # data folder
model_root = dataJson["modelRoot"]       # folder to save the data

label_files = dataJson["labelFile"]      # file with labels
sequence_files = dataJson["matFile"]     # file with sequences
tree_files = dataJson["treeFile"]        # file with tree structure

if "summaryFile" in dataJson:
    summary_file = dataJson["summaryFile"]
else :
    summary_file = "summary_file.txt"


print("------------------------------------------------------------------------") 
print("Loading Sequence Data in " + sequence_files, flush = True)
print("Loading Label Data in " + label_files, flush = True)
print("Loading Tree Data in " + tree_files, flush = True)

# we read the labels as list of strings
with open(data_root+label_files, 'r') as f:
    label_char = f.readlines()

# we read the sequence as a list of strings
with open(data_root+sequence_files, 'r') as f:
    seq_string = f.readlines()

with open(data_root+tree_files, 'r') as f:
    tree_newick = f.readlines()
    
n_samples = len(label_char)
seq_length = len(seq_string[0])-1
print("Number of samples:{}; Sequence length of each sample:{}"
        .format(n_samples, seq_length))
print("------------------------------------------------------------------------")


# Data pre-processing
# Read Sequence data and Newick tree format, return the all graph object with necessary info in the structure


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


# Create all graphs from raw dataset
# EXTREMELY SLOW
dataset = []  # empty dataset for all graphs
# loop through all samples
for i in range(n_samples):
    # Get the ete tree format
    tree = tree_newick[i][:-1]
    t = Tree(tree)
    # get node feature, COO adjacency matrix, and edge feature
    data = construct_single_graph(i, t)
    # Validate if number of node and edges match
    if (not data.validate(raise_on_error=True)):
        print("Error! Node number and edge set does not match!")
        break
    # Add the graph into the dataset
    dataset.append(data)


# Training


# Load data
print("Start the training process...")
train_dataset = dataset[:9000]
test_dataset = dataset[9000:]
train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=True)
print("------------------------------------------------------------------------")
print("Data loaded, loading model...")
# Load model
model = GVAE(seq_length, embedSize, 1)
print("Number of parameters in the model: ", count_parameters(model))
print("------------------------------------------------------------------------")
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

with mlflow.start_run() as run:
    for epoch in range(nEpochs): 
        model.train()
        run_one_epoch(train_loader, type="Train", epoch=epoch, kl_beta=kl_beta)
        if epoch % 5 == 0:
            print("Evluating testset...")
            model.eval()
            run_one_epoch(test_loader, type="Test", epoch=epoch, kl_beta=kl_beta)

