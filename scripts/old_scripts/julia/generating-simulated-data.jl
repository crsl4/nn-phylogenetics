## Input:
## -- DNA sequences in FASTA file
## Output:
## -- List of simulated {(tree, branch lengths, Q matrix)} along with the likelihood score
##   Note that (tree, bl) are represented as adjacency matrix
## Details:
## For i=1,...,nrep, this script will:
## 1) Simulate NJ bootstrap tree T_i from the DNA sequence file
## 2) Simulate branch lengths t_i according to chosen distribution centered on DNA distances
## 3) Simulate Q_i matrix based on base frequencies and rate transitions from DNA sequences
## 4) Calculate likelihood score L(T_i,t_i,Q_i)
## Claudia October 2019

using PhyloNetworks
include("functions.jl")

## Input
rootname = "../data/4taxa-cats_numbers"
nrep = 100 ## number of samples in parameter space


## Reading input DNA file
infile = string(rootname,".phy")
convertPHYLIP2FASTA(infile)
fastafile = string(rootname,".fasta")
df,w = readfastatodna(fastafile)


for i in 1:nrep
    tree = sampleTree(df)
    sampleBL!(tree)
    qmatrix = sampleQ(df)
    adjmatrix = convertTree2Matrix(tree)
    lik = calculateLik(tree,qmatrix)
    ## save into array
end


## roadblock:
## - sampleTree needs dist matrix to get NJ tree. By bootstrapping, we do not change the dist matrix, unless we use Q
## (we don't want to do that in this simple case). We could sample subsets of sites as boostrap, not much variability
## - sampleQ needs estimates of base frequencies, rate transitions
## - it seems that we do not have functions to calculate the likelihood
