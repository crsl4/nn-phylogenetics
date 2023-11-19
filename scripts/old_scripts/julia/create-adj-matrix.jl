## Input:
## -- .tre files from mcmc bistro with list of sampled trees with branch lengths
## Output:
## -- List of adjacency matrices
## Claudia October 2019

using PhyloNetworks, HDF5, DataFrames ##, DelimitedFiles
include("functions.jl")

t1 = readMultiTopology("../data/after---0-249.treeBL")
t2 = readMultiTopology("../data/after---250-499.treeBL")
t3 = readMultiTopology("../data/after---500-749.treeBL")
t4 = readMultiTopology("../data/after---750-999.treeBL")
allt = vcat(t1,t2,t3,t4)

mvec = Array{Float64,2}[]
for t in allt
    m = convertTree2AdjMatrix(t)
    push!(mvec,m)
end

## Reshaping matrices into vector: 1000X100
dat = hcat(reshape.(mvec,:)...)'
dat = dat[1:end,1:end]

h5open("../data/adj-matrices.h5","w") do file
    @write file dat
end

## Not used anymore: saved as txt file
##outfile = "../data/adj-matrices.txt"
##writedlm(outfile,dat,',')

