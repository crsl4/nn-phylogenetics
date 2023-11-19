## Simulate quartets and aminoacid sequences as in Zou2019
## Using PAML, see notebook.md
## Note that this simulations is hard-coded for quartets only
## (cannot be extended to any size of tree)
## Claudia March 2020

using PhyloNetworks, Random, Distributions, Flux
include("functions-zou2019.jl")

## Input
rseed = 5005652
L = 1550 ## sequence length
ratealpha = 0.1580288 ## see notebook.md
model = 3 ##for control file PAML
modeldatfile = "dayhoff.dat"
blL = 0.02 ##lower bound unif for BL
blU = 1.02 ##upper bound unif for BL
nrep = 8000

Random.seed!(rseed)
seeds = sample(1:5555555555,nrep)
makeOdd!(seeds) ## we need odd seed for PAML

labels = zeros(nrep)
matrices = zeros(L)

for i in 1:nrep
    println("=====================================================")
    @show i
    tree,ind = sampleRootedMetricQuartet(blL,blU, seeds[i])
    namectl = string("rep-",i,".dat")
    createCtlFile(namectl, tree, seeds[i], L, ratealpha, model, modeldatfile)
    run(`./evolver 7 MCaa.dat`)
    run(`cp mc.paml rep-$i.paml`)
    mat = convert2onehot("mc.paml",L)
    labels[i] = ind
    global matrices
    matrices = hcat(matrices,mat)
end
matrices = matrices'
matrices = matrices[2:end,:] ## 80*nrep by L: each replicate has a 80xL matrix, all stacked

using HDF5
h5write("labels.h5","labels",labels)
h5write("matrices.h5","matrices",matrices)


