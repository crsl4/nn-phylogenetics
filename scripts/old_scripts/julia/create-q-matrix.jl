## Input:
## -- .out files from mcmc bistro with list of sampled trees with branch lengths
## Output:
## -- List of adjacency matrices
## Claudia October 2019

using PhyloNetworks, CSV, DataFrames, LinearAlgebra, HDF5 ##, DelimitedFiles
include("functions.jl")

dat1 = CSV.read("../data/after---0-249.out", delim=' ',ignorerepeated=true)
dat2 = CSV.read("../data/after---250-499.out", delim=' ',ignorerepeated=true)
dat3 = CSV.read("../data/after---500-749.out", delim=' ',ignorerepeated=true)
dat4 = CSV.read("../data/after---750-999.out", delim=' ',ignorerepeated=true)
allt = vcat(dat1,dat2,dat3,dat4)

mvec = Array{Float64,2}[]
for i in 1:size(allt,1)
    p=[allt[i,:pi1],allt[i,:pi2],allt[i,:pi3],allt[i,:pi4]]
    r=[allt[i,:s1],allt[i,:s2],allt[i,:s3],allt[i,:s4],allt[i,:s5],allt[i,:s6]]
    m = makeQ(r,p)
    push!(mvec,m)
end

## Reshaping matrices into vector: 1000X16
dat = hcat(reshape.(mvec,:)...)'
dat = dat[1:end,1:end]

h5open("../data/q-matrices.h5","w") do file
    @write file dat
end



## Not used anymore: saved as txt file
##outfile = "../data/q-matrices.txt"
##writedlm(outfile,dat,',')

## Saving loglik too:
outfile = "../data/loglik-vector.h5"
loglik=allt[:logl]

h5open(outfile,"w") do file
    @write file loglik
end

##outfile = "../data/loglik-vector.txt"
##CSV.write(outfile,DataFrame(loglik=allt[:logl]), writeheader=false)
