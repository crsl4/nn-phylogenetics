## Simulate quintets and aminoacid sequences (as in Zou2019,
## but for n=5)
## Using PAML, see notebook.md
## Note that this simulations is hard-coded for quintets only
## (cannot be extended to any size of tree)
## Claudia April 2020

using PhyloNetworks, Random, Distributions, Combinatorics
include("functions-zou2019.jl")

## Input
rseed = 03011058
L = 1550 ## sequence length
ratealpha = 0.1580288 ## see notebook.md
model = 3 ##for control file PAML
modeldatfile = "dayhoff.dat"
blL = 0.02 ##lower bound unif for BL
blU = 1.02 ##upper bound unif for BL
nrep = 5000

if length(ARGS) > 0
    rseed = parse(Int,ARGS[1])
    nrep = parse(Int,ARGS[2])
end

Random.seed!(rseed)
seeds = sample(1:5555555555,nrep)
makeOdd!(seeds) ## we need odd seed for PAML

outfilelab = string("labels",rseed,".in")
outfileseq = string("sequences",rseed,".in")
outfiletrees = string("trees",rseed,".in")
f = open(outfilelab,"w")
f2 = open(outfiletrees,"w")

for i in 1:nrep
    println("=====================================================")
    @show i
    app = i == 1 ? false : true
    tree,ind = sampleRootedMetricQuintet(blL,blU, seeds[i])
    
    global f
    write(f,string(ind))
    write(f,"\n")

    global f2
    write(f2,tree)
    write(f2,"\n")
    
    namectl = string("rep-",i,".dat")
    createCtlFile(namectl, tree, seeds[i], L, ratealpha, model, modeldatfile, n=5)
    run(`./evolver 7 MCaa.dat`)

    writeSequence2File("mc.paml",L,outfileseq,append=app)
    run(`rm mc.paml`)
    run(`rm rep-$i.dat`)
end



