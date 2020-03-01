## to make seeds odd because PAML needs odd seeds
function makeOdd!(seeds::Vector{Int64})
    mod = seeds .% 2
    seeds[mod.==0]=seeds[mod.==0] .+ 1
end

## create the control file for PAML
## to simulate aminoacid sequences on a given tree
## note: tree is assumed to be a quartet
## name: name of the output control file (note that the control file
##       needs to be called MCaa.dat, but we will save also as name
##       to avoid overwriting
## tree: tree (quartet, rooted, with bl)
## s: random seed
## L: sequence length
## alpha: rate for gamma
## model: see notebook.md for model options for PAML
## modelfile: we chose one file from the ones provided by PAML with the S matrix
function createCtlFile(name::String, tree::String, s::Integer, L::Integer, alpha::Float64,
                       model::Integer, modelfile::String)
    str = """0        * 0: paml format (mc.paml); 1:paup format (mc.nex)
$s       * random number seed (odd number)

4 $L 1   * <# seqs>  <# sites>  <# replicates>

-1         * <tree length, use -1 if tree below has absolute branch lengths>

$tree

$alpha 1        * <alpha; see notes below>  <#categories for discrete gamma>
$model $modelfile * <model> [aa substitution rate file, need only if model=2 or 3]

0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05
0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05

 A R N D C Q E G H I
 L K M F P S T W Y V

// end of file

=============================================================================
Notes for using the option in evolver to simulate amino acid sequences.
Change values of parameters, but do not delete them.  It is o.k. to add
empty lines, but do not break down the same line into two or more lines.

  model = 0 (poisson), 1 (proportional), 2 (empirical), 3 (empirical_F)
  Use 0 for alpha to have the same rate for all sites.
  Use 0 for <#categories for discrete gamma> to use the continuous gamma
  <aa substitution rate file> can be dayhoff.dat, jones.dat, and so on.
  <aa frequencies> have to be in the right order, as indicated.
=================!! Check screen output carefully!! =====================
"""
    outfile = open(name,"w")
    write(outfile, str)
    close(outfile)

    ## we need to name the file MCaa.dat
    outfile2 = open("MCaa.dat","w")
    write(outfile2, str)
    close(outfile2)
end


### simulate a quartet with branch lengths uniformly dist
### U(l,u)
## ss: seed
### Rooted chosen randomly
function sampleRootedMetricQuartet(l::Number,u::Number, ss::Integer)
    Random.seed!(ss)
    quartets = ["((1,2),(3,4));", "((1,3),(2,4));", "((1,4),(2,3));"] ##only thinking of unrooted
    ind = sample(1:3,1)[1]
    q = quartets[ind]
    quartet = readTopology(q)
    ## choose root randomly (or leave as is=balanced tree)
    r = rand(Uniform(0,1),1)[1]
    if(r<0.2)
        rootatnode!(quartet,"1")
    elseif(r<0.4)
        rootatnode!(quartet,"2")
    elseif(r<0.6)
        rootatnode!(quartet,"3")
    elseif(r<0.8)
        rootatnode!(quartet,"4")
    end
    ## setting branch lenghts
    for e in quartet.edge
        setLength!(e,rand(Uniform(l,u),1)[1])
    end
    return writeTopology(quartet),ind
end


## converts the file of sequences into one hot encoded tensor
## e.g. if file has 4 sequences of aminoacid (20 letters) of length L
## this function should return a tensor of 4x20xL where each letter is represented
## by its one-hot code: l1->000...01
## BUT in Zou2019 they then vectorize the tensor into matrix (e.g. 80xL),
## so we will return the matrix directly
## we need the length (L) to initialize the matrix
## note: it assumes phylip format so that the first row is "n L"
function convert2onehot(name::String, L::Integer)
    mc = readlines(name)
    mc = mc[mc .!= ""]
    mat = zeros(L,1) ##output matrix
    ## assumes phylip format so that first row is "n L"
    ## we do i=2 first to have the same alll vector
    i=2
    mm = split(mc[i])
    @show mm[1]
    seq = join(mm[2:end])
    alll = unique(seq)  ##e.g. the 20 aminoacids
    m = zeros(1,length(alll)) ## output matrix
    for s in seq
        m = [m;Flux.onehot(s,alll)']
    end
    m = m[2:end,:]
    mat = hcat(mat,m)

    for i in 3:length(mc)
        mm = split(mc[i])
        @show mm[1]
        seq = join(mm[2:end])
        m = zeros(1,length(alll)) ## output matrix
        for s in seq
            m = [m;Flux.onehot(s,alll)']
        end
        m = m[2:end,:]
        mat = hcat(mat,m)
    end
    mat = mat[:,2:end]
    return mat
end
