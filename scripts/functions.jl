## convert PHYLIP to FASTA
function convertPHYLIP2FASTA(file::String)
    lines = readlines(file);
    lines = lines[3:end]
    lines = string.(">",lines)
    rootname = split(file,"phy")[1]
    io1 = open(string(rootname,"fasta"), "w")
    for l in lines
        l2 = split(l,' ')
        write(io1,string(l2[1],"\n",l2[end]))
        write(io1,"\n")
    end
    close(io1)
end


## calculate distance from taxon i, j in df with sequences
## df: comes from readfastatodna()
## note: this is simple distance of differences
## in nucleotides
function calculateDistance(i::Int,j::Int,df::DataFrame)
    count = 0
    for k in 2:size(df,2)
        count += (df[i,k] != df[j,k])
    end
    return count
end


## convert HybridNetwork object to (weighted) Adjacency matrix
## note: not traversing tree in any smart way
function convertTree2AdjMatrix(tree::HybridNetwork; weighted=true::Bool)
    mat = fill(0.0,length(tree.node),length(tree.node))
    for i in 1:length(tree.node)
        for j in 1:length(tree.node)
            n1 = tree.node[i]
            n2 = tree.node[j]
            try
                e = PhyloNetworks.getConnectingEdge(n1,n2)
                mat[i,j] = weighted ? e.length : 1.0
            catch
                mat[i,j] = 0.0
            end
        end
    end
    return mat
end



## make Q matrix from base frequences (p) and rates (r)
## based on branch-length_lik.R from bistro
## p=[allt[1,:pi1],allt[1,:pi2],allt[1,:pi3],allt[1,:pi4]]
## r=[allt[1,:s1],allt[1,:s2],allt[1,:s3],allt[1,:s4],allt[1,:s5],allt[1,:s6]]
function makeQ(r,p)
    n = length(p)
    Q = fill(0.0,n,n)
    rows = reshape(repeat(collect(1:n),n),n,n)
    cols = reshape(repeat(collect(1:n),n),n,n)'
    Q[rows .> cols] = r
    Q = Q + Q'
    Q = Q * Diagonal(p)
    Q = Q - Diagonal(reshape(sum(Q,dims=1),:))
    return Q
end

