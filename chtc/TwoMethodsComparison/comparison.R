library(ape)
library(phangorn)
## read true trees
d = read.table("labels4728283-0.5-10.0-0.01.in", header=FALSE)
n = length(d$V1)
## labels from the simulating script:
quartets = c("((1,2),(3,4));", "((1,3),(2,4));", "((1,4),(2,3));")
## to which quartet the label corresponds to:
truetree = read.tree(text=quartets[d[n,]])

## read the NJ tree:
njtree = read.tree(file="nj-tree.txt")
## read the ML tree:
mltree = read.tree(file="T3.raxml.bestTree")
## read the BI tree
#bitree = read.nexus(file="test-mb.nexus.con.tre")

## Calculating the Robinson-Foulds distance with true tree:
njdist = RF.dist(truetree,njtree, rooted=FALSE)
print(njdist)
mldist = RF.dist(truetree,mltree, rooted=FALSE)
print(mldist)
#bidist = RF.dist(truetree,bitree, rooted=FALSE)
#print(bidist)
#save(njdist,file="out.txt")
#save(mldist,file="out.txt")
#save(bidist,file="out.txt")
args <- commandArgs(trailingOnly = F)
#print(args[7])
#print(args[8])
outname = paste("out",args[8],".csv",sep="")
#print(outname)
write.table(njdist,file=outname,append = T,sep=',',row.names="njdist",col.names=F)
write.table(mldist,file=outname,append = T,sep=',',row.names="mldist",col.names=F)
