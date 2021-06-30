library(ape)
library(phangorn)
library(adegenet)

## Reading fasta file
aa = read.aa(file="test.fasta", format="fasta")

## Estimating evolutionary distances using same model as in simulations
D = dist.ml(aa, model="Dayhoff") 

## Estimating tree
tree = nj(D)

## Saving estimated tree to text file
write.tree(tree,file="nj-tree.txt")
