library(ape)
library(phangorn)
library(adegenet)

## Reading fasta file
aa = read.aa(file="test0.fasta", format="fasta")

## Estimating evolutionary distances using same model as in simulations
D = dist.ml(aa, model="Dayhoff") 

## Estimating tree
tree = nj(D)

## Saving estimated tree to text file
write.tree(tree,file="nj-tree0.txt")

#4 more times
aa = read.aa(file="test1.fasta", format="fasta")
D = dist.ml(aa, model="Dayhoff")
tree = nj(D)
write.tree(tree,file="nj-tree1.txt")

aa = read.aa(file="test2.fasta", format="fasta")
D = dist.ml(aa, model="Dayhoff")
tree = nj(D)
write.tree(tree,file="nj-tree2.txt")

aa = read.aa(file="test3.fasta", format="fasta")
D = dist.ml(aa, model="Dayhoff")
tree = nj(D)
write.tree(tree,file="nj-tree3.txt")

aa = read.aa(file="test4.fasta", format="fasta")
D = dist.ml(aa, model="Dayhoff")
tree = nj(D)
write.tree(tree,file="nj-tree4.txt")

aa = read.aa(file="test5.fasta", format="fasta")
D = dist.ml(aa, model="Dayhoff")
tree = nj(D)
write.tree(tree,file="nj-tree5.txt")

aa = read.aa(file="test6.fasta", format="fasta")
D = dist.ml(aa, model="Dayhoff")
tree = nj(D)
write.tree(tree,file="nj-tree6.txt")

aa = read.aa(file="test7.fasta", format="fasta")
D = dist.ml(aa, model="Dayhoff")
tree = nj(D)
write.tree(tree,file="nj-tree7.txt")

aa = read.aa(file="test8.fasta", format="fasta")
D = dist.ml(aa, model="Dayhoff")
tree = nj(D)
write.tree(tree,file="nj-tree8.txt")

aa = read.aa(file="test9.fasta", format="fasta")
D = dist.ml(aa, model="Dayhoff")
tree = nj(D)
write.tree(tree,file="nj-tree9.txt")