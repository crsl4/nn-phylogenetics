library(ape)
library(phangorn)
library(adegenet)

for (i in 1:100) {
  filename = paste("fasta/test", i, ".fasta", sep = "")
  aa = read.aa(file=filename, format = "fasta")
  D = dist.ml(aa, model="Dayhoff") 
  tree = nj(D)
  savename = paste("nj/nj-tree", i, ".txt", sep = "")
  write.tree(tree,file=savename)
}
