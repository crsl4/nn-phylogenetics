library(ape)
library(phangorn)
library(adegenet)

d = read.table("dataset/labels12062021.in", header = FALSE)
d = head(d, 100)
n = length(d$V1)

quintet = c("((1,2),(3,4),5);", "((1,2),(3,5),4);", "((1,2),(4,5),3);",
            "((1,3),(2,4),5);", "((1,3),(2,5),4);", "((1,3),(4,5),2);",
            "((1,4),(2,3),5);", "((1,4),(2,5),3);", "((1,4),(3,5),2);",
            "((1,5),(2,3),4);", "((1,5),(2,4),3);", "((1,5),(3,4),2);", 
            "((2,3),(4,5),1);", "((2,4),(3,5),1);", "((2,5),(3,4),1);")

njcount = 0
mlcount = 0

for (i in 1:100) {
  truetree = read.tree(text=quintet[d[i,]])
  njname = paste("nj/nj-tree", i, ".txt", sep = "")
  mlname = paste("ml/T", i, ".raxml.bestTree", sep = "")
  ## read the NJ tree:
  njtree = read.tree(file=njname)
  ## read the ML tree:
  mltree = read.tree(file=mlname)
  
  njdist = RF.dist(truetree,njtree, rooted=FALSE)
  mldist = RF.dist(truetree,mltree, rooted=FALSE)
  
  if (njdist==0) {
    njcount = njcount + 1
  }
  
  if (mldist==0) {
    mlcount = mlcount + 1
  }
}

njacc = njcount/100
mlacc = mlcount/100


