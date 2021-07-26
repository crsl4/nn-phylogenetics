library(ape)
library(phangorn)
library(adegenet)

## Reading fasta file
aa = read.aa(file="test0.fasta", format="fasta")

## Write as nexus file
write.nexus.data(aa,file="test0.nexus",format="protein")

aa = read.aa(file="test1.fasta", format="fasta")
write.nexus.data(aa,file="test1.nexus",format="protein")

aa = read.aa(file="test2.fasta", format="fasta")
write.nexus.data(aa,file="test2.nexus",format="protein")

aa = read.aa(file="test3.fasta", format="fasta")
write.nexus.data(aa,file="test3.nexus",format="protein")

aa = read.aa(file="test4.fasta", format="fasta")
write.nexus.data(aa,file="test4.nexus",format="protein")

aa = read.aa(file="test5.fasta", format="fasta")
write.nexus.data(aa,file="test5.nexus",format="protein")

aa = read.aa(file="test6.fasta", format="fasta")
write.nexus.data(aa,file="test6.nexus",format="protein")

aa = read.aa(file="test7.fasta", format="fasta")
write.nexus.data(aa,file="test7.nexus",format="protein")

aa = read.aa(file="test8.fasta", format="fasta")
write.nexus.data(aa,file="test8.nexus",format="protein")

aa = read.aa(file="test9.fasta", format="fasta")
write.nexus.data(aa,file="test9.nexus",format="protein")
