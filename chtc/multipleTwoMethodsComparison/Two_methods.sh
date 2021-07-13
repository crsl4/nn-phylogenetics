#!/bin/bash

#Step 1
#grab ten 4-taxon data from a randomly chosen file
wc -l sequences4728283-0.5-10.0-0.01.in  ## 400000
sed -n "$((399997-$1*20)),$((400000-$1*20)) p"  sequences4728283-0.5-10.0-0.01.in > test0.in
sed -n "$((399993-$1*20)),$((399996-$1*20)) p"  sequences4728283-0.5-10.0-0.01.in > test1.in
sed -n "$((399989-$1*20)),$((399992-$1*20)) p"  sequences4728283-0.5-10.0-0.01.in > test2.in
sed -n "$((399985-$1*20)),$((399988-$1*20)) p"  sequences4728283-0.5-10.0-0.01.in > test3.in
sed -n "$((399981-$1*20)),$((399984-$1*20)) p"  sequences4728283-0.5-10.0-0.01.in > test4.in

#Step 2
#untar the julia installation.
tar -xzf julia-1.6.0-linux-x86_64.tar.gz

#make sure the script will use julia installation
export PATH=$PWD/julia-1.6.0/bin:$PATH

#Run the Julia script
julia Convert_all.jl

#Step 3
#Fitting maximum parsimony(MP) and neighbor-joining(NJ) in R
Rscript NeighborJoining.R

#Step 4
#Fitting maximum likelihood(ML) with RAxML
#untar the raxml  installation.
tar -xzf raxml.tar.gz

#Run the raxml-ng  script
raxml-ng/bin/raxml-ng --msa test0.fasta --model Dayhoff --prefix T3 --threads 2 --seed 616
sed -i'' -e $'s/>//g' T3.raxml.bestTree
cp T3.raxml.bestTree T30.raxml.bestTree
raxml-ng/bin/raxml-ng --msa test1.fasta --model Dayhoff --prefix T4 --threads 2 --seed 616
sed -i'' -e $'s/>//g' T4.raxml.bestTree
cp T4.raxml.bestTree T31.raxml.bestTree
rm T4*
raxml-ng/bin/raxml-ng --msa test2.fasta --model Dayhoff --prefix T5 --threads 2 --seed 616
sed -i'' -e $'s/>//g' T5.raxml.bestTree
cp T5.raxml.bestTree T32.raxml.bestTree
rm T5*
raxml-ng/bin/raxml-ng --msa test3.fasta --model Dayhoff --prefix T6 --threads 2 --seed 616
sed -i'' -e $'s/>//g' T6.raxml.bestTree
cp T6.raxml.bestTree T33.raxml.bestTree
rm T6*
raxml-ng/bin/raxml-ng --msa test4.fasta --model Dayhoff --prefix T7 --threads 2 --seed 616
sed -i'' -e $'s/>//g' T7.raxml.bestTree
cp T7.raxml.bestTree T34.raxml.bestTree
rm T7*
#Step 5
#Comparing the estimated trees to the true tree
Rscript Comparison.R $1
