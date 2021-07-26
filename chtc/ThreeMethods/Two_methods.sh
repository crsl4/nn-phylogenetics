#!/bin/bash

#Step 1
#grab ten 4-taxon data from a randomly chosen file
sed -n "$((399997-$1*40)),$((400000-$1*40)) p"  sequences372783-0.1-40.0-1.0.in > test0.in
sed -n "$((399993-$1*40)),$((399996-$1*40)) p"  sequences372783-0.1-40.0-1.0.in > test1.in
sed -n "$((399989-$1*40)),$((399992-$1*40)) p"  sequences372783-0.1-40.0-1.0.in > test2.in
sed -n "$((399985-$1*40)),$((399988-$1*40)) p"  sequences372783-0.1-40.0-1.0.in > test3.in
sed -n "$((399981-$1*40)),$((399984-$1*40)) p"  sequences372783-0.1-40.0-1.0.in > test4.in
sed -n "$((399977-$1*40)),$((399980-$1*40)) p"  sequences372783-0.1-40.0-1.0.in > test5.in
sed -n "$((399973-$1*40)),$((399976-$1*40)) p"  sequences372783-0.1-40.0-1.0.in > test6.in
sed -n "$((399969-$1*40)),$((399972-$1*40)) p"  sequences372783-0.1-40.0-1.0.in > test7.in
sed -n "$((399965-$1*40)),$((399968-$1*40)) p"  sequences372783-0.1-40.0-1.0.in > test8.in
sed -n "$((399961-$1*40)),$((399964-$1*40)) p"  sequences372783-0.1-40.0-1.0.in > test9.in
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
raxml-ng/bin/raxml-ng --msa test5.fasta --model Dayhoff --prefix T8 --threads 2 --seed 616
sed -i'' -e $'s/>//g' T8.raxml.bestTree
cp T8.raxml.bestTree T35.raxml.bestTree
rm T8*
raxml-ng/bin/raxml-ng --msa test6.fasta --model Dayhoff --prefix T9 --threads 2 --seed 616
sed -i'' -e $'s/>//g' T9.raxml.bestTree
cp T9.raxml.bestTree T36.raxml.bestTree
rm T9*
raxml-ng/bin/raxml-ng --msa test7.fasta --model Dayhoff --prefix T4 --threads 2 --seed 616
sed -i'' -e $'s/>//g' T4.raxml.bestTree
cp T4.raxml.bestTree T37.raxml.bestTree
rm T4*
raxml-ng/bin/raxml-ng --msa test8.fasta --model Dayhoff --prefix T4 --threads 2 --seed 616
sed -i'' -e $'s/>//g' T4.raxml.bestTree
cp T4.raxml.bestTree T38.raxml.bestTree
rm T4*
raxml-ng/bin/raxml-ng --msa test9.fasta --model Dayhoff --prefix T4 --threads 2 --seed 616
sed -i'' -e $'s/>//g' T4.raxml.bestTree
cp T4.raxml.bestTree T39.raxml.bestTree
rm T4*

#Step 5
#Fitting bayesian inference (BI) with MrBayes
#get the nexus files in R
Rscript MrBayes.R
#add the mrbayes block in the shell
cat test0.nexus mb-block.txt > test0-mb.nexus
cat test1.nexus mb-block.txt > test1-mb.nexus
cat test2.nexus mb-block.txt > test2-mb.nexus
cat test3.nexus mb-block.txt > test3-mb.nexus
cat test4.nexus mb-block.txt > test4-mb.nexus
cat test5.nexus mb-block.txt > test5-mb.nexus
cat test6.nexus mb-block.txt > test6-mb.nexus
cat test7.nexus mb-block.txt > test7-mb.nexus
cat test8.nexus mb-block.txt > test8-mb.nexus
cat test9.nexus mb-block.txt > test9-mb.nexus
#Run the bayes script
mb test0-mb.nexus
mb test1-mb.nexus
mb test2-mb.nexus
mb test3-mb.nexus
mb test4-mb.nexus
mb test5-mb.nexus
mb test6-mb.nexus
mb test7-mb.nexus
mb test8-mb.nexus
mb test9-mb.nexus

#Step 6
#Comparing the estimated trees to the true tree
Rscript Comparison.R $1

#Step 7
#clean the directory
rm nj*
rm T3*
rm test*