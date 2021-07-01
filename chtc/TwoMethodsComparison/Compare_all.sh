#!/bin/bash
#set up the environment
export PATH
. /etc/profile.d/modules.sh
#try to solve the shared library problem
#sudo apt-get install libreadline6:i386
#Step 1
#grab one 4-taxon dataset from a randomly chosen file
wc -l sequences4728283-0.5-10.0-0.01.in  ## 400000
sed -n "$2,$1 p"  sequences4728283-0.5-10.0-0.01.in > test.in

#Step 2
#untar the julia installation.
tar -xzf julia-1.6.0-linux-x86_64.tar.gz

#make sure the script will use julia installation
export PATH=$PWD/julia-1.6.0/bin:$PATH


#Run the Julia script
julia convert.jl

#Step 3
#Fitting maximum parsimony(MP) and neighbor-joining(NJ) in R
Rscript Nj.R

#Step 5 switch sequence try
Rscript mb.R
tar -xzf bayes.tar.gz
cat test.nexus mb-block.txt > test-mb.nexus
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64 bayes/bin/mb test-mb.nexus

#Step 4
#Fitting maximum likelihood(ML) with RAxML
#untar the raxml  installation.
tar -xzf raxml.tar.gz
#set up the environment
#export PATH
#. /etc/profile.d/modules.sh
module load GCC/8.3.0

#Run the raxml-ng  script
raxml-ng/bin/raxml-ng --msa test.fasta --model Dayhoff --prefix T3 --threads 2 --seed 616

sed -i'' -e $'s/>//g' T3.raxml.bestTree

#try to remove raxml-ng to avoid shared library
#rm -r raxml-ng
#try to remove the shared library
#rm libreadline.so.6
#Skip Step 5 for now
#Fitting bayesian inference (BI) with MrBayes
#get the nexus files in R
#Rscript mb.R

#untar the bi installation.
#tar -xzf bayes.tar.gz

#cat test.nexus mb-block.txt > test-mb.nexus
#set up the environment
#export PATH
#. /etc/profile.d/modules.sh
#Run the bayes script
#LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64 bayes/bin/mb test-mb.nexus

#Step 6
#Comparing the estimated trees to the true tree
Rscript comparison.R $3
