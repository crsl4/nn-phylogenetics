##Introduction
This file will  explain how to run the tests of the two methods maximumlikelihood and neighbor-joining on chtc and where each file is

##Requirement
To run the test, you should have accesee to the CHTC server, mkdir log, output and error directories and
having the dockerhub account to use the docker image

##Files Description and Its Location
Compare_all.sub/sh: the submition and batch script files for running two methods tests multiple times in chtc in github nn-phylogenetics/chtc/TwoMethodsComparison
merge.sub/sh: the submition and btach script files for merge those result csv files into one csv file in chtc in github nn-phylogenetics/chtc/TwoMethodsComparison
getAccuracy.sub/sh: the submittion and batch script files for calculating the accuracy of each methods storing in a csv file in chtc in github nn-phylogenetics/chtc/TwoMethodsComparison
Nj.R: the R file of the neighbor-joinging method in github nn-phylogenetics/chtc/TwoMethodsComparison
comparison.R: the R file of getting the result of two methods in github nn-phylogenetics/chtc/TwoMethodsComparison
accuracy.R: the R file of calculating the accuracy of two methods in github nn-phylogenetics/chtc/TwoMethodsComparison
convert.jl: the julia file of converting the sample data to Fasta file in github nn-phylogenetics/chtc/TwoMethodsComparison
mb-block.txt: the text block file to run the MrBayes in github nn-phylogenetics/chtc/TwoMethodsComparison
twoMethods.dag: the dag file to run those three submittion files in sequence in github nn-phylogenetics/chtc/TwoMethodsComparison
julia-1.6.0-linux-x86_64.tar.gz: the required julia packages compressed file in google drive phylo-nn/chtc/TwoMethodsComparison
packages.tar.gz: the required R plyr pacakages compressed file in google dirve phylo-nn/chtc/TwoMethodsComparison
raxml.tar.gz:the required raXml packages compressed file in google drive phylo-nn/chtc files/TwoMethodsComparison
adegenet.zip: the docker image build file in google drive phylo-nn/chtc files/TwoMethodsComparison
sequences#.in: the  simulated datasets to run the test in google drive phylo-nn/chtc files/TwoMethodsComparison
lbbels#.in:the simulated label datasets to run the test in google drive phylo-nn/TwoMethodsComparison
##Tow-Methods-Test
###1 Set-up for docker image
having a dockerhub account and download the adegenet.zip.Follow the instruction of building a docker image with the dockerfile and push the image to your own repository
and change the second line of Compare_all.sub to your own image's name
###2 Set-up for chtc running
to create log,output, and error directories and have Compare_all.sub/sh, merge.sub/sh, getAccuracy.sub/sh, accuracy.R, comparison.R, convert.jl, mb-block.txt, Nj.R,
twoMethods.dag, packages.tar.gz,julia-1.6.0-linux-x86_64.tar.gz, raxml.tar.gz, sequences#.in and labels#.in in the same directory
###3 Run the tests
to run the tests, you should call "condor_submit_dag twoMethods.dag" in the same directory
###4 get the results
the result of the accuracy of those two methods would be contained in the accuracyresult.csv file after the job is finished, it took around 18 minutes to run 5 tests parallelly
