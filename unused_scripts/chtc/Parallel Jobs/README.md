##Introduction
This file will  explain how to run the parallel tests of the neural network model on chtc and where each file

##Requirement
To run the test, you should have accesee to the CHTC server, mkdir log, output and error directories and
having staging directory access to store large files

##Files Description and Its Location
runparallel.sub: the submition file for running parallel gpu tests in chtc in github nn-phylogenetics/chtc/Parallel Jobs
run.sh: the shell script to execute the running of a test in github nn-phylogentics/chtc/OneJob
network_lba.py: the python file of the neural network model in github nn-phylogenetics/chtc/Parallel Jobs
modules.py: the python file of the modules in the NN model in github nn-phylogenetics/chtc/Parallel Jobs
lba_11gpu.json: the json file of the parameters to run the gpu-test in github nn-phylogenetics/chtc/Parallel Jobs
lba_12gpu.json: another json file of the parameters to parallelly run the gpu-test in github nn-pylogenetics/chtc/Parallel Jobs
(the number of such json files could be as many as possible)
input_files.txt: the txt file includes those names of the json files to run parallely in chtc in github nn-pylogenetics/chtc/Parallel Jobs
packages.tar.gz: the required python pacakages compressed file in google dirve phylo-nn/chtc files
matrices-lba-#.h5: the "easy" simulated datasets to run the test in google drive phylo-nn
lbbels-lba-#.h5:the "easy" simulated label datasets to run the test in google drive phylo-nn

##Gpu-test
You should have log,output and error directories, runparallel.sub, run.sh, network_lba.py, modules.py, packages.tar.gz
and lba_11gpu.json, lba_12gpu.json, input_files.txt  files in the current directory. Storing the compressed .tar file of matrics-lba-#.h5 and labels-lba-#.h5
into the staging directory of your own and changing the run.sh's corresponding staging and datasets information
Run those parallel  Gpu-tests by calling condor_submit runparallel.sub at this directory
It should take about 2 hours to run the whole test with 100 epoches, and output the corresponding output names file specified in json files to the current directory,
some run-time information in the output directory
~

