##Introduction
This file will  explain how to run the tests of the neural network model on chtc and where each file

##Requirement
To run the test, you should have accesee to the CHTC server, mkdir log, output and error directories and 
having staging directory access to store large files

##Files Description and Its Location
gpurun.sub: the submition file for running a gpu test in chtc in github nn-phylogenetics/chtc/OneJob
run.sub: the submition file for running a non-gpu test in chtc in github nn-phylogenetics/chtc/OneJob
run.sh: the shell script to execute the running of a test in github nn-phylogentics/chtc/OneJob
network_lba.py: the python file of the neural network model in github nn-phylogenetics/chtc/OneJob
modules.py: the python file of the modules in the NN model in github nn-phylogenetics/chtc/OneJob
lba_1.json: the json file of the parameters to run the non-gpu-test in github nn-phylogenetics/chtc/OneJob
lba_1gpu.json: the json file of the parameters to run the gpu-test in github nn-phylogenetics/chtc/OneJob
packages.tar.gz: the required python pacakages compressed file in google dirve phylo-nn/chtc files
matrices-lba-#.h5: the "easy" simulated datasets to run the test in google drive phylo-nn
lbbels-lba-#.h5:the "easy" simulated label datasets to run the test in google drive phylo-nn

##Non-Gpu-test
You should have log,output and error directories, run.sub, run.sh, network_lba.py, modules.py, packages.tar.gz
and lba_1.json files in the current directory. Storing the compressed .tar file of matrics-lba-#.h5 and labels-lba-#.h5
into the staging directory of your own and changing the run.sh's corresponding staging and datasets information
Run the Non-Gpu-test by calling condor_submit run.sub at this directory
It should take about 13 hours to run the whole test with 100 epoches, and output the summary_file.txt to the current directory,
some run-time information in the output directory

##Gpu-test
You should have log,output and error directories, gpurun.sub, run.sh, network_lba.py, modules.py, packages.tar.gz
and lba_1gpu.json files in the current directory. Storing the compressed .tar file of matrics-lba-#.h5 and labels-lba-#.h5
into the staging directory of your own and changing the run.sh's corresponding staging and datasets information
Run the Gpu-test by calling condor_submit gpurun.sub at this directory
It should take about 2 hours to run the whole test with 100 epoches, and output the summary_file.txt to the current directory,
some run-time information in the output directory
