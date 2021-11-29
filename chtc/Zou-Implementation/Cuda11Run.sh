#!/bin/bash
#n installation.
tar -xzf python38.tar.gz
#untar a set of packages
tar -xzf Cudapackages.tar.gz

# make sure the script will use your Python installation,
# and the working directory as its home location
export PATH=$PWD/python/bin:$PATH
export PYTHONPATH=$PWD/Cudapackages
export HOME=$PWD
#copy the compressed tar file from /staging into the working directory,
#and un-tar it to reveal large input file:
# run the python  script
python Cuda11sequenceZou.py $1 $2 $3 $4 
#before the script exists, remove the files from the working directory

