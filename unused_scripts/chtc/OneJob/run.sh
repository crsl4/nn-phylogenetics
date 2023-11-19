#!/bin/bash

#untar Python installation.
tar -xzf python38.tar.gz
#untar a set of packages
tar -xzf packages.tar.gz

# make sure the script will use your Python installation,
# and the working directory as its home location
export PATH=$PWD/python/bin:$PATH
export PYTHONPATH=$PWD/packages
export HOME=$PWD
#copy the compressed tar file from /staging into the working directory,
#and un-tar it to reveal large input file:
cp /staging/syang382/lba1.tar ./
tar -xvf lba1.tar
# run the python  script
python network_lba.py $1 $2 $3 $4 
#before the script exists, remove the files from the working directory
rm lba1.tar labels-lba-1.h5 matrices-lba-1.h5
