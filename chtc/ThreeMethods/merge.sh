#!/bin/bash

#merge  outputs from each individual methods and inputs into a directory
cat  *.csv >> output.csv
mkdir intercsv
mv *.csv intercsv/

