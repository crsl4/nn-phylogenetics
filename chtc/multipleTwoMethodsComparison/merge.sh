#!/bin/bash

#merge  outputs from each individual methods and inputs into a directory
mkdir intercsv
mv *.csv intercsv/
cat  intercsv/*.csv >> output.csv
