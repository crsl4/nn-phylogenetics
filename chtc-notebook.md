# Description of the CHTC Runs

This .md file will describe runs of 500 tests of three methods Neighbor-joining, Maximum Likelihood and Bayesian Inference in CHTC

## Input Files

Input file: labels/sequences##-a-b-c.in (ex. labels/sequences68163238-0.1-2.0-1.0.in) .[here](https://drive.google.com/drive/folders/1LUNzOxSrL7QJcaKd7mI4uokUEyFHvsYc)

## CHTC Scripts [here](https://github.com/crsl4/nn-phylogenetics/tree/master/chtc/ThreeMethods)
CHTC dag scripts: 
- twoMethods.dag (to control the flow to merge and calculate accuracy of all 500 jobs)
CHTC submit scripts: 
- Two_methods.sub (The submit file to run 10 jobs of three methods comparison)
- getAccuracy.sub (The submit file to run the accuracy calculation)
CHTC batch scripts:
- Two_methods.sh (The batch script to have one run of 10 jobs of three methods comparison)
- merge.sh (The batch script to remove all output .csv files from 50 runs into one directory and merge them into one output.csv file)
- getAccuracy.sh (The bactch script to calculate the accuracy from output.csv file)
CHTC other scripts:
- Convert_all.jl (The Julia file to convert sample data to fasta file)
- Comparison.R (The R file to compare estimated trees to the true trees)
- MrBayes.R (The R file to create nexus files for MrBayes)
- NeighborJoining.R (The R file to fit maximum parsimony and neighbor-joining)
- accuracy.R (The R file to calculate the accuracy of the 500 jobs of all three methods)
Command to run the job in CHTC: condor_submit_dag twoMethods.dag
Comments on running the submit script: 
- The input files need to be in the same folder as the submit script
- output, log and error directories should be created in the same folder of all scripts
- twoMethods.dag.* should be removed before next run

## Output Files
Output files returned from CHTC: 
- accuracyresult.csv: the .csv file including the accuracy of all three methods. Row 1 for Neighbor-joining, row 2 for Maximum Likelihood and row 3 for Bayesian Inference
- intercsv directory: include {0-49}.csv files, which have all comparison results of 50 runs of 10 jobs for three methods, and output.csv which is the merged result
- twoMethods.dag.condor.sub: File for submitting the DAG file to HTCondor 
- twoMethods.dag.dagman.out: Log of DAGMan debugging messages  
- twoMethods.dag.lib.out: Log of HTCondor library output   
- twoMethods.dag.lib.err: Log of HTCondor library error messages   
- twoMethods.dag.dagman.log: Log of the life of condor_dagman itself  
- Output of one run of all three methods could be returned by changing the Two_methods.sh file
Output files saved in google drive: [here](https://drive.google.com/drive/folders/17l7u3PFQ3o87GJ_6ApjfZWfimUIHbHlF)
