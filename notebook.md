# Neural networks for phylogenetic inference

- Claudia Solis-Lemus
- Leonardo Zepeda-Nunez

## Basic simulation with 4 taxa

- File from bistro project with cats DNA sequences: `4taxa-cats_numbers.phy` copied into `data` folder
- Simulation script `generating-simulated-data.jl`
- Many code roadblocks! Need to see the C code


## Using artiodactyl 6-taxon data

We had run MCMC on a dataset with 6 taxa. Results in files (in `data`):
- `after---*.out`: loglik, base frequencies, rate transitions
- `after---?.treeBL`: sampled trees with branch lengths

```shell
$ cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/data/
$ wc -l after*.treeBL
     250 after---0-249.treeBL
     250 after---250-499.treeBL
     250 after---500-749.treeBL
     250 after---750-999.treeBL
    1000 total
```

- Script `create-adj-matrix.jl` will read the .tre files and create adjencency matrix. It creates the file `adj-matrices.txt` with 1000 rows, one row per vectorized weighted adjacenty matrix (6 taxa => 10 nodes => 10x10 matrix => 100-dim vector)
- Script `create-q-matrix.jl` will read the .par files and create the Q matrix. It creates the file `q-matrices.txt` with 1000 rows, one row per Q matrix (4x4 matrix => 16-dim vector). This script also creates the `loglik-vector.txt` with 1000 rows, one per loglik value.

# Subproblem 2: based on Zou2019

We want to:
1. Replicate the work in Zou2019. They simulate sequences under CTMC and quartet gene trees, and then train a neural network model to classify sequences based on the quartet gene tree they came from. Maybe Leo's student can help us replicate these results
2. Improve the neural network model in Zou2019 (too naive)
3. Simulate training data in a more meaningful manner: instead of classifying sequences based on quartet gene tree they came from, we can classify sequences based on the quartet species tree they came from. This is more meaningful because ultimately biologists want to estimate species tree (not just gene tree). We will first simulate quartet gene trees from quartet species tree under the coalescent model, and then simulate sequences from the quartet gene tree under CTMC model. We will train a neural network model to classify sequences based on the quartet species tree they come from


## Simulating data to replicate Zou2019

- Training data: 100,000 quartets with varying branch lengths (not explicit how)
- Testing on 2000 quartets generated in the same manner as the training
- Further testing: 6 datasets of 1000 20-taxon trees with branch lengths on the intervals: [0.02, 0.2), [0.2, 0.4), [0.4, 0.6), [0.6, 0.8), [0.8, 1.0), and [1.0, 2.0)
- Further testing on 5 1000-tree datasets with sequence length ranging: [100, 200) to [3,000, 10,000) amino acids
- Raw input data: 4 aligned aminoacid sequences of length L => Matrix 4xL; this matrix is converted to tensor due to one-hot encoding: 4x20xL (because there are 20 aminoacids). The tensor is later vectorized as 80xL matrix
- Parameters for simulation: number of taxa M (later pruned to 4), branch lengths B, number of aminoacid sites N_{aa}, exchangeability matrix S, shape parameter \alpha of the gamma distribution of the relative rate r (on all sites), probability p_h with which a branch is heterogeneous, proportion of sites subject to rate shuffling f, number of profile swap operations for each site (Table S2)
- From Table S2 (first row):
    - M~Uniform(5,105)
    - (internal) B~Uniform(0.02,1.02)
    - (external) B~Uniform(0.02,1.02)
    - N_{aa}~Uniform(100,3000)
    - S random
    - \alpha~Uniform(0.05,1)
    - p_h=0.9
    - f~Uniform(0,1)
    - n~Uniform(10,20)


There code to simulate data is not easy to follow/understand, so I could simulate data with SeqGen as I always do. In the Zou2019 paper, they reference PAML (from Ziheng Yang), which also simulates sequences, see [here](http://abacus.gene.ucl.ac.uk/software/pamlDOC.pdf):
```
evolver. This program can be used to simulate sequences under nucleotide, codon and amino acid substitution models. It also has some other options such as generating random trees, and calculating the partition distances (Robinson and Foulds 1981) between trees. 
```

1. Download PAML following instructions [here](http://abacus.gene.ucl.ac.uk/software/pamlDOC.pdf), and downloading `paml4.8a.macosx.tgz` from [here](http://abacus.gene.ucl.ac.uk/software/paml.html)

2. In `paml4.8/bin` there is the `evolver` executable, which can be run as an executable directly:
```
$ cd Dropbox/software/PAML4/paml4.8/bin/
$ ./evolver
EVOLVER in paml version 4.8a, August 2014
Results for options 1-4 & 8 go into evolver.out

	(1) Get random UNROOTED trees?
	(2) Get random ROOTED trees?
	(3) List all UNROOTED trees?
	(4) List all ROOTED trees?
	(5) Simulate nucleotide data sets (use MCbase.dat)?
	(6) Simulate codon data sets      (use MCcodon.dat)?
	(7) Simulate amino acid data sets (use MCaa.dat)?
	(8) Calculate identical bi-partitions between trees?
	(9) Calculate clade support values (evolver 9 treefile mastertreefile <pick1tree>)?
	(11) Label clades?
	(0) Quit?
```
Or by choosing a specfic control file: `./evolver 7 MCaa.dat`

3. We need to write the control file. I copy here the example `MCaa.dat`:
```
 0        * 0: paml format (mc.paml); 1:paup format (mc.nex)
13147       * random number seed (odd number)

5 10000 5   * <# seqs>  <# sites>  <# replicates>

-1         * <tree length, use -1 if tree below has absolute branch lengths>

(((Human:0.06135, Chimpanzee:0.07636):0.03287, Gorilla:0.08197):0.11219, Orangutan:0.28339, Gibbon:0.42389);

.5 8        * <alpha; see notes below>  <#categories for discrete gamma>
2 mtmam.dat * <model> [aa substitution rate file, need only if model=2 or 3]

0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 
0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 

 A R N D C Q E G H I
 L K M F P S T W Y V

// end of file

=============================================================================
Notes for using the option in evolver to simulate amino acid sequences. 
Change values of parameters, but do not delete them.  It is o.k. to add 
empty lines, but do not break down the same line into two or more lines.

  model = 0 (poisson), 1 (proportional), 2 (empirical), 3 (empirical_F)
  Use 0 for alpha to have the same rate for all sites.
  Use 0 for <#categories for discrete gamma> to use the continuous gamma
  <aa substitution rate file> can be dayhoff.dat, jones.dat, and so on.
  <aa frequencies> have to be in the right order, as indicated.
=================!! Check screen output carefully!! =====================
```
Note from documentation: "If you use –1 for the tree length, the program will use the branch lengths given in the tree without the re-scaling.", but if you give a number, the program will re-scale the branch lengths so that the sum of bls is equal to tree length.

**Important** All files need to be in the same path of the executable!

Example: `./evolver 7 MCaa.dat` produces:
- `mc.paml` with the sequences in PHYLIP format
- `ancestral.txt` with the simulated ancestral sequences
- `sites.txt` with the rates for each site
- `evolver.out` which is empty

So, we have all the pieces, now we only need a pipeline for our simulations.

### Simulations pipeline
Fixed parameters:
- global random seed = 03011058
- L = 1550 (average of Uniform(100,3000))
- 1 class partition, with \alpha~Uniform(0.05,1)
- model=3 (empirical) with randomly chosen dat file for rates
- nrep = 100,000
```r
set.seed(03011058)
runif(1,0.05,1) ##alpha
0.1580288
floor(runif(1,1,18)) ##dat file (18 total)
4 ##dayhoff.dat
```

For rep i=1,...nrep:
1. Choose a random tree with 4 taxa
2. Choose random branch lengths: Uniform(0.02,1.02) as in Zou2019
3. Create the control file `rep-i.dat`:
```
 0        * 0: paml format (mc.paml); 1:paup format (mc.nex)
<seed>       * random number seed (odd number)

4 1550 1   * <# seqs>  <# sites>  <# replicates>

-1         * <tree length, use -1 if tree below has absolute branch lengths>

<tree with bl>

0.1580288 1        * <alpha; see notes below>  <#categories for discrete gamma>
3 dayhoff.dat * <model> [aa substitution rate file, need only if model=2 or 3]
```
4. Run `./evolver 7 rep-i.dat` (all files in same path as executable)
5. Rename necessary files and move into folders (to avoid overwriting): `mv mc.paml mc-i.paml`
6. Read `mc-i.paml` as matrix, and convert to 4x20x1550 tensor, and then vectorize as 80x1550 matrix
7. Output two lists one for the "labels" (which quartet) and the input matrices

I created the folder `simulations-zou2019` to put all simulated data there. Copied inside the executable (`evolver`) and the model dat file.
Julia script file: `simulate-zou2019.jl` and `functions-zou2019.jl`

### Simulations on batches:

#### 20,000 replicates on 4 threads (5000 each):
Different folders (because files are overwritten and have same names for PAML): `simulations-zou2019-?`, each is running nrep=5000 (I tried 25000 but the computer crashed twice). Process started 3/2 7:49pm, finished at 10:52pm.
Process re-started after changing the root: 5/8 10pm on laptop, but process died in the night after ~4000.
Re-started on desktop 5/9 830am, finished 12pm
- Each replicate produces a label (tree) and input matrix 80xL
- For nrep replicates, we get two files:
  - `labels.h5` with a vector of dimension nrep
  - `matrices.h5` with a matrix 80*nrep by L, that has all matrices stacked

  I had to check that the sequences were simulated in the correct order in the *.dat files. The S1,S2,S3,S4 in the paml file correspond to the order in the tree in the dat file. Files look ok!


We summarize the files:
```shell
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019

## First thread:
cd simulations-zou2019
tar -czvf simulations-zou2019-1.tar.gz rep-*
rm rep-*
mv labels.h5 labels-1.h5
mv matrices.h5 matrices-1.h5
##cp simulate-zou2019.jl simulate-zou2019-1.jl ## to keep the script ran (not used in the re-run because we had it)
## move to results folder:
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019-1.jl ../results

## Second thread
cd simulations-zou2019-2
tar -czvf simulations-zou2019-2.tar.gz rep-*
rm rep-*
mv labels.h5 labels-2.h5
mv matrices.h5 matrices-2.h5
##cp simulate-zou2019.jl simulate-zou2019-2.jl ## to keep the script ran
## move to results folder:
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019-2.jl ../results

## Third thread
cd simulations-zou2019-3
tar -czvf simulations-zou2019-3.tar.gz rep-*
rm rep-*
mv labels.h5 labels-3.h5
mv matrices.h5 matrices-3.h5
##cp simulate-zou2019.jl simulate-zou2019-3.jl ## to keep the script ran
## move to results folder:
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019-3.jl ../results

## Fourth thread
cd simulations-zou2019-4
tar -czvf simulations-zou2019-4.tar.gz rep-*
rm rep-*
mv labels.h5 labels-4.h5
mv matrices.h5 matrices-4.h5
##cp simulate-zou2019.jl simulate-zou2019-4.jl ## to keep the script ran
## move to results folder:
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019-4.jl ../results
```

#### 80,000 replicates on 10 cores (8000 each)
Different folders (because files are overwritten and have same names for PAML): `simulations-zou2019-?`, each is running nrep=8000.
Process started 3/6 5pm, finished at 3/7 2am

We rerun in Mac desktop. We start only with 5-9 (to make sure we don't run out of memory).
Process 5-9 started 5/17 1pm, finished 9:30pm
Process 10-14 started 5/17 10pm, finished 5/18 7am

We summarize the files:
```shell
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019

## 5th thread:
cd simulations-zou2019-5
mkdir tmp 
mv rep-1* tmp
mv rep-2* tmp
mv rep-* tmp
tar -czvf simulations-zou2019-5.tar.gz tmp
rm -r tmp
mv labels.h5 labels-5.h5
mv matrices.h5 matrices-5.h5
##cp simulate-zou2019.jl simulate-zou2019-5.jl ## to keep the script ran
## move to results folder:
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019-5.jl ../results

## 6th thread:
cd simulations-zou2019-6
mkdir tmp 
mv rep-1* tmp
mv rep-2* tmp
mv rep-* tmp
tar -czvf simulations-zou2019-6.tar.gz tmp
rm -r tmp
mv labels.h5 labels-6.h5
mv matrices.h5 matrices-6.h5
##cp simulate-zou2019.jl simulate-zou2019-6.jl ## to keep the script ran
## move to results folder:
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019-6.jl ../results

## 7th thread:
cd simulations-zou2019-7
mkdir tmp 
mv rep-1* tmp
mv rep-2* tmp
mv rep-* tmp
tar -czvf simulations-zou2019-7.tar.gz tmp
rm -r tmp
mv labels.h5 labels-7.h5
mv matrices.h5 matrices-7.h5
##cp simulate-zou2019.jl simulate-zou2019-7.jl ## to keep the script ran
## move to results folder:
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019-7.jl ../results

## 8th thread:
cd simulations-zou2019-8
mkdir tmp 
mv rep-1* tmp
mv rep-2* tmp
mv rep-* tmp
tar -czvf simulations-zou2019-8.tar.gz tmp
rm -r tmp
mv labels.h5 labels-8.h5
mv matrices.h5 matrices-8.h5
##cp simulate-zou2019.jl simulate-zou2019-8.jl ## to keep the script ran
## move to results folder:
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019-8.jl ../results


## 9th thread:
cd simulations-zou2019-9
mkdir tmp 
mv rep-1* tmp
mv rep-2* tmp
mv rep-* tmp
tar -czvf simulations-zou2019-9.tar.gz tmp
rm -r tmp
mv labels.h5 labels-9.h5
mv matrices.h5 matrices-9.h5
##cp simulate-zou2019.jl simulate-zou2019-9.jl ## to keep the script ran
## move to results folder:
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019-9.jl ../results

## 10th thread:
cd simulations-zou2019-10
mkdir tmp 
mv rep-1* tmp
mv rep-2* tmp
mv rep-* tmp
tar -czvf simulations-zou2019-10.tar.gz tmp
rm -r tmp
mv labels.h5 labels-10.h5
mv matrices.h5 matrices-10.h5
##cp simulate-zou2019.jl simulate-zou2019-10.jl ## to keep the script ran
## move to results folder:
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019-10.jl ../results

## 11th thread:
cd simulations-zou2019-11
mkdir tmp 
mv rep-1* tmp
mv rep-2* tmp
mv rep-* tmp
tar -czvf simulations-zou2019-11.tar.gz tmp
rm -r tmp
mv labels.h5 labels-11.h5
mv matrices.h5 matrices-11.h5
##cp simulate-zou2019.jl simulate-zou2019-11.jl ## to keep the script ran
## move to results folder:
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019-11.jl ../results

## 12th thread:
cd simulations-zou2019-12
mkdir tmp 
mv rep-1* tmp
mv rep-2* tmp
mv rep-* tmp
tar -czvf simulations-zou2019-12.tar.gz tmp
rm -r tmp
mv labels.h5 labels-12.h5
mv matrices.h5 matrices-12.h5
##cp simulate-zou2019.jl simulate-zou2019-12.jl ## to keep the script ran
## move to results folder:
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019-12.jl ../results

## 13th thread:
cd simulations-zou2019-13
mkdir tmp 
mv rep-1* tmp
mv rep-2* tmp
mv rep-* tmp
tar -czvf simulations-zou2019-13.tar.gz tmp
rm -r tmp
mv labels.h5 labels-13.h5
mv matrices.h5 matrices-13.h5
##cp simulate-zou2019.jl simulate-zou2019-13.jl ## to keep the script ran
## move to results folder:
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019-13.jl ../results

## 14th thread:
cd simulations-zou2019-14
mkdir tmp 
mv rep-1* tmp
mv rep-2* tmp
mv rep-* tmp
tar -czvf simulations-zou2019-14.tar.gz tmp
rm -r tmp
mv labels.h5 labels-14.h5
mv matrices.h5 matrices-14.h5
##cp simulate-zou2019.jl simulate-zou2019-14.jl ## to keep the script ran
## move to results folder:
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019-14.jl ../results
```

We have all scripts and results in `simulations-zou2019-results`, so we will remove the folders used to run things in parallel:
```shell
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/
rm -rf simulations-zou2019-*
```

Then, in `simulations-zou2019`, there are two folders:
- `scripts`: with julia scripts and needed executables
- `results`:
  - `labels-i.h5` (i=1,...,14): n-dimensional vector with labels for n replicates; files i=1,2,3,4 have 5000 replicates each (20,000 total) and files i=5,...,14 have 8000 replicates each (80,000 total) => 20k+80k=100k as in Zou2019
  - `matrices-i.h5` (i=1,...,14): 80x1550 input matrix per replicate; 
  files i=1,2,3,4 have 5000 replicates and matrices are stacked on top of each other 
  => (80 * 5000)x1550 matrix; 
  files i=5,...,14 have 8000 replicates each 
  => (80 * 8000)x1550 matrix
  - `simulate-zou2019-i.jl` (i=1,...,14): julia script with random seeds to simulate batch i
  - `simulations-zou2019-1.tar.gz` (i=1,...,14): tar intermediate files per replicate like protein sequences, and paml control file

I will put the labels and matrices files in a shared drive to share with Leo.

Deleting the h5 files locally because they are heavy, and they are in google drive now.

## Using Zou2019 script to simulate trees n=5 (quintets)

Created file `simulate-zou2019-n5.jl` with simulation pipeline to simulate trees of size 5 (quintets). This script is not tested yet.
Careful: this and previous scripts are not exploiting the fact that we could generate data from all possible roots. 

# Understanding Zou2019 permutations
From the main text:
- We generated random trees with more than four taxa and simulated amino acid sequences of varying lengths according to the trees
- After the generation of each tree and associated sequence data, we pruned the tree so that only four taxa remain, hence creating a quartet tree sample ready for training, validation, or testing of the residual network predictor. 
- To ensure that the training process involved diverse and challenging learning materials, we pruned a proportion of trees to four randomly chosen taxa (normal trees), and the other trees to four taxa with high LBA susceptibility
- Training consisted of multiple iterative epochs, based on a total training pool of 100,000 quartets containing 85% normal trees and 15% LBA trees **note:** 100,000 quartets, each with different (24) permutations as explained below

From the "Materials and Methods" section:
- The raw input data, as in conventional phylogenetic inference software, are four aligned amino acid sequences of length L (denoted as taxon0, taxon1, taxon2, and taxon3, hence dimension 4 x L). This is then one-hot-encoded, expanding each amino acid position into a dimension with twenty 0/1 codes indicating which amino acid is in this position. The 4 x 20 x L tensor is transformed to an 80 x L matrix and fed into the residual network
- The output of the network includes three numbers representing the likelihood that taxon0 is a sister of taxon1, taxon2, and taxon3, respectively
- During the training process, the four taxa in each quartet data set were permutated to create 4!=24 different orders, and each serves as an independent training sample, to ensure that the order of taxa in the data set does not influence the phylogenetic inference
- Sequences on a tree were simulated from more ancient to more recent nodes, starting at the root. 

Process:
1. Simulate large tree, then prune to quartet
2. Simulate sequences on quartet
3. Permutate all 4 taxa in the quartet to get 24 permutations for that quartet


In [data.py](https://gitlab.com/ztzou/phydl/-/blob/master/evosimz/data.py) has the function shuffle on line 225. We want to understand why (if?) all 24 permutations make sense for a given tree:
```python
class _QuartetMixin:
    ## this gives us the list of all 24 permutations (see code below)
    _ORDERS = numpy.asarray(list(itertools.permutations(range(4))))

def _shuffle(cls, tree, random_order=False):
        ## here we create a vector of size 24 with repeated tree
        ## note that cls._ORDERS.shape=(24,4)
        trees = [tree] * cls._ORDERS.shape[0] 
        leaves = tree.get_leaves()
        if random_order:
            random.shuffle(leaves)
        ## leaf.sequence is an array of length L, leaves are 4:(tx0,tx1,tx2,tx3)
        sequences = numpy.asarray([leaf.sequence for leaf in leaves])
        ## to understand view('S1') see below
        sequences = sequences.view('S1').reshape(len(leaves), -1)
        sequences = sequences[cls._ORDERS, :]
        ## the following command change the order of the leaves to match the permutations:
        leaf_list = [[leaves[i] for i in order] for order in cls._ORDERS]
        # print(len(trees), sequences.shape, cls._ORDERS.shape, len(leaf_list), sep='\n')
        # 24, (24, 4, 869), (24, 4), 24
        return trees, sequences, cls._ORDERS, leaf_list
```

Trying to understand the commands in python:
```python
$ python
Python 2.7.16 (default, Oct 16 2019, 00:34:56) 
[GCC 4.2.1 Compatible Apple LLVM 10.0.1 (clang-1001.0.37.14)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import bisect
>>> import itertools
>>> import pickle
>>> import random
>>> import numpy
>>> numpy.asarray(list(itertools.permutations(range(4))))
array([[0, 1, 2, 3],
       [0, 1, 3, 2],
       [0, 2, 1, 3],
       [0, 2, 3, 1],
       [0, 3, 1, 2],
       [0, 3, 2, 1],
       [1, 0, 2, 3],
       [1, 0, 3, 2],
       [1, 2, 0, 3],
       [1, 2, 3, 0],
       [1, 3, 0, 2],
       [1, 3, 2, 0],
       [2, 0, 1, 3],
       [2, 0, 3, 1],
       [2, 1, 0, 3],
       [2, 1, 3, 0],
       [2, 3, 0, 1],
       [2, 3, 1, 0],
       [3, 0, 1, 2],
       [3, 0, 2, 1],
       [3, 1, 0, 2],
       [3, 1, 2, 0],
       [3, 2, 0, 1],
       [3, 2, 1, 0]])
>>> strarray = numpy.array([[b"123456"], [b"654321"]])
>>> strarray
array([['123456'],
       ['654321']], 
      dtype='|S6')
>>> strarray.view('S1')
array([['1', '2', '3', '4', '5', '6'],
       ['6', '5', '4', '3', '2', '1']], 
      dtype='|S1')
>>> strarray.view('S1').reshape(2,-1)
array([['1', '2', '3', '4', '5', '6'],
       ['6', '5', '4', '3', '2', '1']], 
      dtype='|S1')
>>> strarray.view('S1').reshape(3,-1)
array([['1', '2', '3', '4'],
       ['5', '6', '6', '5'],
       ['4', '3', '2', '1']], 
      dtype='|S1')
>>> strarray[0]
array(['123456'], 
      dtype='|S6')
>>> strarray[[0,1]]
array([['123456'],
       ['654321']], 
      dtype='|S6')
>>> strarray[[1,0]]
array([['654321'],
       ['123456']], 
      dtype='|S6')
```

Conclusion after talking to Erin: A student in her group presented this paper in lab meeting. He has been trying to use the trained network on a dataset (and failing miserably). It seems that the permutations are on the rows of the matrix only, not on the labels. So all our discussion on symmetries is totally new and something we can exploit (yay!). If you have a matrix 4xL with the sequences, they permute the rows (all 24 permutations), but they keep the labels. This is only to prevent the row order from mattering.


# Notes after studying the code

- The `shuffle` function takes a tree as input (one quartet), which repeats as a vector of dimension 24: `trees`
- The function also takes the sequence simulated on this tree as input, say:
```
0....
1....
2....
3....
```
- The function then creates all 4!=24 permutations of the 4 indices (`cls.ORDER`):
```
array([[0, 1, 2, 3],
       [0, 1, 3, 2],
       [0, 2, 1, 3],
       [0, 2, 3, 1],
       [0, 3, 1, 2],
       [0, 3, 2, 1],
       [1, 0, 2, 3],
       [1, 0, 3, 2],
       [1, 2, 0, 3],
       [1, 2, 3, 0],
       [1, 3, 0, 2],
       [1, 3, 2, 0],
       [2, 0, 1, 3],
       [2, 0, 3, 1],
       [2, 1, 0, 3],
       [2, 1, 3, 0],
       [2, 3, 0, 1],
       [2, 3, 1, 0],
       [3, 0, 1, 2],
       [3, 0, 2, 1],
       [3, 1, 0, 2],
       [3, 1, 2, 0],
       [3, 2, 0, 1],
       [3, 2, 1, 0]])
```
- Then, they adjust the `leaf_list` to the specific order. That is, if the `leaf_list` was `(tx1,tx2,tx3,tx4)`, they will get a 24-dim vector will all the permutations on `cls.ORDER`:
```
array([[tx1, tx2, tx3, tx4],
       [tx1, tx2, tx4, tx3],
       [tx1, tx3, tx2, tx4],
.
.
.
```

After `shuffle`, they call the function `_generate_class_label`, which for every 4-taxon array, change the quartet class (response label) if it was changed. This is done so that we do not need to keep the labels.
That is, the quartet 1 is 01|23. [0, 1, 2, 3] corresponds to this same quartet, so as [1, 0, 2, 3], but this one is not: [3, 1, 2, 0], this corresponds to 02|13.

### Permutation map

Now, we want to do the map of permutation to quartet class for us.
Note that `seqgen` puts the sequences in the order that we expect. That is, for the following tree:
```
((1: 0.636349, 4: 0.324226): 0.549904, (2: 0.389060, 3: 0.153263): 0.433517);
```
`seqgen` converts it to:
```
((S1: 0.636349, S4: 0.324226): 0.549904, (S2: 0.389060, S3: 0.153263): 0.433517);
```
and puts the sequences in order:
```
S1
S2
S3
S4
```
Thus, our quartet specification (12|34, 13|24, 14|23) matches the quartet specification in Zou2019 (quartet1: taxon1 and taxon2 are sisters => S1,S2 sisters for us).

#### Quartet 1 (12|34)
Indices: 1->0, 2->1, 3->2, 4->3
```
[0, 1, 2, 3] => 12|34
[0, 1, 3, 2] => 12|34
[0, 2, 1, 3] => 13|24
[0, 2, 3, 1] => 14|23
[0, 3, 1, 2] => 13|24
[0, 3, 2, 1] => 14|23
[1, 0, 2, 3] => 12|34
[1, 0, 3, 2] => 12|34
[1, 2, 0, 3] => 13|24
[1, 2, 3, 0] => 14|23
[1, 3, 0, 2] => 13|24
[1, 3, 2, 0] => 14|23
[2, 0, 1, 3] => 14|23
[2, 0, 3, 1] => 13|24
[2, 1, 0, 3] => 14|23
[2, 1, 3, 0] => 13|24
[2, 3, 0, 1] => 12|34
[2, 3, 1, 0] => 12|34
[3, 0, 1, 2] => 14|23
[3, 0, 2, 1] => 13|24
[3, 1, 0, 2] => 14|23
[3, 1, 2, 0] => 13|24
[3, 2, 0, 1] => 12|34
[3, 2, 1, 0] => 12|34
```


#### Quartet 2 (13|24)
Indices: 1->0, 3->1, 2->2, 4->3
```
[0, 1, 2, 3] => 13|24
[0, 1, 3, 2] => 14|23
[0, 2, 1, 3] => 12|34
[0, 2, 3, 1] => 12|34
[0, 3, 1, 2] => 14|23
[0, 3, 2, 1] => 13|24
[1, 0, 2, 3] => 14|23
[1, 0, 3, 2] => 13|24
[1, 2, 0, 3] => 14|23
[1, 2, 3, 0] => 13|24
[1, 3, 0, 2] => 12|34
[1, 3, 2, 0] => 12|34
[2, 0, 1, 3] => 12|34
[2, 0, 3, 1] => 12|34
[2, 1, 0, 3] => 13|24
[2, 1, 3, 0] => 14|23
[2, 3, 0, 1] => 13|24
[2, 3, 1, 0] => 14|23
[3, 0, 1, 2] => 13|24
[3, 0, 2, 1] => 14|23
[3, 1, 0, 2] => 12|34
[3, 1, 2, 0] => 12|34
[3, 2, 0, 1] => 14|23
[3, 2, 1, 0] => 13|24
```


#### Quartet 3 (14|23)
Indices: 1->0, 4->1, 2->2, 3->3
```
[0, 1, 2, 3] => 14|23
[0, 1, 3, 2] => 13|24
[0, 2, 1, 3] => 14|23
[0, 2, 3, 1] => 13|24
[0, 3, 1, 2] => 12|24
[0, 3, 2, 1] => 12|34
[1, 0, 2, 3] => 13|24
[1, 0, 3, 2] => 14|23
[1, 2, 0, 3] => 12|34
[1, 2, 3, 0] => 12|34
[1, 3, 0, 2] => 14|23
[1, 3, 2, 0] => 13|24
[2, 0, 1, 3] => 13|24
[2, 0, 3, 1] => 14|23
[2, 1, 0, 3] => 12|34
[2, 1, 3, 0] => 12|34
[2, 3, 0, 1] => 14|23
[2, 3, 1, 0] => 13|24
[3, 0, 1, 2] => 12|34
[3, 0, 2, 1] => 12|34
[3, 1, 0, 2] => 13|24
[3, 1, 2, 0] => 14|23
[3, 2, 0, 1] => 13|24
[3, 2, 1, 0] => 14|23
```


**NOTE** It will not be as straight-forward to implement this permutation strategy due to how PAML works.
In our simulating pipeline, PAML is already permuting the rows of the sequence.
That is, for `rep-1.dat`, PAML is simulating sequences from the tree:
```
(4:0.5499040314743673,(1:0.3242263809077284,(2:0.15326301295458997,3:0.4335172886379941):0.38905991077599955):0.3366654232508077);
```
The order of the taxa will be order read, so S1=4, S2=1, S3=2, S4=3.
So, we might need to simulate the data again to force the order of taxa.


# LBA simulations

We repeat the quartet simulations, but now with long branch attraction branches (Figure 3a Zou2020): "two short external branches have lengths b ranging from 0.1 to 1.0, the two long branches have lengths a ranging from 2b to 40b, and the internal branch has a length c ranging from 0.01b to b".
Added the option `lba = true` to the simulations script.

We will only do 10,000 replicates now.

#### 10,000 replicates on 2 threads (5000 each):
Different folders (because files are overwritten and have same names for PAML): `simulations-zou2019-?`, each is running nrep=5000 (I tried 25000 but the computer crashed twice). We copy the `scripts` folder as two folder: `simulations-zou2019-lba` and `simulations-zou2019-lba-2`
```shell
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-zou2019-lba
julia simulate-zou2019.jl

cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-zou2019-lba-2
julia simulate-zou2019.jl
```
Process started 6/17 330pm, ~finish 830pm

- Each replicate produces a label (tree) and input matrix 80xL
- For nrep replicates, we get two files:
  - `labels.h5` with a vector of dimension nrep
  - `matrices.h5` with a matrix 80*nrep by L, that has all matrices stacked

We summarize the files:
```shell
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019

## First thread:
cd simulations-zou2019-lba
tar -czvf simulations-zou2019-lba-1.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-1.h5
mv matrices.h5 matrices-lba-1.h5
##cp simulate-zou2019.jl simulate-zou2019-1.jl ## to keep the script ran (not used in the re-run because we had it)
## move to results folder:
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-1.jl
mv simulate-zou2019-lba-1.jl ../results

## Second thread
cd simulations-zou2019-lba-2
tar -czvf simulations-zou2019-lba-2.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-2.h5
mv matrices.h5 matrices-lba-2.h5
##cp simulate-zou2019.jl simulate-zou2019-2.jl ## to keep the script ran
## move to results folder:
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-2.jl
mv simulate-zou2019-lba-2.jl ../results
```

It turns out that these simulations are wrong, because in Zou2020 they do no sample b,a,c from a distribution. Instead, they simply fix them.

So, I will delete the lba files, and change the scripts.

In total, there are the following 120 cases:
- b=0.1, 0.2, 0.5, 1 (4)
- a= 2b, 5b, 10b, 20b, 40b (5)
- c=0.01b, 0.02b, 0.05b, 0.1b, 0.2b, 0.5b, b (6)

We will only do the following 27 cases:
- b=0.1, 0.5, 1
- a=2b, 10b, 40b
- c=0.01b, 0.1b, b