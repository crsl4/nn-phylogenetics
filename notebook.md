# Neural networks for phylogenetic inference

- Claudia Solis-Lemus
- Shengwen Yang
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

Actually, no, it seems that the numbers match what we would expect:
```
Model tree & branch lengths:
((S2: 0.100000, S1: 0.200000): 0.000500, (S3: 0.100000, S4: 0.200000): 0.000500);
((2: 0.100000, 1: 0.200000): 0.000500, (3: 0.100000, 4: 0.200000): 0.000500);
```
So, Si corresponds to taxon i.


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

Actually, no, it seems that the numbers match what we would expect:
```
Model tree & branch lengths:
((S2: 0.100000, S1: 0.200000): 0.000500, (S3: 0.100000, S4: 0.200000): 0.000500);
((2: 0.100000, 1: 0.200000): 0.000500, (3: 0.100000, 4: 0.200000): 0.000500);
```
So, Si corresponds to taxon i.

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

First, I need to create all the folders so that they can run in parallel: `simulations-lba-?`.

We will run nrep=8000 and 5 cores in mac desktop (which is the limit that it can run simultaneously without running out of memory)
```shell
## b=0.1, a=2b, c=0.01b
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-1
julia simulate-zou2019.jl 4738282 8000 0.1 2 0.01

## b=0.1, a=2b, c=0.1b
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-2
julia simulate-zou2019.jl 68113228 8000 0.1 2 0.1

## b=0.1, a=2b, c=b
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-3
julia simulate-zou2019.jl 68163228 8000 0.1 2 1.0

## b=0.1, a=10b, c=0.01b
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-4
julia simulate-zou2019.jl 113683228 8000 0.1 10 0.01

## b=0.1, a=10b, c=0.1b
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-5
julia simulate-zou2019.jl 68326728 8000 0.1 10 0.1
```
Started 6/20 9pm, 6am

We summarize the files:
```shell
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019

## b=0.1, a=2b, c=0.01b
cd simulations-lba-1
tar -czvf simulations-lba-1-1.tar.gz rep-1*
rm rep-1*
tar -czvf simulations-lba-1-2.tar.gz rep-2*
rm rep-2*
tar -czvf simulations-lba-1-3.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-1.h5
mv matrices.h5 matrices-lba-1.h5
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-1.jl
mv simulate-zou2019-lba-1.jl ../results

## b=0.1, a=2b, c=0.1b
cd simulations-lba-2
tar -czvf simulations-lba-2-1.tar.gz rep-1*
rm rep-1*
tar -czvf simulations-lba-2-2.tar.gz rep-2*
rm rep-2*
tar -czvf simulations-lba-2-3.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-2.h5
mv matrices.h5 matrices-lba-2.h5
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-2.jl
mv simulate-zou2019-lba-2.jl ../results

## b=0.1, a=2b, c=b
cd simulations-lba-3
tar -czvf simulations-lba-3-1.tar.gz rep-1*
rm rep-1*
tar -czvf simulations-lba-3-2.tar.gz rep-2*
rm rep-2*
tar -czvf simulations-lba-3-3.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-3.h5
mv matrices.h5 matrices-lba-3.h5
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-3.jl
mv simulate-zou2019-lba-3.jl ../results

## b=0.1, a=10b, c=0.01b
cd simulations-lba-4
tar -czvf simulations-lba-4-1.tar.gz rep-1*
rm rep-1*
tar -czvf simulations-lba-4-2.tar.gz rep-2*
rm rep-2*
tar -czvf simulations-lba-4-3.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-4.h5
mv matrices.h5 matrices-lba-4.h5
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-4.jl
mv simulate-zou2019-lba-4.jl ../results

## b=0.1, a=10b, c=0.1b
cd simulations-lba-5
tar -czvf simulations-lba-5-1.tar.gz rep-1*
rm rep-1*
tar -czvf simulations-lba-5-2.tar.gz rep-2*
rm rep-2*
tar -czvf simulations-lba-5-3.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-5.h5
mv matrices.h5 matrices-lba-5.h5
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-5.jl
mv simulate-zou2019-lba-5.jl ../results
```

```shell
## b=0.1, a=10b, c=b
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-6
julia simulate-zou2019.jl 18683228 8000 0.1 10 1.0

## b=0.1, a=40b, c=0.01b
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-7
julia simulate-zou2019.jl 976683228 8000 0.1 40 0.01

## b=0.1, a=40b, c=0.1b
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-8
julia simulate-zou2019.jl 2325654 8000 0.1 40 0.1

## b=0.1, a=40b, c=b
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-9
julia simulate-zou2019.jl 372783 8000 0.1 40 1.0

## b=0.5, a=2b, c=0.01b
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-10
julia simulate-zou2019.jl 58583625 8000 0.5 2 0.01
```
Started 6/21 10:30am, 7:40pm

We summarize the files:
```shell
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019

## b=0.1, a=10b, c=b
cd simulations-lba-6
tar -czvf simulations-lba-6-1.tar.gz rep-1*
rm rep-1*
tar -czvf simulations-lba-6-2.tar.gz rep-2*
rm rep-2*
tar -czvf simulations-lba-6-3.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-6.h5
mv matrices.h5 matrices-lba-6.h5
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-6.jl
mv simulate-zou2019-lba-6.jl ../results

## b=0.1, a=40b, c=0.01b
cd simulations-lba-7
tar -czvf simulations-lba-7-1.tar.gz rep-1*
rm rep-1*
tar -czvf simulations-lba-7-2.tar.gz rep-2*
rm rep-2*
tar -czvf simulations-lba-7-3.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-7.h5
mv matrices.h5 matrices-lba-7.h5
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-7.jl
mv simulate-zou2019-lba-7.jl ../results

## b=0.1, a=40b, c=0.1b
cd simulations-lba-8
tar -czvf simulations-lba-8-1.tar.gz rep-1*
rm rep-1*
tar -czvf simulations-lba-8-2.tar.gz rep-2*
rm rep-2*
tar -czvf simulations-lba-8-3.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-8.h5
mv matrices.h5 matrices-lba-8.h5
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-8.jl
mv simulate-zou2019-lba-8.jl ../results

## b=0.1, a=40b, c=b
cd simulations-lba-9
tar -czvf simulations-lba-9-1.tar.gz rep-1*
rm rep-1*
tar -czvf simulations-lba-9-2.tar.gz rep-2*
rm rep-2*
tar -czvf simulations-lba-9-3.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-9.h5
mv matrices.h5 matrices-lba-9.h5
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-9.jl
mv simulate-zou2019-lba-9.jl ../results

## b=0.5, a=2b, c=0.01b
cd simulations-lba-10
tar -czvf simulations-lba-10-1.tar.gz rep-1*
rm rep-1*
tar -czvf simulations-lba-10-2.tar.gz rep-2*
rm rep-2*
tar -czvf simulations-lba-10-3.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-10.h5
mv matrices.h5 matrices-lba-10.h5
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-10.jl
mv simulate-zou2019-lba-10.jl ../results
```


```shell
## b=0.5, a=2b, c=0.1b
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-11
julia simulate-zou2019.jl 5722724 8000 0.5 2 0.1

## b=0.5, a=2b, c=b
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-12
julia simulate-zou2019.jl 4919173 8000 0.5 2 1.0

## b=0.5, a=10b, c=0.01b
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-13
julia simulate-zou2019.jl 4728283 8000 0.5 10 0.01

## b=0.5, a=10b, c=0.1b
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-14
julia simulate-zou2019.jl 4473421 8000 0.5 10 0.1

## b=0.5, a=10b, c=b
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-15
julia simulate-zou2019.jl 976422 8000 0.5 10 1.0
```
Started 6/21 9pm, finished 6:30am

We summarize the files:
```shell
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019

## b=0.5, a=2b, c=0.1b
cd simulations-lba-11
tar -czvf simulations-lba-11-1.tar.gz rep-1*
rm rep-1*
tar -czvf simulations-lba-11-2.tar.gz rep-2*
rm rep-2*
tar -czvf simulations-lba-11-3.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-11.h5
mv matrices.h5 matrices-lba-11.h5
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-11.jl
mv simulate-zou2019-lba-11.jl ../results

## b=0.5, a=2b, c=b
cd simulations-lba-12
tar -czvf simulations-lba-12-1.tar.gz rep-1*
rm rep-1*
tar -czvf simulations-lba-12-2.tar.gz rep-2*
rm rep-2*
tar -czvf simulations-lba-12-3.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-12.h5
mv matrices.h5 matrices-lba-12.h5
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-12.jl
mv simulate-zou2019-lba-12.jl ../results

## b=0.5, a=10b, c=0.01b
cd simulations-lba-13
tar -czvf simulations-lba-13-1.tar.gz rep-1*
rm rep-1*
tar -czvf simulations-lba-13-2.tar.gz rep-2*
rm rep-2*
tar -czvf simulations-lba-13-3.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-13.h5
mv matrices.h5 matrices-lba-13.h5
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-13.jl
mv simulate-zou2019-lba-13.jl ../results

## b=0.5, a=10b, c=0.1b
cd simulations-lba-14
tar -czvf simulations-lba-14-1.tar.gz rep-1*
rm rep-1*
tar -czvf simulations-lba-14-2.tar.gz rep-2*
rm rep-2*
tar -czvf simulations-lba-14-3.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-14.h5
mv matrices.h5 matrices-lba-14.h5
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-14.jl
mv simulate-zou2019-lba-14.jl ../results

## b=0.5, a=10b, c=b
cd simulations-lba-15
tar -czvf simulations-lba-15-1.tar.gz rep-1*
rm rep-1*
tar -czvf simulations-lba-15-2.tar.gz rep-2*
rm rep-2*
tar -czvf simulations-lba-15-3.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-15.h5
mv matrices.h5 matrices-lba-15.h5
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-15.jl
mv simulate-zou2019-lba-15.jl ../results
```

```shell
## b=0.5, a=40b, c=0.01b
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-16
julia simulate-zou2019.jl 416173 8000 0.5 40 0.01

## b=0.5, a=40b, c=0.1b
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-17
julia simulate-zou2019.jl 3615253 8000 0.5 40 0.1

## b=0.5, a=40b, c=b
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-18
julia simulate-zou2019.jl 467733 8000 0.5 40 1.0

## b=1, a=2b, c=0.01b
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-19
julia simulate-zou2019.jl 675223 8000 1 2 0.01

## b=1, a=2b, c=0.1b
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-20
julia simulate-zou2019.jl 7842344 8000 1 2 0.1
```
Started 6/22 830am, finish 6:30pm

We summarize the files:
```shell
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019

## b=0.5, a=40b, c=0.01b
cd simulations-lba-16
tar -czvf simulations-lba-16-1.tar.gz rep-1*
rm rep-1*
tar -czvf simulations-lba-16-2.tar.gz rep-2*
rm rep-2*
tar -czvf simulations-lba-16-3.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-16.h5
mv matrices.h5 matrices-lba-16.h5
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-16.jl
mv simulate-zou2019-lba-16.jl ../results

## b=0.5, a=40b, c=0.1b
cd simulations-lba-17
tar -czvf simulations-lba-17-1.tar.gz rep-1*
rm rep-1*
tar -czvf simulations-lba-17-2.tar.gz rep-2*
rm rep-2*
tar -czvf simulations-lba-17-3.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-17.h5
mv matrices.h5 matrices-lba-17.h5
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-17.jl
mv simulate-zou2019-lba-17.jl ../results

## b=0.5, a=40b, c=b
cd simulations-lba-18
tar -czvf simulations-lba-18-1.tar.gz rep-1*
rm rep-1*
tar -czvf simulations-lba-18-2.tar.gz rep-2*
rm rep-2*
tar -czvf simulations-lba-18-3.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-18.h5
mv matrices.h5 matrices-lba-18.h5
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-18.jl
mv simulate-zou2019-lba-18.jl ../results

## b=1, a=2b, c=0.01b
cd simulations-lba-19
tar -czvf simulations-lba-19-1.tar.gz rep-1*
rm rep-1*
tar -czvf simulations-lba-19-2.tar.gz rep-2*
rm rep-2*
tar -czvf simulations-lba-19-3.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-19.h5
mv matrices.h5 matrices-lba-19.h5
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-19.jl
mv simulate-zou2019-lba-19.jl ../results

## b=1, a=2b, c=0.1b
cd simulations-lba-20
tar -czvf simulations-lba-20-1.tar.gz rep-1*
rm rep-1*
tar -czvf simulations-lba-20-2.tar.gz rep-2*
rm rep-2*
tar -czvf simulations-lba-20-3.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-20.h5
mv matrices.h5 matrices-lba-20.h5
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-20.jl
mv simulate-zou2019-lba-20.jl ../results
```


```shell
## b=1, a=2b, c=b
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-21
julia simulate-zou2019.jl 88422 8000 1 2 1.0

## b=1, a=10b, c=0.01b
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-22
julia simulate-zou2019.jl 1346243 8000 1 10 0.01

## b=1, a=10b, c=0.1b
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-23
julia simulate-zou2019.jl 3363123 8000 1 10 0.1

## b=1, a=10b, c=b
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-24
julia simulate-zou2019.jl 114134 8000 1 10 1.0

## b=1, a=40b, c=0.01b
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-25
julia simulate-zou2019.jl 3245235 8000 1 40 0.01
```
Started 6/22 630pm, finished 4:30am

We summarize the files:
```shell
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019

## b=1, a=2b, c=b
cd simulations-lba-21
tar -czvf simulations-lba-21-1.tar.gz rep-1*
rm rep-1*
tar -czvf simulations-lba-21-2.tar.gz rep-2*
rm rep-2*
tar -czvf simulations-lba-21-3.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-21.h5
mv matrices.h5 matrices-lba-21.h5
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-21.jl
mv simulate-zou2019-lba-21.jl ../results

## b=1, a=10b, c=0.01b
cd simulations-lba-22
tar -czvf simulations-lba-22-1.tar.gz rep-1*
rm rep-1*
tar -czvf simulations-lba-22-2.tar.gz rep-2*
rm rep-2*
tar -czvf simulations-lba-22-3.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-22.h5
mv matrices.h5 matrices-lba-22.h5
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-22.jl
mv simulate-zou2019-lba-22.jl ../results

## b=1, a=10b, c=0.1b
cd simulations-lba-23
tar -czvf simulations-lba-23-1.tar.gz rep-1*
rm rep-1*
tar -czvf simulations-lba-23-2.tar.gz rep-2*
rm rep-2*
tar -czvf simulations-lba-23-3.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-23.h5
mv matrices.h5 matrices-lba-23.h5
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-23.jl
mv simulate-zou2019-lba-23.jl ../results

## b=1, a=10b, c=b
cd simulations-lba-24
tar -czvf simulations-lba-24-1.tar.gz rep-1*
rm rep-1*
tar -czvf simulations-lba-24-2.tar.gz rep-2*
rm rep-2*
tar -czvf simulations-lba-24-3.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-24.h5
mv matrices.h5 matrices-lba-24.h5
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-24.jl
mv simulate-zou2019-lba-24.jl ../results

## b=1, a=40b, c=0.01b
cd simulations-lba-25
tar -czvf simulations-lba-25-1.tar.gz rep-1*
rm rep-1*
tar -czvf simulations-lba-25-2.tar.gz rep-2*
rm rep-2*
tar -czvf simulations-lba-25-3.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-25.h5
mv matrices.h5 matrices-lba-25.h5
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-25.jl
mv simulate-zou2019-lba-25.jl ../results
```


```shell
## b=1, a=40b, c=0.1b
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-26
julia simulate-zou2019.jl 45435 8000 1 40 0.1

## b=1, a=40b, c=b
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-27
julia simulate-zou2019.jl 346266 8000 1 40 1.0
```
Started 6/23 9am, finished 4pm.

We summarize the files:
```shell
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019

## b=1, a=40b, c=0.1b
cd simulations-lba-26
tar -czvf simulations-lba-26-1.tar.gz rep-1*
rm rep-1*
tar -czvf simulations-lba-26-2.tar.gz rep-2*
rm rep-2*
tar -czvf simulations-lba-26-3.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-26.h5
mv matrices.h5 matrices-lba-26.h5
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-26.jl
mv simulate-zou2019-lba-26.jl ../results

## b=1, a=40b, c=b
cd simulations-lba-27
tar -czvf simulations-lba-27-1.tar.gz rep-1*
rm rep-1*
tar -czvf simulations-lba-27-2.tar.gz rep-2*
rm rep-2*
tar -czvf simulations-lba-27-3.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-27.h5
mv matrices.h5 matrices-lba-27.h5
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-27.jl
mv simulate-zou2019-lba-27.jl ../results
```

We move all the results to `v3`:
```shell
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/results
mv *.h5 v3
mv *.jl v3
mv *.gz v3
```

### Adding more simulations to LBA cases

We noticed that we do not have the same performance as in the Zou2019 paper. It could be that we are using a much smaller sample size (8000 vs 100k in Zou2019).

In total, there are the following 120 cases:
- b=0.1, 0.2, 0.5, 1 (4)
- a= 2b, 5b, 10b, 20b, 40b (5)
- c=0.01b, 0.02b, 0.05b, 0.1b, 0.2b, 0.5b, b (6)

We will only do the following 27 cases:
- b=0.1, 0.5, 1
- a=2b, 10b, 40b
- c=0.01b, 0.1b, b

First, I need to create all the folders so that they can run in parallel: `simulations-lba-?`:
```shell
for i in {1..27}
do
mkdir simulations-lba-$i
done

for i in {1..27}
do
cp scripts/* simulations-lba-$i/
done
```

We will run nrep=100000 and 10 cores in mac desktop (since we are not saving the onehot matrices, I think we can use more cores):
```shell
## b=0.1, a=2b, c=0.01b
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-1
julia simulate-zou2019.jl 4738282 100000 0.1 2 0.01 0

## b=0.1, a=2b, c=0.1b
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-2
julia simulate-zou2019.jl 68113228 100000 0.1 2 0.1 0

## b=0.1, a=2b, c=b
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-3
julia simulate-zou2019.jl 68163228 100000 0.1 2 1.0 0

## b=0.1, a=10b, c=0.01b
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-4
julia simulate-zou2019.jl 113683228 100000 0.1 10 0.01 0

## b=0.1, a=10b, c=0.1b
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-5
julia simulate-zou2019.jl 68326728 100000 0.1 10 0.1 0

## b=0.1, a=10b, c=b
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-6
julia simulate-zou2019.jl 18683228 100000 0.1 10 1.0 0

## b=0.1, a=40b, c=0.01b
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-7
julia simulate-zou2019.jl 976683228 100000 0.1 40 0.01 0

## b=0.1, a=40b, c=0.1b
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-8
julia simulate-zou2019.jl 2325654 100000 0.1 40 0.1 0

## b=0.1, a=40b, c=b
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-9
julia simulate-zou2019.jl 372783 100000 0.1 40 1.0 0

## b=0.5, a=2b, c=0.01b
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-10
julia simulate-zou2019.jl 58583625 100000 0.5 2 0.01 0
```
Started 8/2 9pm, finished 1am.


```shell
## b=0.5, a=2b, c=0.1b
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-11
julia simulate-zou2019.jl 5722724 100000 0.5 2 0.1 0

## b=0.5, a=2b, c=b
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-12
julia simulate-zou2019.jl 4919173 100000 0.5 2 1.0 0

## b=0.5, a=10b, c=0.01b
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-13
julia simulate-zou2019.jl 4728283 100000 0.5 10 0.01 0

## b=0.5, a=10b, c=0.1b
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-14
julia simulate-zou2019.jl 4473421 100000 0.5 10 0.1 0

## b=0.5, a=10b, c=b
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-15
julia simulate-zou2019.jl 976422 100000 0.5 10 1.0 0

## b=0.5, a=40b, c=0.01b
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-16
julia simulate-zou2019.jl 416173 100000 0.5 40 0.01 0

## b=0.5, a=40b, c=0.1b
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-17
julia simulate-zou2019.jl 3615253 100000 0.5 40 0.1 0

## b=0.5, a=40b, c=b
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-18
julia simulate-zou2019.jl 467733 100000 0.5 40 1.0 0

## b=1, a=2b, c=0.01b
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-19
julia simulate-zou2019.jl 675223 100000 1 2 0.01 0

## b=1, a=2b, c=0.1b
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-20
julia simulate-zou2019.jl 7842344 100000 1 2 0.1 0

## b=1, a=2b, c=b
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-21
julia simulate-zou2019.jl 88422 100000 1 2 1.0 0

## b=1, a=10b, c=0.01b
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-22
julia simulate-zou2019.jl 1346243 100000 1 10 0.01 0
```
Started 8/3 12pm, finished 4pm.


```shell
## b=1, a=10b, c=0.1b
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-23
julia simulate-zou2019.jl 3363123 100000 1 10 0.1 0

## b=1, a=10b, c=b
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-24
julia simulate-zou2019.jl 114134 100000 1 10 1.0 0

## b=1, a=40b, c=0.01b
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-25
julia simulate-zou2019.jl 3245235 100000 1 40 0.01 0

## b=1, a=40b, c=0.1b
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-26
julia simulate-zou2019.jl 45435 100000 1 40 0.1 0

## b=1, a=40b, c=b
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-27
julia simulate-zou2019.jl 346266 100000 1 40 1.0 0
```
Started 8/3 830am, finish 1230pm

```shell
for i in {1..27}
do
cp simulations-lba-$i/*.in results
done

cd results
ls *.in | wc -l  ##54
```

Copy the *.in files to shared google drive.
Folders stored in `v4` for now.

Dimensions of files:
```shell
(master) $ awk '{ print length }' sequences976422-0.5-10.0-1.0.in | head
1550
1550
1550
1550
1550
1550
1550
1550
1550
1550
(master) $ wc -l sequences976422-0.5-10.0-1.0.in 
  400000 sequences976422-0.5-10.0-1.0.in
```


Next steps:
- simulations for 5 taxa (don't do the one hot encoding)
- read existing NN papers (done)
- compare with other standard inference methods (done)


## Using Zou2019 script to simulate trees n=5 (quintets)

Created file `simulate-n5.jl` and `functions.jl` which is a copy of the functions for Zou2019 (`functions-zou2019.jl`)

with simulation pipeline to simulate trees of size 5 (quintets). This script is not tested yet.
Careful: this and previous scripts are not exploiting the fact that we could generate data from all possible roots. 

After all work done on quartets, we want to explore the quintets again. See below.

# Onboarding Zelin and Shengwen

Thanks for agreeing to work on this project! We are excited to have you on board.

Next steps:
1. You will receive an invitation to the slack workspace. I hope to use mostly slack to communicate as a group
2. You will receive an invitation to the [google drive](https://drive.google.com/drive/u/2/folders/0ACu5ePKXaJaiUk9PVA) where the simulated data is stored
3. You will receive an invitation to the [github repo](https://github.com/crsl4/nn-phylogenetics) where the scripts are stored. You will have push permissions in the repo, so please make sure to read and adhere to the best computing practices [guidelines](https://github.com/crsl4/mindful-programming/blob/master/lecture.md)
4. Fill out this form: https://www.when2meet.com/?10101364-VKPxm to identify a time that will work regularly to meet as a group. I chose next week as the week in the poll, but chose times that work for you every week of the semester. We can decide if we start next week or in two weeks. We also need to decide if we want to meet every week or every two weeks. Personally, I prefer every two weeks this semester, but we can discuss later (we can do this discussion in slack)
5. Check out the project plan (with list of main papers) in the github repo. Also, expect an email from Leo with more info about the code
6. You will receive an invitation to the [Paperpile folder](https://paperpile.com/app/shared/2qGDFH) where we keep track of the papers for this project

After this email, I hope we can keep communicating via slack where I will also post this same message and we can discuss on the frequency of regular meetings.

Thanks,
Claudia


# Notes from new Leo code (meeting 2/18)

- Code is for 4 taxa only and it follows more closely the notation on the overleaf doc
- In the old code, we had `DescriptorModule` as \phi and `MergeModule` as \Phi so that the pipeline would be from a 1550x20x4 tensor to 1550x20 to 250x20 then into MergeModule to 50x20 and then into MergeModule2 to a score
- The new code had the pipeline of 1550x20 to a vector of 128 to another vector of 128 and then to a score
       - `NonlinearEmbedding`: \phi
       - `NonlinearMergeEmbedding`: \Phi
       - `NonlinearScoreEmbedding`: \Psi
- This new code does not work and Leo suspects it is because we are losing some of the geometry of the sequence
- An alternative version goes from 1550x20 to Matrix mxc (via \phi) then to Matrix mxc again (via \Phi) and finally to score via \Psi
- This new version works better as it seems to keep the geometry of the input
- The files `gen_json_files.py` and `gen_sh_files.py` provide automatic means for tests on slurm
- We want to start exploring extension to 5 or 6 taxa for the paper


# Comparing NN performance to standard phylogenetics inference

## 1. Extracting sample data
We will grab one 4-taxon dataset from a randomly chosen file: `sequences45435-1.0-40.0-0.1.in` and `labels45435-1.0-40.0-0.1.in`.

In the sequence file, we have a 400,000 x 1550 matrix in which we have 100,000 4-taxon datasets. We will grab the last 4 rows which correspond to one 4-taxon dataset.

```shell
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/results
wc -l sequences45435-1.0-40.0-0.1.in  ## 400000
sed -n '399997,400000 p' sequences45435-1.0-40.0-0.1.in > test.in
```

The order of the sequences are always S1,S2,S3,S4.

## 2. Convert sample data to Fasta file

Most phylogenetic methods will need a fasta file as input data. We will create this in julia:

```julia
datafile = "test.in"
lines = readlines(datafile)
fastafile = "test.fasta"
io = open(fastafile, "w")

n = length(lines)
l = length(lines[1])

write(io,"$n $l \n")
for i in 1:n
   write(io, string(">",i,"\n"))
   write(io, lines[i])
   write(io, "\n")
end

close(io)
```

## 3. Fitting maximum parsimony (MP) and neighbor-joining (NJ) in R

The easiest phylogenetic methods to fit are MP and NJ, both in R.

To install the necessary packages in R:
```r
install.packages("ape", dep=TRUE)
install.packages("phangorn", dep=TRUE)
install.packages("adegenet", dep=TRUE) ##I get a warning message
install.packages("seqinr", dep=TRUE)
```

To fit the NJ model:
```r
library(ape)
library(phangorn)
library(adegenet)

## Reading fasta file
aa = read.aa(file="test.fasta", format="fasta")

## Estimating evolutionary distances using same model as in simulations
D = dist.ml(aa, model="Dayhoff") 

## Estimating tree
tree = nj(D)

## Saving estimated tree to text file
write.tree(tree,file="nj-tree.txt")
```

Note that we cannot fit the MP model on aminoacid sequences in R (we need DNA sequences).

## 4. Fitting maximum likelihood (ML) with RAxML

1. Download `raxml-ng` from [here](https://github.com/amkozlov/raxml-ng). You get a zipped folder: `raxml-ng_v1.0.2_macos_x86_64` which I placed in my `software` folder

2. Checking the version
```shell
cd Dropbox/software/raxml-ng_v1.0.2_macos_x86_64/
./raxml-ng -v
```

3. Infer the ML tree using the same model as in the simulations
```shell
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/results
~/Dropbox/software/raxml-ng_v1.0.2_macos_x86_64/raxml-ng --msa test.fasta --model Dayhoff --prefix T3 --threads 2 --seed 616

Final LogLikelihood: -14728.557112

AIC score: 29467.114224 / AICc score: 29467.153084 / BIC score: 29493.844275
Free parameters (model + branch lengths): 5

Best ML tree saved to: /Users/Clauberry/Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/results/T3.raxml.bestTree
All ML trees saved to: /Users/Clauberry/Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/results/T3.raxml.mlTrees
Optimized model saved to: /Users/Clauberry/Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/results/T3.raxml.bestModel

Execution log saved to: /Users/Clauberry/Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/results/T3.raxml.log

Analysis started: 23-Mar-2021 18:18:12 / finished: 23-Mar-2021 18:18:13

Elapsed time: 0.921 seconds
```

Best tree saved in `T3.raxml.bestTree`. We used the `T3` prefix for the files because of the raxml tutorial, but we can choose any prefix.
Warning: Note that the tree has the `>` as part of the taxon names.

We will get rid of the `>` in the shell to avoid problems later:
```shell
sed -i '' -e $'s/>//g' T3.raxml.bestTree
```

## 5. Fitting bayesian inference (BI) with MrBayes

1. Download MrBayes from [here](http://nbisweden.github.io/MrBayes/). In mac:
```shell
brew tap brewsci/bio
brew install mrbayes --with-open-mpi

$ which mb
/usr/local/bin/mb
```

Had to troubleshoot a lot!
```shell
brew reinstall mrbayes
sudo chown -R $(whoami) /usr/local/Cellar/open-mpi/4.1.0
brew reinstall mrbayes
```

2. MrBayes needs nexus files. We will do this in R:
```r
library(ape)
library(phangorn)
library(adegenet)

## Reading fasta file
aa = read.aa(file="test.fasta", format="fasta")

## Write as nexus file
write.nexus.data(aa,file="test.nexus",format="protein")
```

3. Add the mrbayes block to the nexus file. MrBayes requires that you write a text block at the end of the nexus file. We will write this block in a text file called `mb-block.txt` and we can use the same block for all runs.
```
begin mrbayes;
set nowarnings=yes;
set autoclose=yes;
prset aamodel=fixed(dayhoff);
mcmcp ngen=100000 burninfrac=.25 samplefreq=50 printfreq=10000 [increase these for real]
diagnfreq=10000 nruns=2 nchains=2 temp=0.40 swapfreq=10;       [increase for real analysis]
mcmc;
sumt;
end;
```
This block specifies the length of the MCMC chain and the aminoacid model (Dayhoff which is the same used in simulations).

We will add the mrbayes block in the shell:
```shell
cat test.nexus mb-block.txt > test-mb.nexus
```

4. Run MrBayes:
```shell
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/results
mb test-mb.nexus
```

The estimated tree is in `test-mb.nexus.con.tre`.

## 6. Comparing the estimated trees to the true tree

The true tree is in the file `labels45435-1.0-40.0-0.1.in`.
We will use R to compare the trees.

```r
## read true trees
d = read.table("labels45435-1.0-40.0-0.1.in", header=FALSE)
n = length(d$V1)
## labels from the simulating script:
quartets = c("((1,2),(3,4));", "((1,3),(2,4));", "((1,4),(2,3));")
## to which quartet the label corresponds to:
truetree = read.tree(text=quartets[d[n,]])

library(ape)
## read the NJ tree:
njtree = read.tree(file="nj-tree.txt")
## read the ML tree:
mltree = read.tree(file="T3.raxml.bestTree")
## read the BI tree
bitree = read.nexus(file="test-mb.nexus.con.tre")

## Calculating the Robinson-Foulds distance with true tree:
library(phangorn)
njdist = RF.dist(truetree,njtree, rooted=FALSE)
mldist = RF.dist(truetree,mltree, rooted=FALSE)
bidist = RF.dist(truetree,bitree, rooted=FALSE)
```
If the distance is equal to zero, then the method reconstructed the correct tree. For example, `njdist==0` implies that NJ estimated the correct tree.

# New simulations pseudo-code

If the results after the sanity check (comparison with NJ, RAxML and MrBayes) is not favorable, we might want to do more simulations.

It might be easier if the simulation scripts are in python too, so I write an algorithm below that we could convert to python script.

## Option 1: Simulate large trees and then extract 4-taxon trees
The advantage of this approach is that we will see many different tree patterns by simulating large trees first, and then extracting the 4-taxon trees, as opposed to simulating 4-taxon trees directly.

Parameters:
- `n`: number of trees to simulate
- `lambda`: birth rate for tree simulation
- `nu`: death rate for tree simulation
- `tau`: maximum time that we let simulation run to generate a random tree. We want this quantity to be large so that we have trees with different number of taxa
- `seed`: global random seed to generate seeds for each individual run
- `kappa`: transition-tranverstion bias for the simulation of sequences under the HKY model
- `pi`: state frequencies of nucleotides (4 total: A,C,G,T)
- `L`: length of sequence


For i in 1:n
1. Simulate a random tree under the birth death process using [dendropy](https://dendropy.org/library/birthdeath.html):
`tree = dendropy.model.birthdeath.birth_death_tree(lambda, nu, max_time=tau, rng=seeds[i])`
2. Simulate DNA sequences on the tree following [this script by Pyvolve](https://github.com/sjspielman/pyvolve/blob/master/examples/custom_nucleotide.py)
`my_evolver = pyvolve.Evolver(partitions=my_partitions, size=L)`
3. Prune the tree to only 4 leaves with [ETE](http://etetoolkit.org/docs/latest/reference/reference_tree.html?highlight=prune#ete3.TreeNode.prune)
`tree.prune([1,2,3,4], preserve_branch_length=True)`. Note that we also need to extract only the 4 sequences and assign a label to the tree whether it corresponds to the 12|34, 13|24 or 14|23 quartet

Output:
- List of `n` quartet labels
- List of `n` 4-taxon trees in parenthetical format with branch lengths (we will need this to summarize the results)
- `n` 4xL matrices with nucleotides


## Option 2: Simulate 4-taxon trees with different branch length setups
We can use Table 1 in [this paper](https://academic.oup.com/sysbio/article/69/2/221/5559282) for the different branch length setups on 4-taxon trees.

Parameters:
- `n`: number of trees to simulate
- `seed`: global random seed to generate seeds for each individual run
- `kappa`: transition-tranverstion bias for the simulation of sequences under the HKY model
- `pi`: state frequencies of nucleotides (4 total: A,C,G,T)
- `L`: length of sequence


For i in 1:n
1. Choose randomly one quartet: 12|34, 13|24, 14|23 (use specific seed `seeds[i]` for replication) and read the tree with [Dendropy](https://dendropy.org/primer/reading_and_writing.html):
`tree3 = dendropy.Tree.get(data="((A,B),(C,D));", schema="newick")`
2. Choose randomly one scenario from Table 1:
       - Truncated exponential
       - Farris zone
       - Twisted Farris zone
       - Extended Farris zone
       - Felsenstein zone
       - Extended Felsenstein zone
       - Long branches
       - Extra long branches
       - Single long branch
       - Short branches
       - Extra short branches
       - Single short branch
       - Short internal branch
3. Choose randomly branches for each of the 5 branches according to the distributions of the specific selected scenario (e.g. for "Truncated exponential" all branches are sampled from Truncated Exponential(10,0,0.5)). See [this script from Dendropy](https://dendropy.org/primer/reading_and_writing.html) for an example on how to set branch lengths with `edge.length = x` and we can use [numpy](https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html) for the generation of random Uniform or Exponential random variables
4. Simulate DNA sequences on the tree following [this script by Pyvolve](https://github.com/sjspielman/pyvolve/blob/master/examples/custom_nucleotide.py)
`my_evolver = pyvolve.Evolver(partitions=my_partitions, size=L)`


Output:
- List of `n` quartet labels
- List of `n` 4-taxon trees in parenthetical format with branch lengths (we will need this to summarize the results)
- `n` 4xL matrices with nucleotides


# Discrepancy with Zou and our NN implementation

Shengwen identified two cases in which Zou's NN implementation is good (0.84, 0.97 accuracy) and our accuracy is ~0.4.
We want to simulate other datasets with the same settings (different seed) to compare the performance again.

Case 1:
```shell
## b=0.1, a=10b, c=b
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-case1-1
julia simulate-zou2019.jl XXXXX 8000 0.1 10 1.0
```

Case 2:
```shell
## b=1, a=10b, c=b
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-case2-1
julia simulate-zou2019.jl XXXXX 8000 1 10 1.0
```


# Real data analysis

## 1. Cats-dogs dataset

From the [bistro project](https://github.com/crsl4/ccdprobs), we have the `cats-dogs.fasta` file with 8 taxa and ~1500 sites. The problem is that these are nucleotides, not aminoacids.

From bio collaborators:
- You can use this tool: https://web.expasy.org/translate/
- https://www.khanacademy.org/science/ap-biology/gene-expression-and-regulation/translation/a/the-genetic-code-discovery-and-properties
- In that tool you get all 6 frames (because DNA is 2 strands that are complementary to each other, you have frames 1,2,3 and -1,-2,-3)

So, we use the `expasy` tool to copy one sequence at a time to create the `cats-dogs-aa.fasta` file (manually).
However, for each nucleotide sequence, we get 6 aminoacid sequences (due to the reading frames) and we would have to select which one. 
So, it is best if we find a dataset that is on aminoacids already.

## 2. Birds dataset from Reddy et al 2017

- Downloaded data from [dryad](https://datadryad.org/stash/dataset/doi:10.5061/dryad.6536v). File/folder info is there, but also copied into a readme file inside the folder `doi_10.5061_dryad.6536v__v1`.
- These are also nucleotides

## 3. Creating our own data

- Using [NCBI](https://www.ncbi.nlm.nih.gov/genomes/VirusVariation/Database/nph-select.cgi) for Zika virus
- Searching for Human, Mammal, Primate samples; any genome region
- [Query link](https://www.ncbi.nlm.nih.gov/genomes/VirusVariation/Database/nph-select.cgi?cmd=show_builder&country=any&download-select=fP&genregion=any&go=database&host=Primate&isolation=isolation_blood&query_1_count=1124&query_1_count_genome_sets=0&query_1_country=any&query_1_genregion=any&query_1_host=Human&query_1_isolation=any&query_1_line=on&query_1_line_num=1&query_1_query_key=1&query_1_searchin=sequence&query_1_sequence=P&query_1_srcfilter_labs=include&query_1_taxid=64320&query_2_count=101&query_2_count_genome_sets=0&query_2_country=any&query_2_genregion=any&query_2_host=Primate&query_2_isolation=any&query_2_line=on&query_2_line_num=2&query_2_query_key=1&query_2_searchin=sequence&query_2_sequence=P&query_2_srcfilter_labs=include&query_2_taxid=64320&query_3_count=4&query_3_count_genome_sets=0&query_3_country=any&query_3_genregion=any&query_3_host=Mammal&query_3_isolation=any&query_3_line=on&query_3_line_num=3&query_3_query_key=1&query_3_searchin=sequence&query_3_sequence=P&query_3_srcfilter_labs=include&query_3_taxid=64320&query_4_count=0&query_4_count_genome_sets=0&query_4_country=any&query_4_genregion=any&query_4_host=Mammal&query_4_isolation=isolation_blood&query_4_line_num=4&query_4_query_key=1&query_4_searchin=sequence&query_4_sequence=P&query_4_srcfilter_labs=include&query_4_taxid=64320&query_5_count=357&query_5_count_genome_sets=0&query_5_country=any&query_5_genregion=any&query_5_host=Human&query_5_isolation=isolation_blood&query_5_line_num=5&query_5_query_key=1&query_5_searchin=sequence&query_5_sequence=P&query_5_srcfilter_labs=include&query_5_taxid=64320&query_6_count=0&query_6_count_genome_sets=0&query_6_country=any&query_6_genregion=any&query_6_host=Primate&query_6_isolation=isolation_blood&query_6_line_num=6&query_6_query_key=1&query_6_searchin=sequence&query_6_sequence=P&query_6_srcfilter_labs=include&query_6_taxid=64320&searchin=sequence&sequence=P&srcfilter_labs=include&taxid=64320)
- Manually selected (accession, length, host, country, collection year):

       - BBA85762, 3423, Homo sapiens, Japan, 2016
       - QIH53581, 3423, Homo sapiens, Brazil, 2017
       - BAP47441, 3423, Simiiformes, Uganda, 1947
       - ANG09399, 3423, Homo sapiens, Honduras, 2016
       - AXF50052, 3423, Mus Musculus, Colombia, 2016
       - AWW21402, 3423, Simiiformes, Cambodia, 2016
       - AYI50274, 3423, Macaca mulatta, xxxxx, 2015

- Downloaded as `FASTA.fa`. All sequences have the same length, so no need to align.
- The website creates a tree which is downloaded as `tree.nwk`. This tree is strange because it puts Macaca mulatta right in the middle of homo sapiens.

## Creating files with 4 taxa

Our dataset has 7 species, so we need to create subsets of 4 to fit in our NN.

```julia
data = readlines("data/FASTA.fa")

taxa = []
seqs = []

seq = ""
for l in data
   if occursin(">",l)
      push!(taxa,l)
      push!(seqs,seq) ##push previous seq
      seq = ""
   else
      seq *= l
   end
end
push!(seqs,seq) ##push last seq

## by the way it is constructed, we have an extra empty seq in seqs:
deleteat!(seqs, 1)
```

Now we have two vectors: `taxa` with the taxon names and `seqs` with the sequences.

First, we create a translate table with taxon names:
```julia
using DataFrames, CSV
df = DataFrame(id=1:length(taxa), name=taxa)
CSV.write("data/fasta-table.csv",df)
```

Now, we create one datafile for each combination:
```r
> choose(7,4)
[1] 35
```

```julia
using Combinatorics
comb = collect(combinations(1:length(taxa),4))
## 35-element Vector{Vector{Int64}}:

i = 1
for c in comb
   io = open(string("data/zika-fasta",i,".fa"), "w")
   for j in c
      write(io, string(">",j))
      write(io, "\n")
      write(io, seqs[j])
      write(io, "\n")
   end
   close(io)
   i += 1
end   
```

### Removing 7th taxon

The 7th taxon (see `FASTA.fa`) is "BBA85762, 3423, Homo sapiens, Japan, 2016" which contains missing sites X. Because our NN model was not trained with missing sites, we will remove this taxon from the dataset.


## Quartet puzzling

We can do the quartet puzzling step with Quartet Max Cut. It needs an input file that is one line with each quartet separated by a space in the form of a split: "1,2|3,4".

See `final_plots.Rmd` for the plot of the tree.


# Continuation: Simulate trees n=5 (quintets)

We want to simulate data on the 5-taxon tree. We had to download PAML again because the old `evolver` executable did not run in the new mac OS.

1. Git clone: `git clone https://github.com/abacus-gene/paml`
2. Inside `src`, `make`:
```
$ make
cc  -O3 -Wall -Wno-unused-result -c baseml.c
cc  -O3 -Wall -Wno-unused-result -c tools.c
cc  -O3 -Wall -Wno-unused-result -o baseml baseml.o tools.o -lm 
cc  -O3 -Wall -Wno-unused-result -c codeml.c
codeml.c:5052:24: warning: if statement has empty body [-Wempty-body]
         if (com.getSE);
                       ^
codeml.c:5052:24: note: put the semicolon on a separate line to silence this warning
codeml.c:6574:14: warning: using floating point absolute value function 'fabs' when argument is of integer type [-Wabsolute-value]
         if (fabs(square((int)sqrt(k + 1.)) - (k + 1)) < 1e-5) 
             ^
codeml.c:6574:14: note: use function 'abs' instead
         if (fabs(square((int)sqrt(k + 1.)) - (k + 1)) < 1e-5) 
             ^~~~
             abs
codeml.c:6991:11: warning: using floating point absolute value function 'fabs' when argument is of integer type [-Wabsolute-value]
      if (fabs(square((int)sqrt(k + 1.)) - (k + 1)) < 1e-5) 
          ^
codeml.c:6991:11: note: use function 'abs' instead
      if (fabs(square((int)sqrt(k + 1.)) - (k + 1)) < 1e-5) 
          ^~~~
          abs
3 warnings generated.
cc  -O3 -Wall -Wno-unused-result -o codeml codeml.o tools.o -lm 
cc  -O3 -Wall -Wno-unused-result -c basemlg.c
cc  -O3 -Wall -Wno-unused-result -o basemlg basemlg.o tools.o -lm 
cc  -O3 -Wall -Wno-unused-result -c mcmctree.c
cc  -O3 -Wall -Wno-unused-result -o mcmctree mcmctree.c tools.o -lm 
cc  -O3 -Wall -Wno-unused-result -o infinitesites -D INFINITESITES mcmctree.c tools.o -lm 
cc  -O3 -Wall -Wno-unused-result -c pamp.c
cc  -O3 -Wall -Wno-unused-result -o pamp pamp.o tools.o -lm 
cc  -O3 -Wall -Wno-unused-result -c evolver.c
cc  -O3 -Wall -Wno-unused-result -o evolver evolver.o tools.o -lm 
cc  -O3 -Wall -Wno-unused-result -c yn00.c
cc  -O3 -Wall -Wno-unused-result -o yn00 yn00.o tools.o -lm 
cc  -O3 -Wall -Wno-unused-result   -c -o chi2.o chi2.c
cc  -O3 -Wall -Wno-unused-result -o chi2 chi2.c -lm 

$ ls
BFdriver.c    basemlg       codeml.ctl    mcmctree.c    tools.o
Makefile      basemlg.c     codeml.o      mcmctree.ctl  treespace.c
Makefile.VC   basemlg.o     ds.c          mcmctree.o    treesub.c
README.txt    chi2          evolver       paml.h        yn00
baseml        chi2.c        evolver.c     pamp          yn00.c
baseml.c      chi2.o        evolver.o     pamp.c        yn00.o
baseml.ctl    codeml        infinitesites pamp.o
baseml.o      codeml.c      mcmctree      tools.c
```
3. Copy `evolver` executable to `simulations-5taxa/scripts`.

```shell
cd nn-phylogenetics/simulations-5taxa/scripts
julia simulate-n5.jl 12062021 10000
```

We do 10000 for now, but I think we will need more for 5 taxa.

We now move all files to a subfolder:
```shell
mkdir sim-5taxa
mv rep*.dat sim-5taxa
```
and this subfolder is added to the google drive.

# Running IQ-Tree on Zika data

```
$ iqtree -s FASTA.fa 
IQ-TREE multicore version 1.6.12 for Mac OS X 64-bit built Aug 15 2019
Developed by Bui Quang Minh, Nguyen Lam Tung, Olga Chernomor,
Heiko Schmidt, Dominik Schrempf, Michael Woodhams.

Host:    C02Z60BVM0XV.local (AVX512, FMA3, 256 GB RAM)
Command: /Users/Clauberry/Dropbox/software/iqtree-1.6.12-MacOSX/bin/iqtree -s FASTA.fa
Seed:    991296 (Using SPRNG - Scalable Parallel Random Number Generator)
Time:    Sat Dec  9 09:41:10 2023
Kernel:  AVX+FMA - 1 threads (36 CPU cores detected)

HINT: Use -nt option to specify number of threads because your CPU has 36 cores!
HINT: -nt AUTO will automatically determine the best number of threads to use.

Reading alignment file FASTA.fa ... Fasta format detected
Alignment most likely contains protein sequences
Alignment has 7 sequences with 3423 columns, 107 distinct patterns
9 parsimony-informative, 131 singleton sites, 3283 constant sites
          Gap/Ambiguity  Composition  p-value
   1  AXF50052    0.00%    passed    100.00%
   2  AWW21402    0.00%    passed    100.00%
   3  AYI50274    0.00%    passed    100.00%
   4  QIH53581    0.00%    passed    100.00%
   5  BAP47441    0.00%    passed    100.00%
   6  ANG09399    0.00%    passed    100.00%
   7  BBA85762    0.03%    passed    100.00%
****  TOTAL       0.00%  0 sequences failed composition chi2 test (p-value<5%; df=19)


Create initial parsimony tree by phylogenetic likelihood library (PLL)... 0.000 seconds
NOTE: ModelFinder requires 1 MB RAM!
ModelFinder will test 546 protein models (sample size: 3423) ...
 No. Model         -LnL         df  AIC          AICc         BIC
  1  Dayhoff       11111.578    11  22245.157    22245.234    22312.678
  2  Dayhoff+I     11111.130    12  22246.261    22246.352    22319.920
  3  Dayhoff+G4    11111.143    12  22246.286    22246.377    22319.945
  4  Dayhoff+I+G4  11111.141    13  22248.283    22248.390    22328.080
  5  Dayhoff+R2    11111.190    13  22248.381    22248.488    22328.178
  6  Dayhoff+R3    11111.150    15  22252.300    22252.441    22344.375
 14  Dayhoff+F     10918.517    30  21897.034    21897.582    22081.182
 15  Dayhoff+F+I   10918.190    31  21898.379    21898.964    22088.666
 16  Dayhoff+F+G4  10918.203    31  21898.406    21898.992    22088.693
 17  Dayhoff+F+I+G4 10918.238    32  21900.476    21901.099    22096.901
 18  Dayhoff+F+R2  10918.212    32  21900.424    21901.047    22096.849
 19  Dayhoff+F+R3  10918.201    34  21904.402    21905.104    22113.103
 27  mtMAM         11499.646    11  23021.291    23021.369    23088.812
 28  mtMAM+I       11497.449    12  23018.899    23018.990    23092.558
 29  mtMAM+G4      11497.414    12  23018.828    23018.919    23092.487
 30  mtMAM+I+G4    11497.390    13  23020.779    23020.886    23100.577
 31  mtMAM+R2      11498.509    13  23023.018    23023.125    23102.816
 32  mtMAM+R3      11498.204    15  23026.408    23026.549    23118.482
 40  mtMAM+F       10879.697    30  21819.393    21819.942    22003.541
 41  mtMAM+F+I     10877.732    31  21817.465    21818.050    22007.751
 42  mtMAM+F+G4    10877.719    31  21817.438    21818.023    22007.724
 43  mtMAM+F+I+G4  10877.713    32  21819.427    21820.050    22015.852
 44  mtMAM+F+R2    10878.621    32  21821.242    21821.865    22017.667
 45  mtMAM+F+R3    10878.343    34  21824.685    21825.388    22033.386
 53  JTT           10966.782    11  21955.565    21955.642    22023.086
 54  JTT+I         10966.643    12  21957.285    21957.377    22030.945
 55  JTT+G4        10966.652    12  21957.304    21957.396    22030.964
 56  JTT+I+G4      10966.855    13  21959.710    21959.817    22039.507
 57  JTT+R2        10966.660    13  21959.320    21959.427    22039.117
 58  JTT+R3        10966.659    15  21963.318    21963.459    22055.392
 66  JTT+F         10877.255    30  21814.510    21815.058    21998.658
 67  JTT+F+I       10877.114    31  21816.228    21816.813    22006.515
 68  JTT+F+G4      10877.126    31  21816.251    21816.836    22006.538
 69  JTT+F+I+G4    10877.325    32  21818.651    21819.274    22015.076
 70  JTT+F+R2      10877.132    32  21818.264    21818.887    22014.689
 71  JTT+F+R3      10877.131    34  21822.262    21822.965    22030.963
 79  WAG           10999.159    11  22020.317    22020.395    22087.838
 80  WAG+I         10998.977    12  22021.955    22022.046    22095.614
 81  WAG+G4        10998.991    12  22021.983    22022.074    22095.642
 82  WAG+I+G4      10999.133    13  22024.266    22024.373    22104.064
 83  WAG+R2        10998.988    13  22023.975    22024.082    22103.773
 84  WAG+R3        10998.987    15  22027.974    22028.115    22120.049
 92  WAG+F         10901.940    30  21863.881    21864.429    22048.029
 93  WAG+F+I       10901.787    31  21865.575    21866.160    22055.861
 94  WAG+F+G4      10901.802    31  21865.605    21866.190    22055.891
 95  WAG+F+I+G4    10901.979    32  21867.957    21868.580    22064.382
 96  WAG+F+R2      10901.804    32  21867.607    21868.230    22064.032
 97  WAG+F+R3      10901.803    34  21871.605    21872.308    22080.307
105  cpREV         11027.828    11  22077.656    22077.734    22145.177
106  cpREV+I       11027.507    12  22079.015    22079.106    22152.674
107  cpREV+G4      11027.532    12  22079.064    22079.156    22152.724
108  cpREV+I+G4    11027.558    13  22081.117    22081.223    22160.914
109  cpREV+R2      11027.533    13  22081.066    22081.173    22160.863
110  cpREV+R3      11027.533    15  22085.066    22085.207    22177.140
118  cpREV+F       10902.650    30  21865.299    21865.848    22049.447
119  cpREV+F+I     10902.289    31  21866.578    21867.163    22056.865
120  cpREV+F+G4    10902.316    31  21866.632    21867.218    22056.919
121  cpREV+F+I+G4  10902.324    32  21868.647    21869.270    22065.072
122  cpREV+F+R2    10902.325    32  21868.651    21869.274    22065.076
123  cpREV+F+R3    10902.325    34  21872.651    21873.353    22081.352
131  mtREV         11488.117    11  22998.235    22998.312    23065.756
132  mtREV+I       11487.666    12  22999.333    22999.424    23072.992
133  mtREV+G4      11487.685    12  22999.371    22999.462    23073.030
134  mtREV+I+G4    11487.679    13  23001.357    23001.464    23081.155
135  mtREV+R2      11487.727    13  23001.454    23001.561    23081.252
136  mtREV+R3      11487.727    15  23005.454    23005.595    23097.529
144  mtREV+F       10888.891    30  21837.783    21838.331    22021.931
145  mtREV+F+I     10888.459    31  21838.917    21839.502    22029.204
146  mtREV+F+G4    10888.483    31  21838.966    21839.551    22029.252
147  mtREV+F+I+G4  10888.474    32  21840.948    21841.571    22037.373
148  mtREV+F+R2    10888.515    32  21841.030    21841.653    22037.455
149  mtREV+F+R3    10888.515    34  21845.030    21845.733    22053.732
157  rtREV         11122.791    11  22267.581    22267.659    22335.102
158  rtREV+I       11122.236    12  22268.473    22268.564    22342.132
159  rtREV+G4      11122.259    12  22268.517    22268.609    22342.177
160  rtREV+I+G4    11122.246    13  22270.491    22270.598    22350.289
161  rtREV+R2      11122.316    13  22270.633    22270.740    22350.431
162  rtREV+R3      11122.317    15  22274.633    22274.774    22366.707
170  rtREV+F       10917.610    30  21895.221    21895.769    22079.369
171  rtREV+F+I     10916.959    31  21895.918    21896.503    22086.204
172  rtREV+F+G4    10916.988    31  21895.975    21896.561    22086.262
173  rtREV+F+I+G4  10916.960    32  21897.919    21898.542    22094.344
174  rtREV+F+R2    10917.080    32  21898.160    21898.783    22094.585
175  rtREV+F+R3    10917.080    34  21902.160    21902.863    22110.861
183  mtART         11558.410    11  23138.821    23138.898    23206.342
184  mtART+I       11553.524    12  23131.049    23131.140    23204.708
185  mtART+G4      11553.549    12  23131.099    23131.190    23204.758
186  mtART+I+G4    11553.446    13  23132.892    23132.999    23212.690
187  mtART+R2      11556.628    13  23139.257    23139.364    23219.054
188  mtART+R3      11556.612    15  23143.223    23143.364    23235.297
196  mtART+F       10933.524    30  21927.048    21927.597    22111.196
197  mtART+F+I     10929.190    31  21920.381    21920.966    22110.667
198  mtART+F+G4    10929.190    31  21920.380    21920.965    22110.667
199  mtART+F+I+G4  10929.187    32  21922.375    21922.998    22118.800
200  mtART+F+R2    10931.760    32  21927.521    21928.144    22123.946
201  mtART+F+R3    10931.755    34  21931.510    21932.212    22140.211
209  mtZOA         11426.250    11  22874.500    22874.578    22942.021
210  mtZOA+I       11425.364    12  22874.727    22874.819    22948.386
211  mtZOA+G4      11425.373    12  22874.745    22874.837    22948.404
212  mtZOA+I+G4    11425.369    13  22876.737    22876.844    22956.535
213  mtZOA+R2      11425.626    13  22877.252    22877.359    22957.049
214  mtZOA+R3      11425.624    15  22881.248    22881.389    22973.323
222  mtZOA+F       10907.943    30  21875.886    21876.434    22060.034
223  mtZOA+F+I     10907.214    31  21876.428    21877.013    22066.715
224  mtZOA+F+G4    10907.233    31  21876.465    21877.050    22066.752
225  mtZOA+F+I+G4  10907.220    32  21878.439    21879.062    22074.864
226  mtZOA+F+R2    10907.392    32  21878.783    21879.406    22075.208
227  mtZOA+F+R3    10907.390    34  21882.781    21883.483    22091.482
235  VT            11028.624    11  22079.247    22079.324    22146.768
236  VT+I          11028.513    12  22081.026    22081.117    22154.685
237  VT+G4         11028.522    12  22081.045    22081.136    22154.704
238  VT+I+G4       11028.765    13  22083.530    22083.636    22163.327
239  VT+R2         11028.546    13  22083.092    22083.199    22162.890
240  VT+R3         11028.546    15  22087.092    22087.233    22179.166
248  VT+F          10909.800    30  21879.600    21880.149    22063.749
249  VT+F+I        10909.693    31  21881.386    21881.971    22071.672
250  VT+F+G4       10909.704    31  21881.407    21881.992    22071.694
251  VT+F+I+G4     10909.951    32  21883.903    21884.526    22080.327
252  VT+F+R2       10909.728    32  21883.456    21884.079    22079.881
253  VT+F+R3       10909.727    34  21887.455    21888.157    22096.156
261  LG            11023.793    11  22069.587    22069.664    22137.108
262  LG+I          11023.610    12  22071.221    22071.312    22144.880
263  LG+G4         11023.625    12  22071.249    22071.340    22144.908
264  LG+I+G4       11023.766    13  22073.533    22073.639    22153.330
265  LG+R2         11023.622    13  22073.244    22073.351    22153.041
266  LG+R3         11023.622    15  22077.243    22077.384    22169.317
274  LG+F          10896.262    30  21852.525    21853.073    22036.673
275  LG+F+I        10896.049    31  21854.097    21854.682    22044.384
276  LG+F+G4       10896.066    31  21854.131    21854.716    22044.417
277  LG+F+I+G4     10896.175    32  21856.350    21856.973    22052.774
278  LG+F+R2       10896.059    32  21856.117    21856.740    22052.542
279  LG+F+R3       10896.059    34  21860.117    21860.820    22068.818
287  DCMut         11111.703    11  22245.405    22245.483    22312.926
288  DCMut+I       11111.264    12  22246.527    22246.619    22320.186
289  DCMut+G4      11111.276    12  22246.552    22246.644    22320.211
290  DCMut+I+G4    11111.275    13  22248.550    22248.656    22328.347
291  DCMut+R2      11111.314    13  22248.627    22248.734    22328.425
292  DCMut+R3      11111.313    15  22252.627    22252.768    22344.701
300  DCMut+F       10918.625    30  21897.251    21897.799    22081.399
301  DCMut+F+I     10918.304    31  21898.608    21899.193    22088.894
302  DCMut+F+G4    10918.318    31  21898.635    21899.220    22088.921
303  DCMut+F+I+G4  10918.352    32  21900.704    21901.327    22097.129
304  DCMut+F+R2    10918.322    32  21900.643    21901.266    22097.068
305  DCMut+F+R3    10918.322    34  21904.643    21905.346    22113.344
313  PMB           11023.693    11  22069.386    22069.464    22136.907
314  PMB+I         11023.547    12  22071.094    22071.185    22144.753
315  PMB+G4        11023.560    12  22071.121    22071.212    22144.780
316  PMB+I+G4      11023.739    13  22073.478    22073.585    22153.275
317  PMB+R2        11023.566    13  22073.131    22073.238    22152.929
318  PMB+R3        11023.565    15  22077.131    22077.272    22169.205
326  PMB+F         10934.528    30  21929.055    21929.603    22113.203
327  PMB+F+I       10934.374    31  21930.749    21931.334    22121.035
328  PMB+F+G4      10934.389    31  21930.778    21931.363    22121.065
329  PMB+F+I+G4    10934.557    32  21933.114    21933.737    22129.539
330  PMB+F+R2      10934.392    32  21932.783    21933.406    22129.208
331  PMB+F+R3      10934.391    34  21936.783    21937.485    22145.484
339  HIVb          10987.685    11  21997.369    21997.447    22064.890
340  HIVb+I        10986.712    12  21997.425    21997.516    22071.084
341  HIVb+G4       10986.757    12  21997.514    21997.605    22071.173
342  HIVb+I+G4     10986.724    13  21999.448    21999.555    22079.246
343  HIVb+R2       10986.963    13  21999.927    22000.034    22079.725
344  HIVb+R3       10986.962    15  22003.925    22004.065    22095.999
352  HIVb+F        10861.344    30  21782.687    21783.236    21966.835
353  HIVb+F+I      10860.273    31  21782.546    21783.131    21972.833
354  HIVb+F+G4     10860.322    31  21782.645    21783.230    21972.931
355  HIVb+F+I+G4   10860.293    32  21784.586    21785.209    21981.011
356  HIVb+F+R2     10860.579    32  21785.159    21785.782    21981.584
357  HIVb+F+R3     10860.578    34  21789.156    21789.858    21997.857
365  HIVw          11235.035    11  22492.069    22492.147    22559.590
366  HIVw+I        11232.841    12  22489.682    22489.773    22563.341
367  HIVw+G4       11232.879    12  22489.758    22489.850    22563.418
368  HIVw+I+G4     11232.891    13  22491.782    22491.889    22571.580
369  HIVw+R2       11233.798    13  22493.595    22493.702    22573.393
370  HIVw+R3       11233.794    15  22497.588    22497.729    22589.662
378  HIVw+F        10867.886    30  21795.772    21796.320    21979.920
379  HIVw+F+I      10865.804    31  21793.607    21794.192    21983.894
380  HIVw+F+G4     10865.843    31  21793.685    21794.270    21983.972
381  HIVw+F+I+G4   10865.854    32  21795.707    21796.330    21992.132
382  HIVw+F+R2     10866.696    32  21797.391    21798.014    21993.816
383  HIVw+F+R3     10866.692    34  21801.383    21802.086    22010.085
391  JTTDCMut      10967.512    11  21957.025    21957.102    22024.546
392  JTTDCMut+I    10967.371    12  21958.741    21958.833    22032.401
393  JTTDCMut+G4   10967.380    12  21958.761    21958.852    22032.420
394  JTTDCMut+I+G4 10967.573    13  21961.147    21961.253    22040.944
395  JTTDCMut+R2   10967.390    13  21960.779    21960.886    22040.577
396  JTTDCMut+R3   10967.389    15  21964.777    21964.918    22056.852
404  JTTDCMut+F    10877.432    30  21814.865    21815.413    21999.013
405  JTTDCMut+F+I  10877.292    31  21816.585    21817.170    22006.871
406  JTTDCMut+F+G4 10877.304    31  21816.608    21817.193    22006.894
407  JTTDCMut+F+I+G4 10877.498    32  21818.996    21819.619    22015.420
408  JTTDCMut+F+R2 10877.313    32  21818.625    21819.248    22015.050
409  JTTDCMut+F+R3 10877.312    34  21822.623    21823.326    22031.325
417  FLU           11060.519    11  22143.038    22143.115    22210.559
418  FLU+I         11060.137    12  22144.275    22144.366    22217.934
419  FLU+G4        11060.172    12  22144.343    22144.435    22218.002
420  FLU+I+G4      11060.179    13  22146.357    22146.464    22226.155
421  FLU+R2        11060.173    13  22146.346    22146.453    22226.144
422  FLU+R3        11060.173    15  22150.346    22150.487    22242.420
430  FLU+F         10855.218    30  21770.436    21770.985    21954.584
431  FLU+F+I       10854.816    31  21771.632    21772.217    21961.918
432  FLU+F+G4      10854.857    31  21771.714    21772.299    21962.000
433  FLU+F+I+G4    10854.851    32  21773.701    21774.324    21970.126
434  FLU+F+R2      10854.860    32  21773.719    21774.342    21970.144
435  FLU+F+R3      10854.860    34  21777.719    21778.422    21986.420
443  Blosum62      11034.203    11  22090.407    22090.484    22157.928
444  Blosum62+I    11034.061    12  22092.122    22092.213    22165.781
445  Blosum62+G4   11034.073    12  22092.145    22092.237    22165.805
446  Blosum62+I+G4 11034.257    13  22094.513    22094.620    22174.311
447  Blosum62+R2   11034.080    13  22094.159    22094.266    22173.957
448  Blosum62+R3   11034.079    15  22098.159    22098.300    22190.233
456  Blosum62+F    10929.846    30  21919.693    21920.241    22103.841
457  Blosum62+F+I  10929.687    31  21921.375    21921.960    22111.661
458  Blosum62+F+G4 10929.701    31  21921.401    21921.986    22111.688
459  Blosum62+F+I+G4 10929.860    32  21923.721    21924.344    22120.145
460  Blosum62+F+R2 10929.702    32  21923.404    21924.027    22119.829
461  Blosum62+F+R3 10929.702    34  21927.404    21928.106    22136.105
469  mtMet         11637.321    11  23296.643    23296.720    23364.164
470  mtMet+I       11636.906    12  23297.812    23297.904    23371.472
471  mtMet+G4      11636.915    12  23297.830    23297.921    23371.489
472  mtMet+I+G4    11636.925    13  23299.851    23299.958    23379.648
473  mtMet+R2      11636.946    13  23299.892    23299.999    23379.690
474  mtMet+R3      11636.946    15  23303.892    23304.033    23395.966
482  mtMet+F       10887.603    30  21835.207    21835.755    22019.355
483  mtMet+F+I     10887.110    31  21836.219    21836.804    22026.506
484  mtMet+F+G4    10887.123    31  21836.246    21836.831    22026.533
485  mtMet+F+I+G4  10887.114    32  21838.228    21838.851    22034.652
486  mtMet+F+R2    10887.181    32  21838.362    21838.985    22034.786
487  mtMet+F+R3    10887.181    34  21842.361    21843.064    22051.062
495  mtVer         11581.317    11  23184.633    23184.711    23252.154
496  mtVer+I       11580.779    12  23185.558    23185.649    23259.217
497  mtVer+G4      11580.776    12  23185.552    23185.643    23259.211
498  mtVer+I+G4    11580.781    13  23187.561    23187.668    23267.359
499  mtVer+R2      11580.858    13  23187.716    23187.823    23267.514
500  mtVer+R3      11580.858    15  23191.716    23191.857    23283.790
508  mtVer+F       10874.965    30  21809.931    21810.479    21994.079
509  mtVer+F+I     10874.491    31  21810.981    21811.567    22001.268
510  mtVer+F+G4    10874.496    31  21810.993    21811.578    22001.279
511  mtVer+F+I+G4  10874.498    32  21812.996    21813.619    22009.421
512  mtVer+F+R2    10874.550    32  21813.100    21813.723    22009.524
513  mtVer+F+R3    10874.550    34  21817.099    21817.802    22025.800
521  mtInv         11852.767    11  23727.533    23727.611    23795.054
522  mtInv+I       11852.378    12  23728.756    23728.848    23802.415
523  mtInv+G4      11852.392    12  23728.785    23728.876    23802.444
524  mtInv+I+G4    11852.403    13  23730.807    23730.914    23810.604
525  mtInv+R2      11852.412    13  23730.825    23730.931    23810.622
526  mtInv+R3      11852.412    15  23734.824    23734.965    23826.898
534  mtInv+F       10902.563    30  21865.126    21865.674    22049.274
535  mtInv+F+I     10902.146    31  21866.293    21866.878    22056.579
536  mtInv+F+G4    10902.161    31  21866.323    21866.908    22056.609
537  mtInv+F+I+G4  10902.164    32  21868.327    21868.950    22064.752
538  mtInv+F+R2    10902.190    32  21868.379    21869.002    22064.804
539  mtInv+F+R3    10902.189    34  21872.379    21873.081    22081.080
Akaike Information Criterion:           FLU+F
Corrected Akaike Information Criterion: FLU+F
Bayesian Information Criterion:         FLU+F
Best-fit model: FLU+F chosen according to BIC

All model information printed to FASTA.fa.model.gz
CPU time for ModelFinder: 0.900 seconds (0h:0m:0s)
Wall-clock time for ModelFinder: 1.092 seconds (0h:0m:1s)

NOTE: 0 MB RAM (0 GB) is required!
Estimate model parameters (epsilon = 0.100)
1. Initial log-likelihood: -10855.218
Optimal log-likelihood: -10855.218
Parameters optimization took 1 rounds (0.001 sec)
Computing ML distances based on estimated model parameters... 0.001 sec
Computing BIONJ tree...
0.000 seconds
Log-likelihood of BIONJ tree: -10855.219
--------------------------------------------------------------------
|             INITIALIZING CANDIDATE TREE SET                      |
--------------------------------------------------------------------
Generating 98 parsimony trees... 0.016 second
Computing log-likelihood of 65 initial trees ... 0.028 seconds
Current best score: -10855.216

Do NNI search on 20 best initial trees
Estimate model parameters (epsilon = 0.100)
BETTER TREE FOUND at iteration 1: -10855.216
Iteration 10 / LogL: -10855.216 / Time: 0h:0m:0s
Iteration 20 / LogL: -10855.216 / Time: 0h:0m:0s
Finish initializing candidate tree set (7)
Current best tree score: -10855.216 / CPU time: 0.087
Number of iterations: 20
--------------------------------------------------------------------
|               OPTIMIZING CANDIDATE TREE SET                      |
--------------------------------------------------------------------
Iteration 30 / LogL: -10855.216 / Time: 0h:0m:0s (0h:0m:0s left)
Iteration 40 / LogL: -10855.216 / Time: 0h:0m:0s (0h:0m:0s left)
Iteration 50 / LogL: -10855.216 / Time: 0h:0m:0s (0h:0m:0s left)
Iteration 60 / LogL: -10855.216 / Time: 0h:0m:0s (0h:0m:0s left)
UPDATE BEST LOG-LIKELIHOOD: -10855.216
UPDATE BEST LOG-LIKELIHOOD: -10855.216
Iteration 70 / LogL: -10855.216 / Time: 0h:0m:0s (0h:0m:0s left)
UPDATE BEST LOG-LIKELIHOOD: -10855.216
Iteration 80 / LogL: -10855.216 / Time: 0h:0m:0s (0h:0m:0s left)
UPDATE BEST LOG-LIKELIHOOD: -10855.216
Iteration 90 / LogL: -10855.216 / Time: 0h:0m:0s (0h:0m:0s left)
UPDATE BEST LOG-LIKELIHOOD: -10855.216
Iteration 100 / LogL: -10855.216 / Time: 0h:0m:0s (0h:0m:0s left)
TREE SEARCH COMPLETED AFTER 102 ITERATIONS / Time: 0h:0m:0s

--------------------------------------------------------------------
|                    FINALIZING TREE SEARCH                        |
--------------------------------------------------------------------
Performs final model parameters optimization
Estimate model parameters (epsilon = 0.010)
1. Initial log-likelihood: -10855.216
Optimal log-likelihood: -10855.216
Parameters optimization took 1 rounds (0.001 sec)
BEST SCORE FOUND : -10855.216
Total tree length: 0.043

Total number of iterations: 102
CPU time used for tree search: 0.206 sec (0h:0m:0s)
Wall-clock time used for tree search: 0.263 sec (0h:0m:0s)
Total CPU time used: 0.214 sec (0h:0m:0s)
Total wall-clock time used: 0.272 sec (0h:0m:0s)

Analysis results written to: 
  IQ-TREE report:                FASTA.fa.iqtree
  Maximum-likelihood tree:       FASTA.fa.treefile
  Likelihood distances:          FASTA.fa.mldist
  Screen log file:               FASTA.fa.log

Date and Time: Sat Dec  9 09:41:12 2023
```

Best model: FLU+F, not Dayhoff as the one used to train the model.
FLU is a model for influenza virus [Dang et al, 2010](https://bmcecolevol.biomedcentral.com/articles/10.1186/1471-2148-10-99). 
`+F` simply means "Empirical base frequencies" which is the default anyway.

## Comparing the different trees

In R:

Tree estimated by us:
```r
library(ape)
tre = read.tree("phylo-nn-googledrive/outputs/Real_Data/MXCNtree.dat")

## Root at 1:
tr <- root(tre, 1, resolve.root=TRUE)

## Changing tips to taxon names (from fasta-table.csv)
new_tiplabels <- c("Mus_musculus", "Simiiformes_Cambodia", "Macaca_mulatta", "Homo_sapiens_Brazil", "Simiiformes_Uganda", "Homo_sapiens_Honduras")
tr$tip.label <- new_tiplabels

plot(tr)
```

Tree by IQ-Tree:
```r
tre2 = read.tree("data/FASTA.fa.treefile")

## Root at 1:
tr2 <- root(tre2, "AXF50052", resolve.root=TRUE)

new_tiplabels2 <- c("Mus_musculus", "Simiiformes_Cambodia", 
"Simiiformes_Uganda", "Homo_sapiens_Japan", "Macaca_mulatta", "Homo_sapiens_Honduras", "Homo_sapiens_Brazil")

tr2$tip.label <- new_tiplabels2

plot(tr2)
```

This tree is also weird. IQ-Tree has Simiiformes as the most derived clade, Humans do not form a clade, and Macaca is in between human samples.

## Tree estimation with 5 taxa

We manually create a new fasta (`FASTA2.fa`) with only 5 taxa:
```
1,>AXF50052 |Mus musculus|Colombia|2016/01/17
2,>AWW21402 |Simiiformes|Cambodia|2016/06/14
3,>AYI50274 |Macaca mulatta||2015/07/07
4,>QIH53581 |Homo sapiens|Brazil|2017/06/29
6,>ANG09399 |Homo sapiens|Honduras|2016/01/06
```

We have to make sure the file only has 5 lines, so we read in a script:

```julia
data = readlines("FASTA2.fa")

taxa = []
seqs = []

seq = ""
for l in data
   if occursin(">",l)
      push!(taxa,l)
      push!(seqs,seq) ##push previous seq
      seq = ""
   else
      seq *= l
   end
end
push!(seqs,seq) ##push last seq

## by the way it is constructed, we have an extra empty seq in seqs:
deleteat!(seqs, 1)
```

Now we have two vectors: `taxa` with the taxon names and `seqs` with the sequences. We had already manually changed taxon names to numbers in `FASTA2.fa`.

Now we write to file:

```julia
io = open("FASTA2-mod.fa", "w")

for j in 1:length(taxa)
   write(io, seqs[j])
   write(io, "\n")
end
close(io)
```
