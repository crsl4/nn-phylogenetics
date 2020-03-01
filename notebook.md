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