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

## Subproblem 2: based on Zou2019

We want to:
1. Replicate the work in Zou2019. They simulate sequences under CTMC and quartet gene trees, and then train a neural network model to classify sequences based on the quartet gene tree they came from. Maybe Leo's student can help us replicate these results
2. Improve the neural network model in Zou2019 (too naive)
3. Simulate training data in a more meaningful manner: instead of classifying sequences based on quartet gene tree they came from, we can classify sequences based on the quartet species tree they came from. This is more meaningful because ultimately biologists want to estimate species tree (not just gene tree). We will first simulate quartet gene trees from quartet species tree under the coalescent model, and then simulate sequences from the quartet gene tree under CTMC model. We will train a neural network model to classify sequences based on the quartet species tree they come from