# New description (Sept 2020)
We don't do the one-hot encoding now so that we do this step on the fly in python (and we can have larger sample sizes in simulations).

So, we have 27 LBA scenarios, each identified by a random seed, and each with 100,000 samples. Each sample has 4 taxa and sequence of length 1550.

For example, the LBA scenario: b=0.1, a=2b, c=0.01b corresponds to the random seed: 4738282 and the files are:
- `labels4738282-0.1-2.0-0.01.in` which is a 100,000 x 1 vector with the quartet type
- `sequences4738282-0.1-2.0-0.01.in` is a 400,000 x 1550 matrix. Appended 4x1550 matrices per sample (e.g. first 4 rows correspond to sample 1). This file has the actual sequences (no encoding)

# Old description

Training set of 100,000 replicates (as in Zou2019) ran in 14 batches:
- batch i=1,2,3,4 with 5000 replicates each (20,000 total)
- batch i=5,...,14 with 8000 replicates each (80,000 total)

Each batch produces two files: 
- labels file (with tree label that generated the sequences: three labels 1,2,3)
- input data file 80x1550 with the vectorized one-hot encoding tensor of the protein sequences: 20 aminoacid sequences of length 1550 for 4 taxa => tensor 4x1550x20 vectorized into a matrix 80x1550 (same format as in Zou2019)

- `labels-i.h5` (i=1,...,14): n-dimensional vector with labels for n replicates; files i=1,2,3,4 have 5000 replicates each (20,000 total) and files i=5,...,14 have 8000 replicates each (80,000 total) => 20k+80k=100k as in Zou2019

- `matrices-i.h5` (i=1,...,14): 80x1550 input matrix per replicate, all matrices are stacked one on top of the other; 
  - files i=1,2,3,4 have 5000 replicates and matrices are stacked on top of each other => (80 * 5000)x1550 matrix; 
  - files i=5,...,14 have 8000 replicates each  => (80 * 8000)x1550 matrix