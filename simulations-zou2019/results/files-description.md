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