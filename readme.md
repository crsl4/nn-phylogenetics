# Neural networks for phylogenetic inference

- Claudia Solis-Lemus
- Leonardo Zepeda-Nunez

## Motivation

See Erick Matsen awesome [post](https://matsen.fhcrc.org/general/2019/06/18/pt.html) on recent approaches to bayesian phylogenetic inference.

His group is already exploring Bayesian networks to interpolate posterior distributions, see [here](https://matsen.fhcrc.org/general/2018/12/05/sbn.html).

They are also exploring variational approximations to estimate the posterior, see [here](https://matsen.fhcrc.org/general/2019/08/24/vbpi.html)

But no one is trying to use neural networks!

## Subproblem 1 

Using neural network to improve computation speed on maximum likelihood inference by using a trained NN instead of numerical optimization of likelihood for a given tree (see `notebook.Rmd`).
This seems to be a standalone paper to show prospect.

## Subproblem 2
Based on Zou2019:
- validate their results on quartet gene tree
- extend their work by simulating sequences under both CTMC (gene tree) and coalescent model (species tree)