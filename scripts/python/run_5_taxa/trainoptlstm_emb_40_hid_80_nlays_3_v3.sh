#!/usr/bin/env bash

#SBATCH --partition=research
#SBATCH --time=1-1:0:0
#SBATCH --mem 16000
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

module load anaconda/full/2021.05
bootstrap_conda
conda activate pytorch

python ../network_zou/network_perm_eq_lstm_5_taxa_v3.py trainoptlstm_emb_40_hid_80_nlays_3.json
