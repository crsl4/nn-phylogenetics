import os 

name_script = "network_perm_equivariant_modular_conv_new_lba.py"

string = "#!/usr/bin/env bash \n\n#SBATCH --partition=batch_default\n#SBATCH --cpus-per-task=2\n#SBATCH --gres=gpu:1\n#SBATCH --time=48:00:00\n\nmodule load anaconda/wml\n./bootstrap_conda\nmodule load cuda\n\nconda activate pytorch\n\nCUDA_VISIBLE_DEVICES=0 python "


for ii in range(2):

	name_json = "permutation_equivariant_new_lba_"+str(ii+1)+".json"

	program_str = string + " " + name_script + " " + name_json

	name_sbatch = "permutation_equivariant_new_lba_"+str(ii+1)+".sh"

	with open(name_sbatch, "w") as file:
		
		file.write(program_str)
