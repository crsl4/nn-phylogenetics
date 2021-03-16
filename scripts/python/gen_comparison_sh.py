import os 

name_script_old = "network_permutation_equivariant_shallow_new_lba.py"
name_script_new = "network_perm_equivariant_modular_conv_new_lba.py"


string = "#!/usr/bin/env bash \n\n#SBATCH --partition=batch_default\n#SBATCH --cpus-per-task=2\n#SBATCH --gres=gpu:1\n#SBATCH --time=48:00:00\n\nsource ~/.bashrc\nmodule load anaconda/wml\nbootstrap_conda\nmodule load cuda\n\nconda activate pytorch\n\nCUDA_VISIBLE_DEVICES=0 python "


for ii in [1, 4, 17, 22, 25]:
    for lr in [0.01, 0.001, 0.0001]:
        for bs in [16, 32, 64, 128]:

            name_json = "permutation_equivariant_new_lba_file_"+str(ii+1)+"_lr_"+str(lr)+"_bs_"+str(bs)+".json"
            
            program_str_old = string + " " + name_script_old + " " + name_json
            program_str_new = string + " " + name_script_new + " " + name_json

            name_sbatch_old = "old_permutation_equivariant_new_lba_file_"+str(ii+1)+"_lr_"+str(lr)+"_bs_"+str(bs)+".sh"
            name_sbatch_new = "new_permutation_equivariant_new_lba_file_"+str(ii+1)+"_lr_"+str(lr)+"_bs_"+str(bs)+".sh"

            with open(name_sbatch_old, "w") as file:
        
                file.write(program_str_old)

            with open(name_sbatch_new, "w") as file:
        
                file.write(program_str_new)