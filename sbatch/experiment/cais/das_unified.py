import os
import json


if __name__ == "__main__":
    
    all_layers = list(range(0, 32))
    
    experiment_name = "DAS_Unified"
    
    all_sbatch_files = []
    
    if not os.path.exists(f"../../scripts/{experiment_name}"):
        os.makedirs(f"../../scripts/{experiment_name}")
        
        
    for layer in all_layers:
        
        with open(f"../../scripts/{experiment_name}/L{layer}.sh", 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f"#SBATCH --job-name={experiment_name}\n")
            f.write(f"#SBATCH --output={experiment_name}-L{layer}.out\n")
            f.write("#SBATCH --nodes=1\n")
            f.write("#SBATCH --gpus-per-node=1\n")
            f.write("#SBATCH --mem=32GB\n")
            f.write("#SBATCH --time=4:00:00\n")
            
            f.write("\n\n")
            f.write("source /data/jiuding_sun/miniconda3/bin/activate\n")
            f.write("cd /data/jiuding_sun/function_vectors/src\n")
            f.write("conda activate fv\n")
            f.write("\n\n")
            
            f.write(f"python compute_rotational_subspace_all_at_once_hell_yea.py --edit_layer {layer}")
            f.close()
        
        all_sbatch_files.append(f"../../scripts/{experiment_name}/L{layer}.sh")


    for f in all_sbatch_files:
        os.system(f"sbatch {f}")
    