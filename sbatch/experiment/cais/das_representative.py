import os
import json


if __name__ == "__main__":
    all_datasets = ["antonym", "capitalize", "country-currency", "english-french", "present-past", "singular-plural"]
    
    all_layers = list(range(0, 32, 3))
    
    experiment_name = "DAS_ICL"
    
    all_sbatch_files = []
    
    if not os.path.exists(f"../../scripts/{experiment_name}"):
        os.makedirs(f"../../scripts/{experiment_name}")
        
    for dataset_name in all_datasets:
        
        for layer in all_layers:
            
            with open(f"../../scripts/{experiment_name}/{dataset_name}_{layer}.sh", 'w') as f:
                f.write("#!/bin/bash\n")
                f.write(f"#SBATCH --job-name={experiment_name}\n")
                f.write(f"#SBATCH --output={experiment_name}-{dataset_name}.out\n")
                f.write("#SBATCH --nodes=1\n")
                f.write("#SBATCH --gpus-per-node=1\n")
                f.write("#SBATCH --mem=32GB\n")
                f.write("#SBATCH --time=4:00:00\n")
                
                f.write("\n\n")
                f.write("source /data/jiuding_sun/miniconda3/bin/activate\n")
                f.write("cd /data/jiuding_sun/function_vectors/src\n")
                f.write("conda activate fv\n")
                f.write("\n\n")
                
                f.write(f"python compute_rotational_subspace.py --dataset_name {dataset_name} --edit_layer {layer}")
                f.close()
            
            all_sbatch_files.append(f"../../scripts/{experiment_name}/{dataset_name}_{layer}.sh")


    for f in all_sbatch_files:
        os.system(f"sbatch {f}")
    