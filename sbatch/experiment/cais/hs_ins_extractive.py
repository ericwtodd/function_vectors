import os
import json


if __name__ == "__main__":
    

    flan_model_path = "/data/jiuding_sun/function_vectors/flan-llama-7b"
    
    hs_save_path = "/data/jiuding_sun/function_vectors/results/INS/HS/flan-llama-7b"
        
    experiment_name = "hs-ins-extractive"
    
    all_datasets = ["alphabetically_last_3", "animal_v_object_3", "color_v_animal_3", "concept_v_object_5"]
    print(all_datasets)
    
    all_sbatch_files = []
    
    if not os.path.exists(f"../../scripts/{experiment_name}"):
        os.makedirs(f"../../scripts/{experiment_name}")
        
    for dataset_name in all_datasets:
        
        for prompt_idx in range(1, 6):
            
            prompt_save_path_root = os.path.join(hs_save_path, f"Prompt{prompt_idx}")
            
            prefixes_path = os.path.join("/data/jiuding_sun/function_vectors/templates_files/extractive", dataset_name, f"{prompt_idx}", "prefixes.json")
            separators_path = os.path.join("/data/jiuding_sun/function_vectors/templates_files/extractive", dataset_name, f"{prompt_idx}", "separators.json")
            
            with open(f"../../scripts/{experiment_name}/{dataset_name}_prompt{prompt_idx}.sh", 'w') as f:
                f.write("#!/bin/bash\n")
                f.write(f"#SBATCH --job-name={experiment_name}\n")
                f.write(f"#SBATCH --output={experiment_name}-{dataset_name}_{prompt_idx}.out\n")
                f.write("#SBATCH --nodes=1\n")
                f.write("#SBATCH --gpus-per-node=1\n")
                f.write("#SBATCH --mem=32GB\n")
                f.write("#SBATCH --time=12:00:00\n")
                
                f.write("\n\n")
                f.write("source /data/jiuding_sun/miniconda3/bin/activate\n")
                f.write("cd /data/jiuding_sun/function_vectors/src\n")
                f.write("conda activate fv\n")
                
                f.write("\n\n")
                f.write(f"python compute_avg_hidden_state.py ")
                f.write(f"--dataset_name {dataset_name} ")
                f.write(f"--model_name {flan_model_path} ")
                f.write(f"--save_path_root {prompt_save_path_root} ")
                f.write(f"--prefixes {prefixes_path} ")
                f.write(f"--separators {separators_path} ")
                f.close()
                
            all_sbatch_files.append(f"../../scripts/{experiment_name}/{dataset_name}_prompt{prompt_idx}.sh")
        
    for f in all_sbatch_files:
        os.system(f"sbatch {f}")
            