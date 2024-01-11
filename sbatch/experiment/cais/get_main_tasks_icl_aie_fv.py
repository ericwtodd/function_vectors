import os
import json


if __name__ == "__main__":
    
    model_path = "/data/public_models/llama/llama_hf_weights/llama-7b"
    
    save_path = "/data/jiuding_sun/function_vectors/results/ICL/FV-AIE"
    
    aie_path = "/data/jiuding_sun/function_vectors/results/ICL/CIE/llama-7b/main_6_tasks/average_indirect_effect.pt"
    
    llama_fv_save_path = "/data/jiuding_sun/function_vectors/results/ICL/FV-AIE/llama-7b"
    
    experiment_name = "icl-aie-main"
    
    all_datasets = ["antonym", "capitalize", "country-currency", "english-french", "present-past", "singular-plural"]
    print(all_datasets)
    
    all_sbatch_files = []
    
    if not os.path.exists(f"../../scripts/{experiment_name}"):
        os.makedirs(f"../../scripts/{experiment_name}")
        
    for dataset_name in all_datasets:
        
        llama_mean_activations_path = f"../results/ICL/CIE/llama-7b/{dataset_name}/{dataset_name}_mean_head_activations.pt"
        
        with open(f"../../scripts/{experiment_name}/{dataset_name}.sh", 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f"#SBATCH --job-name={experiment_name}\n")
            f.write(f"#SBATCH --output={experiment_name}-{dataset_name}.out\n")
            f.write("#SBATCH --nodes=1\n")
            f.write("#SBATCH --gpus-per-node=1\n")
            f.write("#SBATCH --mem=32GB\n")
            f.write("#SBATCH --time=8:00:00\n")
            f.write("\n\n")
            f.write("source /data/jiuding_sun/.bashrc\n")
            f.write("cd /data/jiuding_sun/function_vectors/src\n")
            f.write("conda activate fv\n")
            f.write("\n\n")
            
            f.write(f"python evaluate_function_vector.py ")
            f.write(f"--dataset_name {dataset_name} ")
            f.write(f"--model_name {model_path} ")
            f.write(f"--save_path_root {llama_fv_save_path} ")
            f.write(f"--mean_activations_path {llama_mean_activations_path} ")
            f.write(f"--indirect_effect_path {aie_path} ")            
            f.close()
            
        all_sbatch_files.append(f"../../scripts/{experiment_name}/{dataset_name}.sh")
        
    for f in all_sbatch_files:
        os.system(f"sbatch {f}")
            