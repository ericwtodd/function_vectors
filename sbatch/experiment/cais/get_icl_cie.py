import os
import json


if __name__ == "__main__":
    
    model_path = "/data/jiuding_sun/function_vectors/flan-llama-7b"
    flan_model_path = "/data/jiuding_sun/function_vectors/flan-llama-7b"
    
    save_path = "/data/jiuding_sun/function_vectors/results/ICL"
    
    llama_cie_save_path = "/data/jiuding_sun/function_vectors/results/ICL/CIE/llama-7b"
    flan_cie_save_path = "/data/jiuding_sun/function_vectors/results/ICL/CIE/flan-llama-7b"
    
    llama_fv_save_path = "/data/jiuding_sun/function_vectors/results/ICL/FV/llama-7b"
    flan_fv_save_path = "/data/jiuding_sun/function_vectors/results/ICL/FV/flan-llama-7b"
    flan_with_llama_fv_save_path = "/data/jiuding_sun/function_vectors/results/ICL/FV/flan-llama-7b_with_llama_fv"
    flan_with_llama_ie_save_path = "/data/jiuding_sun/function_vectors/results/ICL/FV/flan-llama-7b_with_llama_ie"
    
    experiment_name = "icl-exp-10"
    
    all_datasets = os.listdir("../../../dataset_files/abstractive") + os.listdir("../../../dataset_files/extractive")
    all_datasets = [i.split(".")[0] for i in all_datasets]
    print(all_datasets)
    
    all_sbatch_files = []
    
    if not os.path.exists(f"../../scripts/{experiment_name}"):
        os.makedirs(f"../../scripts/{experiment_name}")
        
    for dataset_name in all_datasets:
        
        llama_mean_activations_path = os.path.join(llama_cie_save_path, dataset_name, f"{dataset_name}_mean_head_activations.pt")
        llama_indirect_effect_path = os.path.join(llama_cie_save_path, dataset_name, f"{dataset_name}_indirect_effect.pt")
        
        flan_mean_activations_path = os.path.join(flan_cie_save_path, dataset_name, f"{dataset_name}_mean_head_activations.pt")
        flan_indirect_effect_path = os.path.join(flan_cie_save_path, dataset_name, f"{dataset_name}_indirect_effect.pt")
        
        with open(f"../../scripts/{experiment_name}/{dataset_name}.sh", 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f"#SBATCH --job-name={experiment_name}\n")
            f.write(f"#SBATCH --output={experiment_name}-{dataset_name}.out\n")
            f.write("#SBATCH --nodes=1\n")
            f.write("#SBATCH --gpus-per-node=1\n")
            f.write("#SBATCH --mem=80GB\n")
            f.write("#SBATCH --time=12:00:00\n")
            f.write("\n\n")
            f.write("source /data/jiuding_sun/.bashrc\n")
            f.write("cd /data/jiuding_sun/function_vectors/src\n")
            f.write("conda activate fv\n")
            f.write("\n\n")
            f.write(f"python compute_indirect_effect.py ")
            f.write(f"--dataset_name {dataset_name} ")
            f.write(f"--model_name {model_path} ")
            f.write(f"--save_path_root {llama_cie_save_path} ")
            f.write(f"&&\n")
            f.write(f"python compute_indirect_effect.py ")
            f.write(f"--dataset_name {dataset_name} ")
            f.write(f"--model_name {flan_model_path} ")
            f.write(f"--save_path_root {flan_cie_save_path} ")
            f.write(f"&&\n")
            f.write(f"python evaluate_function_vector.py ")
            f.write(f"--dataset_name {dataset_name} ")
            f.write(f"--model_name {model_path} ")
            f.write(f"--save_path_root {llama_fv_save_path} ")
            f.write(f"--mean_activations_path {llama_mean_activations_path} ")
            f.write(f"--indirect_effect_path {llama_indirect_effect_path} ")            
            f.write(f"&&\n")
            f.write(f"python evaluate_function_vector.py ")
            f.write(f"--dataset_name {dataset_name} ")
            f.write(f"--model_name {flan_model_path} ")
            f.write(f"--save_path_root {flan_fv_save_path} ")
            f.write(f"--mean_activations_path {flan_mean_activations_path} ")
            f.write(f"--indirect_effect_path {flan_indirect_effect_path} ")            
            f.write(f"&&\n")
            f.write(f"--dataset_name {dataset_name} ")
            f.write(f"--model_name {flan_model_path} ")
            f.write(f"--save_path_root {flan_with_llama_fv_save_path} ")
            f.write(f"--mean_activations_path {llama_mean_activations_path} ")
            f.write(f"--indirect_effect_path {llama_indirect_effect_path} ")            
            f.write(f"&&\n")
            f.write(f"--dataset_name {dataset_name} ")
            f.write(f"--model_name {flan_model_path} ")
            f.write(f"--save_path_root {flan_with_llama_ie_save_path} ")
            f.write(f"--mean_activations_path {flan_mean_activations_path} ")
            f.write(f"--indirect_effect_path {llama_indirect_effect_path} ")            
            f.write(f"&&\n")
            
            f.close()
            
        all_sbatch_files.append(f"../../scripts/{experiment_name}/{dataset_name}.sh")
        
    for f in all_sbatch_files:
        os.system(f"sbatch {f}")
            