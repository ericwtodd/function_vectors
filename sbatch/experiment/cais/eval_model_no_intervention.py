import os


if __name__ == "__main__":
    
    model_path = "/data/jiuding_sun/function_vectors/flan-llama-7b"
    save_path = "/data/jiuding_sun/function_vectors/results/flan-llama-7b"
    experiment_name = "flan-llama-7b_no_intervention"
    
    all_datasets = os.listdir("../../dataset_files/abstractive") + os.listdir("../../dataset_files/extractive")
    all_datasets = [i.split(".")[0] for i in all_datasets]
    print(all_datasets)
    
    all_sbatch_files = []
    
    if not os.path.exists(f"../scripts/{experiment_name}"):
        os.makedirs(f"../scripts/{experiment_name}")
        
    for dataset_name in all_datasets:
        with open(f"../scripts/{experiment_name}/{dataset_name}.sh", 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f"#SBATCH --job-name={experiment_name}\n")
            f.write(f"#SBATCH --output={experiment_name}-{dataset_name}.out\n")
            f.write("#SBATCH --nodes=1\n")
            f.write("#SBATCH --gpus-per-node=1\n")
            f.write("#SBATCH --mem=80GB\n")
            f.write("#SBATCH --time=1:00:00\n")
            f.write("\n\n")
            f.write("source /data/jiuding_sun/.bashrc\n")
            f.write("cd /data/jiuding_sun/function_vectors/src\n")
            f.write("conda activate fv\n")
            f.write(f"python evaluate.py --dataset_name {dataset_name} --model_name {model_path} --save_path_root {save_path}")
            f.close()
            
        all_sbatch_files.append(f"../scripts/{experiment_name}/{dataset_name}.sh")
        
    for f in all_sbatch_files:
        os.system(f"sbatch {f}")
            