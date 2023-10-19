import os
import time
import numpy as np

# Submit slurm jobs for many tasks

dataset_names = ['antonym', 'capitalize', 'country-capital', 'english-french', 'present-past', 'singular-plural']
MODEL_NAMES = ["EleutherAI/gpt-j-6b"]
MODEL_NICKNAMES = ['gptj']


job_path = str(time.ctime()).replace(" ", "_")
print(job_path)
os.makedirs(job_path, exist_ok=True)

d_name_to_cmd = {}

## creating the jobs
for model_name,model_nickname in zip(MODEL_NAMES, MODEL_NICKNAMES):
    current_seed = np.random.randint(1000000)
    for idx, d_name in enumerate(dataset_names):
        results_path = os.path.join('results', f'{model_nickname}')
        n_fv_heads = 10

        cmd = f"python evaluate_function_vector.py --dataset_name='{d_name}' --save_path_root='{results_path}' --model_name='{model_name}' --n_top_heads={n_fv_heads} --seed={current_seed}"
        if 'squad' in d_name:
            cmd += " --n_shots=5 --generate_str --metric='f1_score'"
        elif 'ag_news' in d_name:
            cmd += " --n_shots=10 --generate_str --metric='first_word_score'"

        key = model_nickname + '_' + d_name
        d_name_to_cmd[key] = cmd


for key in d_name_to_cmd:
    with open("template.sh", "r") as f:
        bash_template = f.readlines()
        bash_template.append(d_name_to_cmd[key])

    with open(f"{job_path}/{key}.sh", "w") as f:
        f.writelines(bash_template)


## running the jobs
for job in os.listdir(job_path):
    job_script = f"{job_path}/{job}"
    cmd = f"sbatch --gpus=1 --time=48:00:00 {job_script}"
    print("submitting job: ", job)
    print(cmd)
    os.system(cmd)
    print("\n\n")

print("------------------------------------------------------------------")
print(f"submitted {len(os.listdir(job_path))} jobs!")
