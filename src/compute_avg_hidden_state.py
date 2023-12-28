import os, json
import torch, numpy as np
import argparse

# Include prompt creation helper functions
from utils.prompt_utils import *
from utils.intervention_utils import *
from utils.model_utils import *
from utils.eval_utils import *
from utils.extract_utils import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', help='Name of the dataset to be loaded', type=str, required=True)
    parser.add_argument('--model_name', help='Name of model to be loaded', type=str, required=False, default='EleutherAI/gpt-j-6b')
    parser.add_argument('--root_data_dir', help='Root directory of data files', type=str, required=False, default='../dataset_files')
    parser.add_argument('--save_path_root', help='File path to save mean activations to', type=str, required=False, default='../results')
    parser.add_argument('--n_seeds', help='Number of seeds', type=int, required=False, default=5)
    parser.add_argument('--n_shots', help="Number of shots in each in-context prompt", required=False, default=10)
    parser.add_argument('--n_trials', help="Number of in-context prompts to average over", required=False, default=100)
    parser.add_argument('--test_split', help="Percentage corresponding to test set split size", required=False, default=0.3)
    parser.add_argument('--device', help='Device to run on', required=False, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--prefixes', help='Prompt template prefixes to be used', type=json.loads, required=False, default={"input":"Q:", "output":"A:", "instructions":""})
    parser.add_argument('--separators', help='Prompt template separators to be used', type=json.loads, required=False, default={"input":"\n", "output":"\n\n", "instructions":""})    
        
    
    args = parser.parse_args()
    
    dataset_name = args.dataset_name
    model_name = args.model_name
    root_data_dir = args.root_data_dir
    save_path_root = f"{args.save_path_root}/{dataset_name}"
    n_seeds = args.n_seeds
    n_shots = args.n_shots
    n_trials = args.n_trials
    test_split = args.test_split
    device = args.device
    
    prefixes = load_prefixes_or_separators(args.prefixes)
    separators = load_prefixes_or_separators(args.separators)
    

    # Load Model & Tokenizer
    torch.set_grad_enabled(False)
    print("Loading Model")
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(model_name, device=device)
    
    seeds = np.random.choice(100000, size=n_seeds)
    
    for seed in seeds:
        set_seed(seed)

        # Load the dataset
        print("Loading Dataset")
        dataset = load_dataset(dataset_name, root_data_dir=root_data_dir, test_size=test_split, seed=seed)

        print("Computing Mean Activations")
        dataset = load_dataset(dataset_name, seed=seed)
        mean_activations = get_mean_layer_activations(dataset, model=model, model_config=model_config, tokenizer=tokenizer, 
                                                     n_icl_examples=n_shots, N_TRIALS=n_trials)


        print("Saving mean layer activations")
        if not os.path.exists(save_path_root):
            os.makedirs(save_path_root)

        # Write args to file
        args.save_path_root = save_path_root # update for logging
        with open(f'{save_path_root}/mean_layer_activation_args.txt', 'w') as arg_file:
            json.dump(args.__dict__, arg_file, indent=2)

        torch.save(mean_activations, f'{save_path_root}/{dataset_name}_mean_layer_activations.pt')

        print("Evaluating Layer Avgs. Baseline")
        fs_results = n_shot_eval_no_intervention(dataset, n_shots, model, model_config, tokenizer)
        filter_set = np.where(np.array(fs_results['clean_rank_list']) == 0)[0]

        zs_res = {}
        fss_res = {}
        for i in range(model_config['n_layers']):
            zs_res[i] = n_shot_eval(dataset, mean_activations[i].unsqueeze(0), i, 0, model, model_config, tokenizer, filter_set=filter_set)
            fss_res[i] = n_shot_eval(dataset, mean_activations[i].unsqueeze(0), i, 10, model, model_config, tokenizer, filter_set=filter_set, shuffle_labels=True)

        with open(f'{save_path_root}/mean_layer_intervention_zs_results_sweep_{seed}.json', 'w') as interv_zsres_file:
            json.dump(zs_res, interv_zsres_file, indent=2)
        with open(f'{save_path_root}/mean_layer_intervention_fss_results_sweep_{seed}.json', 'w') as interv_fssres_file:
            json.dump(fss_res, interv_fssres_file, indent=2)
