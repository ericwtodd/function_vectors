import os, json
import torch, numpy as np
import argparse

# Include prompt creation helper functions
from utils.prompt_utils import *
from utils.intervention_utils import *
from utils.model_utils import *
from utils.extract_utils import *



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', help='Name of the dataset to be loaded', type=str, required=True)
    parser.add_argument('--model_name', help='Name of model to be loaded', type=str, required=False, default='EleutherAI/gpt-j-6b')
    parser.add_argument('--root_data_dir', help='Root directory of data files', type=str, required=False, default='../dataset_files')
    parser.add_argument('--save_path_root', help='File path to save mean activations to', type=str, required=False, default='../results')
    parser.add_argument('--seed', help='Randomized seed', type=int, required=False, default=42)
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
    seed = args.seed
    n_shots = args.n_shots
    n_trials = args.n_trials
    test_split = args.test_split
    device = args.device
    prefixes = args.prefixes
    separators = args.separators
    

    # Load Model & Tokenizer
    torch.set_grad_enabled(False)
    print("Loading Model")
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(model_name, device=device)

    set_seed(seed)

    # Load the dataset
    print("Loading Dataset")
    dataset = load_dataset(dataset_name, root_data_dir=root_data_dir, test_size=test_split, seed=seed)

    print("Computing Mean Activations")
    mean_activations = get_mean_head_activations(dataset, model=model, model_config=model_config, tokenizer=tokenizer, 
                                                 n_icl_examples=n_shots, N_TRIALS=n_trials, prefixes=prefixes, separators=separators)

    if not os.path.exists(save_path_root):
        os.makedirs(save_path_root)

    # Write args to file
    args.save_path_root = save_path_root # update for logging
    with open(f'{save_path_root}/mean_head_activation_args.txt', 'w') as arg_file:
        json.dump(args.__dict__, arg_file, indent=2)
    
    torch.save(mean_activations, f'{save_path_root}/{dataset_name}_mean_head_activations.pt')
    
