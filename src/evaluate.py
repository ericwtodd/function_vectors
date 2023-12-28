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

    parser.add_argument('--dataset_name', help='Name of the dataset to be loaded', type=str, required=True, default="antonym")
    parser.add_argument('--model_name', help='Name of model to be loaded', type=str, required=False, default='../flan-llama-7b')
    parser.add_argument('--root_data_dir', help='Root directory of data files', type=str, required=False, default='../dataset_files')
    parser.add_argument('--save_path_root', help='File path to save mean activations to', type=str, required=False, default='../results/flan-llama-7b-INST')
    parser.add_argument('--seed', help='Randomized seed', type=int, required=False, default=42)
    parser.add_argument('--n_shots', help="Number of shots in each in-context prompt", required=False, default=0)
    parser.add_argument('--test_split', help="Percentage corresponding to test set split size", required=False, default=0.3)
    parser.add_argument('--device', help='Device to run on', required=False, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--prefixes', help='Prompt template prefixes to be used', type=json.loads, required=False, default={"input":"Q:", "output":"A:", "instructions":""})
    parser.add_argument('--separators', help='Prompt template separators to be used', type=json.loads, required=False, default={"input":"\n", "output":"\n\n", "instructions":""})  
    parser.add_argument('--metric', help='Metric to use for eval', type=str, required=False, default='exact_match_score', choices=['exact_match_score', 'f1_score', 'first_word_score'])
        
    args = parser.parse_args()
    
    dataset_name = args.dataset_name
    model_name = args.model_name
    root_data_dir = args.root_data_dir
    save_path_root = f"{args.save_path_root}/{dataset_name}"
    seed = args.seed
    n_shots = args.n_shots
    test_split = args.test_split
    device = args.device
    
    prefixes = load_prefixes_or_separators(args.prefixes) 
    separators = load_prefixes_or_separators(args.separators)
    
    metric = args.metric
    
    # Load Model & Tokenizer
    torch.set_grad_enabled(False)
    print("Loading Model")
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(model_name, device=device)

    set_seed(seed)

    # Load the dataset
    print("Loading Dataset")
    dataset = load_dataset(dataset_name, root_data_dir=root_data_dir, test_size=test_split, seed=seed)
    
    if not os.path.exists(save_path_root):
        os.makedirs(save_path_root)
    
    results = n_shot_eval_no_intervention(dataset=dataset, n_shots=n_shots, model=model, tokenizer=tokenizer, model_config=model_config, prefixes=prefixes, separators=separators, metric=metric)
    
    json.dump(results, open(os.path.join(save_path_root, f"{metric}_results.json"), 'w'))
    print(f"Results saved to {save_path_root}")
    
    