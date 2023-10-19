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
    parser.add_argument('--n_eval_templates', help='Number of templates to evaluate with', required=True, type=int, default=15)
    parser.add_argument('--edit_layer', help='Layer for intervention. If -1, sweep over all layers', type=int, required=False, default=9) # 

    parser.add_argument('--n_top_heads', help='Number of attenion head outputs used to compute function vector', required=False, type=int, default=10)
    parser.add_argument('--model_name', help='Name of model to be loaded', type=str, required=False, default='EleutherAI/gpt-j-6b')
    parser.add_argument('--root_data_dir', help='Root directory of data files', type=str, required=False, default='../dataset_files')
    parser.add_argument('--save_path_root', help='File path to save to', type=str, required=False, default='../results')
    parser.add_argument('--seed', help='Randomized seed', type=int, required=False, default=5678)
    parser.add_argument('--device', help='Device to run on',type=str, required=False, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--mean_activations_path', help='Path to file containing mean_head_activations for the specified task', required=False, type=str, default=None)
    parser.add_argument('--test_split', help="Percentage corresponding to test set split size", required=False, default=0.3)    
    parser.add_argument('--n_shots', help="Number of shots in each in-context prompt", type=int, required=False, default=10)
    parser.add_argument('--n_trials', help="Number of in-context prompts to average over for indirect_effect", type=int, required=False, default=25)
    parser.add_argument('--prefixes', help='Prompt template prefixes to be used', type=json.loads, required=False, default={"input":"Q:", "output":"A:", "instructions":""})
    parser.add_argument('--separators', help='Prompt template separators to be used', type=json.loads, required=False, default={"input":"\n", "output":"\n\n", "instructions":""})    
        
    args = parser.parse_args()  

    dataset_name = args.dataset_name
    model_name = args.model_name
    root_data_dir = args.root_data_dir
    save_path_root = f"{args.save_path_root}/{dataset_name}"
    seed = args.seed
    device = args.device
    mean_activations_path = args.mean_activations_path
    n_top_heads = args.n_top_heads
    eval_edit_layer = args.edit_layer

    test_split = args.test_split
    n_shots = args.n_shots
    n_trials = args.n_trials

    prefixes = args.prefixes 
    separators = args.separators
    
    n_eval_templates = args.n_eval_templates

    print(args)

    # Load Model & Tokenizer
    torch.set_grad_enabled(False)
    print("Loading Model")
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(model_name, device=device)

    if args.edit_layer == -1: # sweep over all layers if edit_layer=-1
        eval_edit_layer = [0, model_config['n_layers']]

    # Load the dataset
    print("Loading Dataset")
    set_seed(seed)
    dataset = load_dataset(dataset_name, root_data_dir=root_data_dir, test_size=test_split, seed=seed)

    if not os.path.exists(save_path_root):
        os.makedirs(save_path_root)    

    # Load or Re-Compute mean_head_activations
    if mean_activations_path is not None and os.path.exists(mean_activations_path):
        mean_activations = torch.load(mean_activations_path)
    elif mean_activations_path is None and os.path.exists(f'{save_path_root}/{dataset_name}_mean_head_activations.pt'):
        mean_activations_path = f'{save_path_root}/{dataset_name}_mean_head_activations.pt'
        mean_activations = torch.load(mean_activations_path)        
    else:
        print("Computing Mean Activations")
        set_seed(seed)
        mean_activations = get_mean_head_activations(dataset, model=model, model_config=model_config, tokenizer=tokenizer, 
                                                     n_icl_examples=n_shots, N_TRIALS=n_trials, prefixes=prefixes, separators=separators)
        args.mean_activations_path = f'{save_path_root}/{dataset_name}_mean_head_activations.pt'
        torch.save(mean_activations, args.mean_activations_path)
        
    # Compute Function Vector
    fv, top_heads = compute_universal_function_vector(mean_activations, model, model_config=model_config, n_top_heads=n_top_heads)   
    
    print("Computing Portability")
    fs_res_dict, zs_res_dict,fs_shuffled_res_dict, templates = portability_eval(dataset, fv, eval_edit_layer, model, model_config, tokenizer, n_eval_templates=n_eval_templates)

    args.templates = templates

    save_path_root = f"{args.save_path_root}_port/{dataset_name}"
    if not os.path.exists(save_path_root):
        os.makedirs(save_path_root)   
    
    fs_results_file_name = make_valid_path_name(f'{save_path_root}/fs_port_eval.json')
    args.fs_results_file_name = fs_results_file_name
    with open(fs_results_file_name,'w') as fs_results_file:
        json.dump(fs_res_dict, fs_results_file,indent=2)

    fs_shuffled_results_file_name = make_valid_path_name(f'{save_path_root}/fs_shuffled_port_eval.json')
    args.fs_shuffled_results_file_name = fs_shuffled_results_file_name
    with open(fs_shuffled_results_file_name,'w') as fs_shuffled_results_file:
        json.dump(fs_shuffled_res_dict, fs_shuffled_results_file,indent=2)

    zs_results_file_name = make_valid_path_name(f'{save_path_root}/zs_port_eval.json')
    args.zs_results_file_name = zs_results_file_name
    with open(zs_results_file_name,'w') as zs_results_file:
        json.dump(zs_res_dict, zs_results_file,indent=2)

    args_file_name = make_valid_path_name(f'{save_path_root}/port_eval_args.txt')
    with open(args_file_name, 'w') as arg_file:
        json.dump(args.__dict__, arg_file, indent=2)