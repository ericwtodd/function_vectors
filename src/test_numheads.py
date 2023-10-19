import argparse
import json
import os
import numpy as np
import torch

from src.utils.eval_utils import n_shot_eval, n_shot_eval_no_intervention
from src.utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from src.utils.prompt_utils import load_dataset
from src.evaluate_function_vector import compute_universal_function_vector

# Evaluates how performance changes as the number of heads used to create a Function Vector increases
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', help="dataset to be evaluated", type=str, required=True)
    parser.add_argument('--mean_act_root', help="root path to mean activations", type=str, required=False, default='IE_template_QA/gptj')
    parser.add_argument('--model_name', type=str, required=True, default='EleutherAI/gpt-j-6b')
    parser.add_argument('--model_nickname', type=str, required=False, default='gptj')
    parser.add_argument('--n_heads', type=int, help="upper bound of the number of heads to create the FV", required=True, default=40)
    parser.add_argument('--edit_layer', type=int, help="layer at which to add the function vector", required=True, default=9)
    parser.add_argument('--seed', required=False, type=int, default=42)
    parser.add_argument('--save_path_root', required=True, type=str, default='../results')

    
    args = parser.parse_args()
    mean_act_root = args.mean_act_root
    model_name = args.model_name
    model_nickname = args.model_nickname
    dataset_name = args.dataset_name
    n_heads = args.n_heads
    edit_layer = args.edit_layer
    seed = args.seed
    save_path_root = args.save_path_root


    # Load Model & Tokenizer, doing inference so don't need gradients
    torch.set_grad_enabled(False)
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(model_name)
    dataset = load_dataset(dataset_name)
    mean_activations = torch.load(f'{save_path_root}/{mean_act_root}/{dataset_name}/{dataset_name}_mean_head_activations.pt')


    set_seed(seed)
    fs_results = n_shot_eval_no_intervention(dataset, n_shots=10, model=model, model_config=model_config, tokenizer=tokenizer)
    filter_set = np.where(np.array(fs_results['clean_rank_list']) == 0)[0]
    print("Sanity Check, cleantopk: ", fs_results['clean_topk'])
    zs_results = {}

    for i in range(n_heads+1):   
        fv, _ = compute_universal_function_vector(mean_activations, model, model_config, i)
        zs_results[i] = n_shot_eval(dataset, fv, edit_layer, 0, model, model_config, tokenizer, filter_set=filter_set)
    
    
    os.makedirs(f'{save_path_root}/{model_nickname}_test_numheads', exist_ok=True)
    json.dump(zs_results, open(f'{save_path_root}/{model_nickname}_test_numheads/{dataset_name}_perf_v_heads.json', 'w'))