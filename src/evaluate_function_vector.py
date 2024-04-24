import os, json
import torch, numpy as np
import argparse

# Include prompt creation helper functions
from utils.prompt_utils import *
from utils.intervention_utils import *
from utils.model_utils import *
from utils.eval_utils import *
from utils.extract_utils import *
from compute_indirect_effect import compute_indirect_effect

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', help='Name of the dataset to be loaded', type=str, required=True)
    parser.add_argument('--n_top_heads', help='Number of attenion head outputs used to compute function vector', required=False, type=int, default=10)
    parser.add_argument('--edit_layer', help='Layer for intervention. If -1, sweep over all layers', type=int, required=False, default=-1) # 
    parser.add_argument('--model_name', help='Name of model to be loaded', type=str, required=False, default='EleutherAI/gpt-j-6b')
    parser.add_argument('--root_data_dir', help='Root directory of data files', type=str, required=False, default='../dataset_files')
    parser.add_argument('--save_path_root', help='File path to save to', type=str, required=False, default='../results')
    parser.add_argument('--ie_path_root', help='File path to load indirect effects from', type=str, required=False, default=None)
    parser.add_argument('--seed', help='Randomized seed', type=int, required=False, default=42)
    parser.add_argument('--device', help='Device to run on',type=str, required=False, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--mean_activations_path', help='Path to file containing mean_head_activations for the specified task', required=False, type=str, default=None)
    parser.add_argument('--indirect_effect_path', help='Path to file containing indirect_effect scores for the specified task', required=False, type=str, default=None)    
    parser.add_argument('--test_split', help="Percentage corresponding to test set split size", required=False, default=0.3)    
    parser.add_argument('--n_shots', help="Number of shots in each in-context prompt", type=int, required=False, default=10)
    parser.add_argument('--n_trials', help="Number of in-context prompts to average over for indirect_effect", type=int, required=False, default=25)
    parser.add_argument('--prefixes', help='Prompt template prefixes to be used', type=json.loads, required=False, default={"input":"Q:", "output":"A:", "instructions":""})
    parser.add_argument('--separators', help='Prompt template separators to be used', type=json.loads, required=False, default={"input":"\n", "output":"\n\n", "instructions":""})    
    parser.add_argument('--compute_baseline', help='Whether to compute the model baseline 0-shot -> n-shot performance', type=bool, required=False, default=True)
    parser.add_argument('--generate_str', help='Whether to generate long-form completions for the task', action='store_true', required=False)
    parser.add_argument("--metric", help="Metric to use when evaluating generated strings", type=str, required=False, default="f1_score")
    parser.add_argument("--universal_set", help="Flag for whether to evaluate using the univeral set of heads", action="store_true", required=False)
        
    args = parser.parse_args()  

    dataset_name = args.dataset_name
    model_name = args.model_name
    root_data_dir = args.root_data_dir
    save_path_root = f"{args.save_path_root}/{dataset_name}"
    ie_path_root = f"{args.ie_path_root}/{dataset_name}" if args.ie_path_root else save_path_root
    seed = args.seed
    device = args.device
    mean_activations_path = args.mean_activations_path
    indirect_effect_path = args.indirect_effect_path
    n_top_heads = args.n_top_heads
    eval_edit_layer = args.edit_layer

    test_split = float(args.test_split)
    n_shots = args.n_shots
    n_trials = args.n_trials

    prefixes = args.prefixes 
    separators = args.separators
    compute_baseline = args.compute_baseline

    generate_str = args.generate_str
    metric = args.metric
    universal_set = args.universal_set

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

    print(f"Filtering Dataset via {n_shots}-shot Eval")
    # 1. Compute Model 10-shot Baseline & 2. Filter test set to cases where model gets it correct

    fs_results_file_name = f'{save_path_root}/fs_results_layer_sweep.json'
    print(fs_results_file_name)
    if os.path.exists(fs_results_file_name):
        with open(fs_results_file_name, 'r') as indata:
            fs_results = json.load(indata)
        key = 'score' if generate_str else 'clean_rank_list'
        target_val = 1 if generate_str else 0
        filter_set = np.where(np.array(fs_results[key]) == target_val)[0]
        filter_set_validation = None
    elif generate_str:
        set_seed(seed+42)
        fs_results_validation = n_shot_eval_no_intervention(dataset=dataset, n_shots=n_shots, model=model, model_config=model_config, tokenizer=tokenizer, compute_ppl=False,
                                                 generate_str=True, metric=metric, test_split='valid', prefixes=prefixes, separators=separators)
        filter_set_validation = np.where(np.array(fs_results_validation['score']) == 1)[0]
        set_seed(seed)
        fs_results = n_shot_eval_no_intervention(dataset=dataset, n_shots=n_shots, model=model, model_config=model_config, tokenizer=tokenizer, compute_ppl=False,
                                                 generate_str=True, metric=metric, prefixes=prefixes, separators=separators)
        filter_set = np.where(np.array(fs_results['score']) == 1)[0]
    else:
        set_seed(seed+42)
        fs_results_validation = n_shot_eval_no_intervention(dataset=dataset, n_shots=n_shots, model=model, model_config=model_config, tokenizer=tokenizer, compute_ppl=True, test_split='valid', prefixes=prefixes, separators=separators)
        filter_set_validation = np.where(np.array(fs_results_validation['clean_rank_list']) == 0)[0]
        set_seed(seed)
        fs_results = n_shot_eval_no_intervention(dataset=dataset, n_shots=n_shots, model=model, model_config=model_config, tokenizer=tokenizer, compute_ppl=True, prefixes=prefixes, separators=separators)
        filter_set = np.where(np.array(fs_results['clean_rank_list']) == 0)[0]
    
    args.fs_results_file_name = fs_results_file_name
    with open(fs_results_file_name, 'w') as results_file:
        json.dump(fs_results, results_file, indent=2)

    set_seed(seed)
    # Load or Re-Compute mean_head_activations
    if mean_activations_path is not None and os.path.exists(mean_activations_path):
        mean_activations = torch.load(mean_activations_path)
    elif mean_activations_path is None and os.path.exists(f'{ie_path_root}/{dataset_name}_mean_head_activations.pt'):
        mean_activations_path = f'{ie_path_root}/{dataset_name}_mean_head_activations.pt'
        mean_activations = torch.load(mean_activations_path)        
    else:
        print("Computing Mean Activations")
        set_seed(seed)
        mean_activations = get_mean_head_activations(dataset, model=model, model_config=model_config, tokenizer=tokenizer, n_icl_examples=n_shots,
                                                     N_TRIALS=n_trials, prefixes=prefixes, separators=separators, filter_set=filter_set_validation)
        args.mean_activations_path = f'{save_path_root}/{dataset_name}_mean_head_activations.pt'
        torch.save(mean_activations, args.mean_activations_path)

    # Load or Re-Compute indirect_effect values
    if indirect_effect_path is not None and os.path.exists(indirect_effect_path):
        indirect_effect = torch.load(indirect_effect_path)
    elif indirect_effect_path is None and os.path.exists(f'{ie_path_root}/{dataset_name}_indirect_effect.pt'):
        indirect_effect_path = f'{ie_path_root}/{dataset_name}_indirect_effect.pt'
        indirect_effect = torch.load(indirect_effect_path) 
    elif not universal_set:     # Only compute indirect effects if we need to
        print("Computing Indirect Effects")
        set_seed(seed)
        indirect_effect = compute_indirect_effect(dataset, mean_activations, model=model, model_config=model_config, tokenizer=tokenizer, n_shots=n_shots,
                                                  n_trials=n_trials, last_token_only=True, prefixes=prefixes, separators=separators, filter_set=filter_set_validation)
        args.indirect_effect_path = f'{save_path_root}/{dataset_name}_indirect_effect.pt'
        torch.save(indirect_effect, args.indirect_effect_path)
        
    # Compute Function Vector
    if universal_set:
        fv, top_heads = compute_universal_function_vector(mean_activations, model, model_config=model_config, n_top_heads=n_top_heads)   
    else:
        fv, top_heads = compute_function_vector(mean_activations, indirect_effect, model, model_config=model_config, n_top_heads=n_top_heads)   
    
    # Run Evaluation
    if isinstance(eval_edit_layer, int):
        print(f"Running ZS Eval with edit_layer={eval_edit_layer}")
        set_seed(seed)
        if generate_str:
            pred_filepath = f"{save_path_root}/preds/{model_config['name_or_path'].replace('/', '_')}_ZS_intervention_layer{eval_edit_layer}.txt"
            zs_results = n_shot_eval(dataset=dataset, fv_vector=fv, edit_layer=eval_edit_layer, n_shots=0,
                                     model=model, model_config=model_config, tokenizer=tokenizer, filter_set=filter_set,
                                     generate_str=generate_str, metric=metric, pred_filepath=pred_filepath, prefixes=prefixes, separators=separators)
        else:
            zs_results = n_shot_eval(dataset=dataset, fv_vector=fv, edit_layer=eval_edit_layer, n_shots=0,
                                    model=model, model_config=model_config, tokenizer=tokenizer, filter_set=filter_set, prefixes=prefixes, separators=separators)
        zs_results_file_suffix = f'_editlayer_{eval_edit_layer}.json'   


        print(f"Running {n_shots}-Shot Shuffled Eval")
        set_seed(seed)
        if generate_str:
            pred_filepath = f"{save_path_root}/preds/{model_config['name_or_path'].replace('/', '_')}_{n_shots}shots_shuffled_intervention_layer{eval_edit_layer}.txt"
            fs_shuffled_results = n_shot_eval(dataset=dataset, fv_vector=fv, edit_layer=eval_edit_layer, n_shots=n_shots, 
                                              model=model, model_config=model_config, tokenizer=tokenizer, filter_set=filter_set, shuffle_labels=True,
                                              generate_str=generate_str, metric=metric, pred_filepath=pred_filepath, prefixes=prefixes, separators=separators)
        else:
            fs_shuffled_results = n_shot_eval(dataset=dataset, fv_vector=fv, edit_layer=eval_edit_layer, n_shots=n_shots, 
                                              model=model, model_config=model_config, tokenizer=tokenizer, filter_set=filter_set, shuffle_labels=True, prefixes=prefixes, separators=separators)
        fs_shuffled_results_file_suffix = f'_editlayer_{eval_edit_layer}.json'   
        
    else:
        print(f"Running sweep over layers {eval_edit_layer}")
        zs_results = {}
        fs_shuffled_results = {}
        for edit_layer in range(eval_edit_layer[0], eval_edit_layer[1]):
            set_seed(seed)
            if generate_str:
                zs_results[edit_layer] = n_shot_eval(dataset=dataset, fv_vector=fv, edit_layer=edit_layer, n_shots=0, 
                                                    model=model, model_config=model_config, tokenizer=tokenizer, filter_set=filter_set,
                                                    generate_str=generate_str, metric=metric, prefixes=prefixes, separators=separators)
            else:
                zs_results[edit_layer] = n_shot_eval(dataset=dataset, fv_vector=fv, edit_layer=edit_layer, n_shots=0, prefixes=prefixes, separators=separators,
                                                    model=model, model_config=model_config, tokenizer=tokenizer, filter_set=filter_set)
            set_seed(seed)
            if generate_str:
                fs_shuffled_results[edit_layer] = n_shot_eval(dataset=dataset, fv_vector=fv, edit_layer=edit_layer, n_shots=n_shots, 
                                                    model=model, model_config=model_config, tokenizer=tokenizer, filter_set = filter_set,
                                                    generate_str=generate_str, metric=metric, shuffle_labels=True, prefixes=prefixes, separators=separators)
            else:
                fs_shuffled_results[edit_layer] = n_shot_eval(dataset=dataset, fv_vector=fv, edit_layer=edit_layer, n_shots=n_shots, 
                                                    model=model, model_config=model_config, tokenizer=tokenizer, filter_set = filter_set, shuffle_labels=True, prefixes=prefixes, separators=separators)
        zs_results_file_suffix = '_layer_sweep.json'
        fs_shuffled_results_file_suffix = '_layer_sweep.json'


    # Save results to files
    zs_results_file_name = make_valid_path_name(f'{save_path_root}/zs_results' + zs_results_file_suffix)
    args.zs_results_file_name = zs_results_file_name
    with open(zs_results_file_name, 'w') as results_file:
        json.dump(zs_results, results_file, indent=2)
    
    fs_shuffled_results_file_name = make_valid_path_name(f'{save_path_root}/fs_shuffled_results' + fs_shuffled_results_file_suffix)
    args.fs_shuffled_results_file_name = fs_shuffled_results_file_name
    with open(fs_shuffled_results_file_name, 'w') as results_file:
        json.dump(fs_shuffled_results, results_file, indent=2)

    if compute_baseline:
        print(f"Computing model baseline results for {n_shots}-shots")
        baseline_results = compute_dataset_baseline(dataset, model, model_config, tokenizer, n_shots=n_shots, seed=seed, prefixes=prefixes, separators=separators)        
    
        baseline_file_name = make_valid_path_name(f'{save_path_root}/model_baseline.json')
        args.baseline_file_name = baseline_file_name
        with open(baseline_file_name, 'w') as results_file:
            json.dump(baseline_results, results_file, indent=2)

    # Write args to file
    args_file_name = make_valid_path_name(f'{save_path_root}/fv_eval_args.txt')
    with open(args_file_name, 'w') as arg_file:
        json.dump(args.__dict__, arg_file, indent=2)
