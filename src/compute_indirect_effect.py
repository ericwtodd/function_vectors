import os, re, json
from tqdm import tqdm
import torch, numpy as np
import argparse
from baukit import TraceDict

# Include prompt creation helper functions
from utils.prompt_utils import *
from utils.intervention_utils import *
from utils.model_utils import *
from utils.extract_utils import *


def activation_replacement_per_class_intervention(prompt_data, avg_activations, dummy_labels, model, model_config, tokenizer, last_token_only=True):
    """
    Experiment to determine top intervention locations through avg activation replacement. 
    Performs a systematic sweep over attention heads (layer, head) to track their causal influence on probs of key tokens.

    Parameters: 
    prompt_data: dict containing ICL prompt examples, and template information
    avg_activations: avg activation of each attention head in the model taken across n_trials ICL prompts
    dummy_labels: labels and indices for a baseline prompt with the same number of example pairs
    model: huggingface model
    model_config: contains model config information (n layers, n heads, etc.)
    tokenizer: huggingface tokenizer
    last_token_only: If True, only computes indirect effect for heads at the final token position. If False, computes indirect_effect for heads for all token classes

    Returns:   
    indirect_effect_storage: torch tensor containing the indirect_effect of each head for each token class.
    """
    device = model.device

    # Get sentence and token labels
    query_target_pair = prompt_data['query_target']

    query = query_target_pair['input']
    token_labels, prompt_string = get_token_meta_labels(prompt_data, tokenizer, query=query)

    idx_map, idx_avg = compute_duplicated_labels(token_labels, dummy_labels)
    idx_map = update_idx_map(idx_map, idx_avg)
      
    sentences = [prompt_string]# * model.config.n_head # batch things by head

    # Figure out tokens of interest
    target = [query_target_pair['output']]
    token_id_of_interest = get_answer_id(sentences[0], target[0], tokenizer)
    if isinstance(token_id_of_interest, list):
        token_id_of_interest = token_id_of_interest[:1]
        
    inputs = tokenizer(sentences, return_tensors='pt').to(device)

    # Speed up computation by only computing causal effect at last token
    if last_token_only:
        token_classes = ['query_predictive']
        token_classes_regex = ['query_predictive_token']
    # Compute causal effect for all token classes (instead of just last token)
    else:
        token_classes = ['demonstration', 'label', 'separator', 'predictive', 'structural','end_of_example', 
                        'query_demonstration', 'query_structural', 'query_separator', 'query_predictive']
        token_classes_regex = ['demonstration_[\d]{1,}_token', 'demonstration_[\d]{1,}_label_token', 'separator_token', 'predictive_token', 'structural_token','end_of_example_token', 
                            'query_demonstration_token', 'query_structural_token', 'query_separator_token', 'query_predictive_token']
    

    indirect_effect_storage = torch.zeros(model_config['n_layers'], model_config['n_heads'],len(token_classes))

    # Clean Run of Baseline:
    clean_output = model(**inputs).logits[:,-1,:]
    clean_probs = torch.softmax(clean_output[0], dim=-1)

    # For every layer, head, token combination perform the replacement & track the change in meaningful tokens
    for layer in range(model_config['n_layers']):
        head_hook_layer = [model_config['attn_hook_names'][layer]]
        
        for head_n in range(model_config['n_heads']):
            for i,(token_class, class_regex) in enumerate(zip(token_classes, token_classes_regex)):
                reg_class_match = re.compile(f"^{class_regex}$")
                class_token_inds = [x[0] for x in token_labels if reg_class_match.match(x[2])]

                intervention_locations = [(layer, head_n, token_n) for token_n in class_token_inds]
                intervention_fn = replace_activation_w_avg(layer_head_token_pairs=intervention_locations, avg_activations=avg_activations, 
                                                           model=model, model_config=model_config,
                                                           batched_input=False, idx_map=idx_map, last_token_only=last_token_only)
                with TraceDict(model, layers=head_hook_layer, edit_output=intervention_fn) as td:                
                    output = model(**inputs).logits[:,-1,:] # batch_size x n_tokens x vocab_size, only want last token prediction
                
                # TRACK probs of tokens of interest
                intervention_probs = torch.softmax(output, dim=-1) # convert to probability distribution
                indirect_effect_storage[layer,head_n,i] = (intervention_probs-clean_probs).index_select(1, torch.LongTensor(token_id_of_interest).to(device).squeeze()).squeeze()

    return indirect_effect_storage


def compute_indirect_effect(dataset, mean_activations, model, model_config, tokenizer, n_shots=10, n_trials=25, last_token_only=True, prefixes=None, separators=None, filter_set=None):
    """
    Computes Indirect Effect of each head in the model

    Parameters:
    dataset: ICL dataset
    mean_activations:
    model: huggingface model
    model_config: contains model config information (n layers, n heads, etc.)
    tokenizer: huggingface tokenizer
    n_shots: Number of shots in each in-context prompt
    n_trials: Number of in-context prompts to average over
    last_token_only: If True, only computes Indirect Effect for heads at the final token position. If False, computes Indirect Effect for heads for all token classes


    Returns:
    indirect_effect: torch tensor of the indirect effect for each attention head in the model, size n_trials * n_layers * n_heads
    """
    n_test_examples = 1

    if prefixes is not None and separators is not None:
        dummy_gt_labels = get_dummy_token_labels(n_shots, tokenizer=tokenizer, prefixes=prefixes, separators=separators)
    else:
        dummy_gt_labels = get_dummy_token_labels(n_shots, tokenizer=tokenizer)

    is_llama = 'llama' in model_config['name_or_path']
    prepend_bos = not is_llama

    if last_token_only:
        indirect_effect = torch.zeros(n_trials,model_config['n_layers'], model_config['n_heads'])
    else:
        indirect_effect = torch.zeros(n_trials,model_config['n_layers'], model_config['n_heads'],10) # have 10 classes of tokens

    if filter_set is None:
        filter_set = np.arange(len(dataset['valid']))
    
    for i in tqdm(range(n_trials), total=n_trials):
        word_pairs = dataset['train'][np.random.choice(len(dataset['train']),n_shots, replace=False)]
        word_pairs_test = dataset['valid'][np.random.choice(filter_set,n_test_examples, replace=False)]
        if prefixes is not None and separators is not None:
            prompt_data_random = word_pairs_to_prompt_data(word_pairs, query_target_pair=word_pairs_test, shuffle_labels=True, 
                                                           prepend_bos_token=prepend_bos, prefixes=prefixes, separators=separators)
        else:
            prompt_data_random = word_pairs_to_prompt_data(word_pairs, query_target_pair=word_pairs_test, 
                                                           shuffle_labels=True, prepend_bos_token=prepend_bos)
        
        ind_effects = activation_replacement_per_class_intervention(prompt_data=prompt_data_random, 
                                                                    avg_activations = mean_activations, 
                                                                    dummy_labels=dummy_gt_labels, 
                                                                    model=model, model_config=model_config, tokenizer=tokenizer, 
                                                                    last_token_only=last_token_only)
        indirect_effect[i] = ind_effects.squeeze()

    return indirect_effect


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', help='Name of the dataset to be loaded', type=str, required=True)
    parser.add_argument('--model_name', help='Name of model to be loaded', type=str, required=False, default='EleutherAI/gpt-j-6b')
    parser.add_argument('--root_data_dir', help='Root directory of data files', type=str, required=False, default='../dataset_files')
    parser.add_argument('--save_path_root', help='File path to save indirect effect to', type=str, required=False, default='../results')
    parser.add_argument('--seed', help='Randomized seed', type=int, required=False, default=42)
    parser.add_argument('--n_shots', help="Number of shots in each in-context prompt", type =int, required=False, default=10)
    parser.add_argument('--n_trials', help="Number of in-context prompts to average over", type=int, required=False, default=25)
    parser.add_argument('--test_split', help="Percentage corresponding to test set split size", required=False, default=0.3)
    parser.add_argument('--device', help='Device to run on',type=str, required=False, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--mean_activations_path', help='Path to mean activations file used for intervention', required=False, type=str, default=None)
    parser.add_argument('--last_token_only', help='Whether to compute indirect effect for heads at only the final token position, or for all token classes', required=False, type=bool, default=True)
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
    mean_activations_path = args.mean_activations_path
    last_token_only = args.last_token_only
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
    

    if not os.path.exists(save_path_root):
        os.makedirs(save_path_root)

    # Load or Re-Compute Mean Activations
    if mean_activations_path is not None and os.path.exists(mean_activations_path):
        mean_activations = torch.load(mean_activations_path)
    elif mean_activations_path is None and os.path.exists(f'{save_path_root}/{dataset_name}_mean_head_activations.pt'):
        mean_activations_path = f'{save_path_root}/{dataset_name}_mean_head_activations.pt'
        mean_activations = torch.load(mean_activations_path)        
    else:
        print("Computing Mean Activations")
        mean_activations = get_mean_head_activations(dataset, model=model, model_config=model_config, tokenizer=tokenizer, 
                                                     n_icl_examples=n_shots, N_TRIALS=n_trials, prefixes=prefixes, separators=separators)
        torch.save(mean_activations, f'{save_path_root}/{dataset_name}_mean_head_activations.pt')

    print("Computing Indirect Effect")
    indirect_effect = compute_indirect_effect(dataset, mean_activations, model=model, model_config=model_config, tokenizer=tokenizer, 
                                              n_shots=n_shots, n_trials=n_trials, last_token_only=last_token_only, prefixes=prefixes, separators=separators)

    # Write args to file
    args.save_path_root = save_path_root
    args.mean_activations_path = mean_activations_path
    with open(f'{save_path_root}/indirect_effect_args.txt', 'w') as arg_file:
        json.dump(args.__dict__, arg_file, indent=2)

    torch.save(indirect_effect, f'{save_path_root}/{dataset_name}_indirect_effect.pt')

    