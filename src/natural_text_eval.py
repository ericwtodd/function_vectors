import os, re, json
import torch, numpy as np
from tqdm import tqdm
import argparse

# Include prompt creation helper functions
from utils.eval_utils import n_shot_eval_no_intervention
from utils.extract_utils import get_mean_head_activations, compute_universal_function_vector
from utils.intervention_utils import fv_intervention_natural_text
from utils.model_utils import load_gpt_model_and_tokenizer
from utils.prompt_utils import load_dataset


def natural_text_eval(dataset, fv_vector, model, model_config, tokenizer, filter_set, edit_layer=9, MNT=5, verbose=False, loc_data=False):
    """
    Evaluates the causal effects of a function vector on natural text templates for the given dataset.

    Parameters:
    dataset: ICL dataset with pairs of words
    fv_vector: function vector to use for intervention
    model: huggingface model
    model_config: contains model config information (n layers, n heads, etc.)
    tokenizer: huggingface tokenizer
    filter_set: list of samples to filter to, used to include samples the model gets correct via ICL
    edit_layer: the layer to add the function vector to
    MNT: max number of tokens to generate
    verbose: whether to print outputs of clean & +FV generations.
    loc_data: whether the dataset is locations (e.g. country-capital, national parks, etc.)

    Returns:
    all_scores: scores for model when adding the FV during generation
    all_clean_scores: scores for base model (no intervention)
    sentences: sentence templates used during eval
    """    
    all_scores = {}
    all_clean_scores = {}

    if loc_data: # country-capital & similar datasets
        sentences = ["A couple years ago I visited {X}, and",
                     "If you ever travel to {X}, you have to visit",   
                     "When you think of {X},"]
    else:
        sentences = ["The word \"{X}\", means", 
                     "When I think of the word \"{X}\", it usually means",
                     "When I think of \"{X}\", I usually",
                     "While reading a book, I came across the word \"{X}\". I looked it up in a dictionary and it turns out that it means",
                     "The word \"{X}\" can be understood as a synonym for"]
        
    for j in range(len(sentences)):
        scores = []
        clean_scores = []
        for i in tqdm(range(len(filter_set)), total=len(filter_set)):
            ind = int(filter_set[i])
            q_pair = dataset['test'][ind]       
            if isinstance(q_pair['input'], list):
                q_pair['input'] = q_pair['input'][0]
            if isinstance(q_pair['output'], list):
                q_pair['output'] = q_pair['output'][0]

            sentence = sentences[j]
            sentence = sentence.replace('{X}', f"{q_pair['input']}")       

            clean_output, fv_output = fv_intervention_natural_text(sentence, edit_layer, fv_vector, model, model_config, tokenizer, max_new_tokens=MNT)
            clean_out_str = repr(tokenizer.decode(clean_output.squeeze()[-MNT:]))
            fv_out_str = repr(tokenizer.decode(fv_output.squeeze()[-MNT:]))

            if verbose:
                print("\nQuery/Target: ", q_pair)
                print("Prompt: ", repr(sentence))
                print("clean completion:" , clean_out_str)
                print("+FV completion:", fv_out_str, '\n')          
            
            scores.append(int(q_pair['output'] in fv_out_str))
            clean_scores.append(int(q_pair['output'] in clean_out_str))
        
        all_scores[j] = scores
        all_clean_scores[j] = clean_scores

    return all_scores, all_clean_scores, sentences

def nattext_main(datasets, model, model_config, tokenizer, root_data_dir='../dataset_files', edit_layer=9, n_shots=10, n_trials=100, n_seeds=5):
    """
    Main function that evaluates causal effects of function vectors on natural text templates.

    Parameters:
    datasets: list of dataset names to evaluate
    model: huggingface model
    model_config: contains model config information (n layers, n heads, etc.)
    tokenizer: huggingface tokenizer
    root_data_dir: directory data is contained in
    edit_layer: layer to add the function vector to during intervention
    n_shots: number of shots for prompts used when computing task-conditioned mean head activations
    n_trials: number of prompts to include when computing task-conditioned mean head activations
    n_seeds: number of seeds to average results over

    Returns:
    clean_results_dict: dict containing results for base model (no intervention)
    interv_results_dict: results for model when adding the function vector at edit_layer during generation
    seeds_dict: dict containing the seeds used during evaluation
    """
    interv_results_dict = {k:[] for k in datasets}
    clean_results_dict = {k:[] for k in datasets}
    seeds_dict = {k:[] for k in datasets}
    
    # Test Loop:
    for dataset_name in datasets:
        if dataset_name == 'country-capital':
            loc_data = True
            max_new_tokens = 10
        else:
            loc_data = False
            max_new_tokens = 5

        for _ in range(n_seeds):
            seed = np.random.randint(100000)
            seeds_dict[dataset_name].append(seed)
            dataset = load_dataset(dataset_name, seed=seed, root_data_dir=root_data_dir)

            fs_results_validation = n_shot_eval_no_intervention(dataset=dataset, n_shots=n_shots, model=model, model_config=model_config, tokenizer=tokenizer, compute_ppl=False, test_split='valid')
            filter_set_validation = np.where(np.array(fs_results_validation['clean_rank_list']) == 0)[0]
            
            mean_activations = get_mean_head_activations(dataset, model=model, model_config=model_config, tokenizer=tokenizer, n_icl_examples=n_shots,
                                                        N_TRIALS=n_trials, filter_set=filter_set_validation)
            fv, _ = compute_universal_function_vector(mean_activations, model, model_config)
            
            fs_results = n_shot_eval_no_intervention(dataset=dataset, n_shots=n_shots, model=model, model_config=model_config, tokenizer=tokenizer, compute_ppl=False, test_split='test')
            filter_set = np.where(np.array(fs_results['clean_rank_list']) == 0)[0]
            
            results, clean_results, _ = natural_text_eval(dataset, fv, model, model_config, tokenizer, filter_set, MNT=max_new_tokens, edit_layer=edit_layer, verbose=False, loc_data=loc_data)

            clean_results_dict[dataset_name].append([np.mean(clean_results[i]) for i in clean_results.keys()])
            interv_results_dict[dataset_name].append([np.mean(results[i]) for i in results.keys()])
    
    return clean_results_dict, interv_results_dict, seeds_dict


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', help='Name of model to be loaded', type=str, required=False, default='EleutherAI/gpt-j-6b')
    parser.add_argument('--root_data_dir', help='Root directory of data files', type=str, required=False, default='../dataset_files')
    parser.add_argument('--save_path_root', help='File path to save mean activations to', type=str, required=False, default='../results')
    parser.add_argument('--n_seeds', help='Number of seeds', type=int, required=False, default=5)
    parser.add_argument('--n_trials', help='Number of trials to use for computing task-conditioned mean head activations', type=int, required=False, default=100)
    parser.add_argument('--n_shots', help='Number of shots to use for prompts when computing task-conditioned mean head activations', type=int, required=False, default=10)
    parser.add_argument('--edit_layer', help='Layer to add function vector to', type=int, required=False, default=9)

    args = parser.parse_args()
    
    # Gather inputs
    model_name = args.model_name
    root_data_dir = args.root_data_dir
    save_path_root = args.save_path_root
    n_seeds = args.n_seeds
    n_trials = args.n_trials
    n_shots = args.n_shots
    edit_layer = args.edit_layer
    

    # Load Model & Tokenizer
    torch.set_grad_enabled(False)
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(model_name)

    datasets = ['antonym', 'capitalize', 'country-capital', 'english-french', 'present-past', 'singular-plural']
    args.datasets = datasets

    # Run Natural Text Eval
    clean_results_dict, interv_results_dict, seeds_dict = nattext_main(datasets, model, model_config, tokenizer, 
                                                                       root_data_dir=root_data_dir, edit_layer=edit_layer, 
                                                                       n_shots=n_shots, n_trials=n_trials, n_seeds=n_seeds)

    # Extract Summary Results:
    os.makedirs(os.path.join(save_path_root), exist_ok=True)
    with open(os.path.join(save_path_root, 'nattext_eval_results.txt'), 'w') as out_file:
        for d in datasets:
            print(f"{d.title()}:", file=out_file)
            clean_acc = np.array(clean_results_dict[d]).mean(axis=0)
            clean_std = np.array(clean_results_dict[d]).std(axis=0)
            fv_acc = np.array(interv_results_dict[d]).mean(axis=0)
            fv_std = np.array(interv_results_dict[d]).std(axis=0)

            print("clean results:", clean_acc.round(3)*100, '% +/-', clean_std.round(3)*100, file=out_file)
            print("fv results:", fv_acc.round(3)*100, '% +/-', fv_std.round(3)*100, file=out_file)

    # Write args to a file
    args.seeds_dict = seeds_dict
    with open(os.path.join(save_path_root, 'nattext_eval_args.txt'), 'w') as arg_file:
        print(args.__dict__, file=arg_file)


    