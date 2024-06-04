import torch
import string
import itertools
import re
import numpy as np
from tqdm import tqdm
from .prompt_utils import *
from .model_utils import *
from .intervention_utils import *


def compute_top_k_accuracy(target_token_ranks, k=10) -> float:
    """
    Evaluation to compute topk accuracy.

    Parameters:
    target_token_ranks: the distribution of output token ranks
    k: how many tokens we're looking at (top K)

    Return:
    The accuracy of the token in the top k of tokens
    """

    target_token_ranks = np.array(target_token_ranks)
    return (target_token_ranks < k).sum(axis=0) / len(target_token_ranks) 

def compute_individual_token_rank(prob_dist, target_id) -> int:
    """
    Individual computation of token ranks across a single distribution.

    Parameters:
    prob_dist: the distribution of scores for a single output
    target_id: the target id we care about

    Return:
    A single value representing the token rank for that single token
    """
    if isinstance(target_id, list):
        target_id = target_id[0]

    return torch.where(torch.argsort(prob_dist.squeeze(), descending=True) == target_id)[0].item()


def compute_best_token_rank(prob_dist, target_ids) -> int:
    """
    Computes the best rank given a list of potential targets (target_ids) for a given probability distribution (prob_dist)
    """
    related_token_ranks = [compute_individual_token_rank(prob_dist, x) for x in target_ids]
    return min(related_token_ranks)

def compute_top_k_elements(x, K=10) -> list:
    """
    Computes the top k elements of a torch tensor (x), and returns them as a list of index tuples
    """
    h_shape = x.shape
    topk_vals, topk_inds  = torch.topk(x.view(-1), k=K, largest=True)
    top_lh = list(zip(*np.unravel_index(topk_inds, h_shape), [round(x.item(),4) for x in topk_vals]))
    top_elements = top_lh[:K]
    return top_elements

def decode_to_vocab(prob_dist, tokenizer, k=5) -> list:
    """
    Decodes and returns the top K words of a probability distribution

    Parameters:
    prob_dist: torch tensor of model logits (distribution over the vocabulary)
    tokenizer: huggingface model tokenizer
    k: number of vocabulary words to include

    Returns:
    list of top K decoded vocabulary words in the probability distribution as strings, along with their probabilities (float)
    """
    get_topk = lambda  x,K=1: torch.topk(torch.softmax(x, dim=-1), dim=-1, k=K)
    if not isinstance(prob_dist, torch.Tensor):
        prob_dist = torch.Tensor(prob_dist)

    return [(tokenizer.decode(x),round(y.item(), 5)) for x,y in zip(get_topk(prob_dist,k).indices[0],get_topk(prob_dist,k).values[0])]

def get_answer_id(query, answer, tokenizer):
    """
    Parameters:
    query (str): query as a string
    answer (str): expected answer as a string
    tokenizer: huggingface tokenizer
    
    Returns: 
    answer_ids (list): A list of the contextualized tokens of the answer
    """
    source = tokenizer(query, truncation=False, padding=False).input_ids
    target = tokenizer(query + answer, truncation=False, padding=False).input_ids
    assert len(source) < len(target) < tokenizer.model_max_length
    answer_ids = target[len(source): ]
    return answer_ids

def fv_to_vocab(function_vector, model, model_config, tokenizer, n_tokens=10):
    """
    Decodes a provided function vector into the model's vocabulary embedding space.

    Parameters:
    function_vector: torch vector extracted from ICL contexts that represents a particular function
    model: huggingface model
    model_config: dict with model information - n_layers, n_heads, etc.
    tokenizer: huggingface tokenizer
    n_tokens: number of top tokens to include in the decoding

    Returns:
    decoded_tokens: list of tuples of the form [(token, probability), ...]
    """

    if 'gpt-j' in model_config['name_or_path']:
        decoder = torch.nn.Sequential(model.transformer.ln_f, model.lm_head, torch.nn.Softmax(dim=-1))
    elif 'llama' in model_config['name_or_path']:
        decoder = torch.nn.Sequential(model.model.norm, model.lm_head, torch.nn.Softmax(dim=-1))
    else:
        raise ValueError("Model not yet supported")
    
    d_out = decoder(function_vector.reshape(1,1,model_config['resid_dim']).to(model.device))

    vals, inds = torch.topk(d_out, k=n_tokens,largest=True)
    decoded_tokens = [(tokenizer.decode(x),round(y.item(), 4)) for x,y in zip(inds.squeeze(), vals.squeeze())]
    return decoded_tokens

def compute_dataset_baseline(dataset, model, model_config, tokenizer, n_shots=10, seed=42, generate_str=False, metric=None, prefixes=None, separators=None) -> dict:
    """
    Computes the ICL performance of the model on the provided dataset for a varying number of shots.

    Parameters:
    dataset: ICL dataset
    model: huggingface model
    model_config: contains model config information (n layers, n heads, etc.)
    tokenizer: huggingface tokenizer
    n_shots: The upper bound of ICL examples to be used when evaluating the ICL performance of the model
    seed: seed for determining dataset split
    generate_str: whether to generate a string of tokens or predict a single token
    metric: metric to use for longer generations (F1, exact match, etc.), or None for single token prediction accuracy is used

    Returns:
    results_dict: dictionary containing the ICL performance results as the number of shots in ICL prompts varies.
    """
    results_dict = {}
    for N in range(n_shots+1):
        set_seed(seed)
        results_dict[N] = n_shot_eval_no_intervention(dataset, n_shots=N, model=model, model_config=model_config, tokenizer=tokenizer,
                                                      generate_str=generate_str, metric=metric, prefixes=prefixes, separators=separators)
    return results_dict

def is_nontrivial_prefix(prediction: str, target: str) -> bool:
    """Return true if prediction is (case insensitive) prefix of the target."""
    target = target.lower().strip()
    prediction = prediction.lower().strip()
    return len(prediction) > 0 and target.startswith(prediction)

# Evaluate a sentence
def sentence_eval(sentence, target, model, tokenizer, compute_nll=True, generate_str=False, pred_file=None, metric_fn=None):
    """
    Evaluate a single sentence completion for a model, comparing to the given target.

    Parameters:
    sentence: sentence to have the model process and predict
    target: expected response of the model
    model: huggingface model
    tokenizer: huggingface tokenizer
    compute_nll: whether to compute the negative log likelihood of a teacher-forced answer prompt (used for computing PPL)
    generate_str: whether to generate a string of tokens or predict a single token
    pred_file: filepath to save intermediate generations for debugging
    metric: metric to use for longer generations (F1, exact match, etc.)

    Returns:
    model output on the provided sentence
    """
    # Clean Run, No Intervention:
    device = model.device
    inputs = tokenizer(sentence, return_tensors='pt').to(device)
    original_pred_idx = len(inputs.input_ids.squeeze()) - 1

    if compute_nll:
        target_completion = "".join(sentence + target)
        nll_inputs = tokenizer(target_completion, return_tensors='pt').to(device)
        nll_targets = nll_inputs.input_ids.clone()
        target_len = len(nll_targets.squeeze()) - len(inputs.input_ids.squeeze()) 
        nll_targets[:,:-target_len] = -100  # This is the accepted value to skip indices when computing loss in nn.CrossEntropyLoss

        output = model(**nll_inputs, labels=nll_targets)

        clean_nll = output.loss.item()
        clean_output = output.logits[:,original_pred_idx,:]
    elif generate_str:
        MAX_NEW_TOKENS = 16
        output = model.generate(inputs.input_ids, top_p=0.9, temperature=0.1,
                                max_new_tokens=MAX_NEW_TOKENS,
                                pad_token_id=tokenizer.eos_token_id)
        output_str = tokenizer.decode(output.squeeze()[-MAX_NEW_TOKENS:])
        parsed_str, score = parse_generation(output_str, target, metric_fn)
        if pred_file:
            pred_file.write(f"{parsed_str.strip()}\n")
    else:
        clean_output = model(**inputs).logits[:,-1,:]
    

    if compute_nll:
        return clean_output, clean_nll
    elif generate_str:
        return score
    else:
        return clean_output 


def n_shot_eval(dataset, fv_vector, edit_layer: int, n_shots: int, model, model_config, tokenizer, shuffle_labels:bool=False,
                filter_set=None, prefixes=None, separators=None, generate_str=False, pred_filepath=None,
                metric="f1_score"):
    """
    Evaluate a model and FV intervention on the model using the provided ICL dataset.

    Parameters:
    dataset: ICL dataset
    function_vector: torch vector that triggers execution of a task when added to a particular layer
    edit_layer: layer index 
    n_shots: the number of ICL examples in each in-context prompt
    model: huggingface model
    model_config: contains model config information (n layers, n heads, etc.)
    tokenizer: huggingface tokenizer
    shuffle_labels: Whether to shuffle the ICL labels or not
    filter_set: whether to only include samples the model gets correct via ICL
    prefixes: dict of ICL template prefixes for each ICL component (input, output, instructions)
    separators: dict of ICL template separators for each ICL component (input, output, instructions)
    generate_str: whether to generate a string of tokens or predict a single token
    pred_filepath: filepath to save intermediate generations for debugging
    metric: metric to use for longer generations (F1, exact match, etc.)

    Returns:
    results: dict of topk accuracy on the test dataset, for both the model's n-shot, and n-shot + FV intervention, as well as the token rank of each prediction
    """
    clean_rank_list = []
    intervention_rank_list = []

    if generate_str:
        clean_score_list = []
        intervention_score_list = []

    # If the model already prepends a bos token by default, we don't want to add one
    prepend_bos =  False if model_config['prepend_bos'] else True

    if filter_set is None:
        filter_set = np.arange(len(dataset['test']))

    if pred_filepath:
        pred_file = open(pred_filepath, 'w')
    else:
        pred_file = None        

    for j in tqdm(range(len(dataset['test'])), total=len(dataset['test'])):
        if j not in filter_set:
            continue
        if n_shots == 0:
            word_pairs = {'input':[], 'output':[]}
        else:
            word_pairs = dataset['train'][np.random.choice(len(dataset['train']),n_shots, replace=False)]
        word_pairs_test = dataset['test'][j]

        if prefixes is not None and separators is not None:
            prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair = word_pairs_test, prepend_bos_token=prepend_bos, 
                                                    shuffle_labels=shuffle_labels, prefixes=prefixes, separators=separators)
        else:
            prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair = word_pairs_test, prepend_bos_token=prepend_bos, shuffle_labels=shuffle_labels)
            
        # Get relevant parts of the Prompt
        query, target = prompt_data['query_target']['input'], prompt_data['query_target']['output']
        query = query[0] if isinstance(query, list) else query

        if generate_str:
            target = [target] if not isinstance(target, list) else target
        else:
            target = target[0] if isinstance(target, list) else target
        
        sentence = [create_prompt(prompt_data)]
        
        # Figure out token of interest        
        target_token_id = get_answer_id(sentence[0], target, tokenizer)

        if generate_str:
            if metric == "f1_score":
                metric_fn = f1_score
            elif metric == "exact_match_score":
                metric_fn = exact_match_score
            elif metric == "first_word_score":
                metric_fn = first_word_score
            else:
                raise ValueError(f"Unknown metric: {metric}. Recognized metrics: [\"f1_score\", \"exact_match_score\"]")
            clean_output, intervention_output = function_vector_intervention(sentence, target = target, edit_layer = edit_layer, 
                                                                            function_vector = fv_vector,
                                                                            model=model, model_config=model_config, tokenizer=tokenizer, 
                                                                            compute_nll=False, generate_str=generate_str)
            clean_parsed_str, clean_score = parse_generation(clean_output, target, metric_fn)
            intervention_parsed_str, intervention_score = parse_generation(intervention_output, target, metric_fn)
            
            clean_score_list.append(clean_score)
            intervention_score_list.append(intervention_score)

            if pred_file:
                pred_file.write(f"{clean_parsed_str.strip()}\t|||\t{intervention_parsed_str}\n")

        else:
            clean_output, intervention_output = function_vector_intervention(sentence, target = [target], edit_layer = edit_layer, 
                                                                              function_vector = fv_vector,
                                                                              model=model, model_config=model_config, tokenizer=tokenizer, 
                                                                              compute_nll=False) 
        

            clean_rank = compute_individual_token_rank(clean_output, target_token_id)
            intervention_rank = compute_individual_token_rank(intervention_output, target_token_id)
            
            clean_rank_list.append(clean_rank)
            intervention_rank_list.append(intervention_rank)

    if generate_str:
        results = {"clean_score": clean_score_list,
                   "intervention_score": intervention_score_list} 
    else:      
        results = {"clean_topk": [(K, compute_top_k_accuracy(clean_rank_list, K)) for K in range(1,4)],
                   "clean_rank_list": clean_rank_list,
                   
                   "intervention_topk": [(K, compute_top_k_accuracy(intervention_rank_list, K)) for K in range(1,4)],
                   "intervention_rank_list":intervention_rank_list}
    
    if pred_filepath:
        pred_file.close()
    
    return results


# Evaluate few-shot dataset w/o intervention
def n_shot_eval_no_intervention(dataset, n_shots, model, model_config, tokenizer, compute_ppl=True, generate_str=False,
                                shuffle_labels=False, prefixes=None, separators=None, pred_filepath=None,
                                metric="f1_score", test_split='test'):
    """
    Evaluate a model (without any interventions) on the provided ICL dataset.

    Parameters:
    dataset: ICL dataset
    n_shots: the number of ICL examples in each in-context prompt
    model: huggingface model
    model_config: contains model config information (n layers, n heads, etc.)
    tokenizer: huggingface tokenizer
    compute_ppl: whether to compute perplexity of teacher-forced correct completion for base model & intervened model
    generate_str: whether to generate a string of tokens or predict a single token
    shuffle_labels: Whether to shuffle the ICL labels or not
    prefixes: dict of ICL template prefixes for each ICL component (input, output, instructions)
    separators: dict of ICL template separators for each ICL component (input, output, instructions)
    pred_filepath: filepath to save intermediate generations for debugging
    metric: metric to use for longer generations (F1, exact match, etc.)
    test_split: the dataset test split to use as the "test" dataset, typically set to 'test' or 'valid'

    Returns:
    results: dict of topk (k=1,2,3) accuracy on the test_split dataset, for both the model's n-shot
    """
    clean_rank_list = []

    if compute_ppl:
        clean_nll_list = []

    if generate_str:
        score_list = []

    # If the model already prepends a bos token by default, we don't want to add one
    prepend_bos =  False if model_config['prepend_bos'] else True

    if pred_filepath:
        pred_file = open(pred_filepath, 'w')
    else:
        pred_file = None

    for j in tqdm(range(len(dataset[test_split])), total=len(dataset[test_split])):
        if n_shots == 0:
            word_pairs = {'input':[], 'output':[]}
        else:
            word_pairs = dataset['train'][np.random.choice(len(dataset['train']),n_shots, replace=False)]
        word_pairs_test = dataset[test_split][j]
        if prefixes is not None and separators is not None:
            prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair = word_pairs_test, prepend_bos_token=prepend_bos, 
                                                    shuffle_labels=shuffle_labels, prefixes=prefixes, separators=separators)
        else:
            prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair = word_pairs_test, prepend_bos_token=prepend_bos, shuffle_labels=shuffle_labels)
            
        # Get relevant parts of the Prompt
        query, target = prompt_data['query_target']['input'], prompt_data['query_target']['output']
        query = query[0] if isinstance(query, list) else query
        if generate_str:
            target = [target] if not isinstance(target, list) else target
        else:
            target = target[0] if isinstance(target, list) else target
        
        sentence = [create_prompt(prompt_data)]
        
        # Figure out tokens of interest
        target_token_id = get_answer_id(sentence[0], target, tokenizer)
        
        if compute_ppl:
            clean_output, clean_nll = sentence_eval(sentence, target = [target],
                                                    model=model, tokenizer=tokenizer, 
                                                    compute_nll=compute_ppl)
            clean_nll_list.append(clean_nll)
            
        elif generate_str:
            if metric == "f1_score":
                metric_fn = f1_score
            elif metric == "exact_match_score":
                metric_fn = exact_match_score
            elif metric == "first_word_score":
                metric_fn = first_word_score
            else:
                raise ValueError(f"Unknown metric: {metric}. Recognized metrics: [\"f1_score\", \"exact_match_score\"]")
            score = sentence_eval(sentence, target=target, model=model,
                                  tokenizer=tokenizer, compute_nll=False,
                                  generate_str=True, pred_file=pred_file,
                                  metric_fn=metric_fn)
            score_list.append(score)
        else:
            clean_output = sentence_eval(sentence, target = [target],
                                         model=model, tokenizer=tokenizer, compute_nll=False)

        if not generate_str:
            clean_rank = compute_individual_token_rank(clean_output, target_token_id)
            clean_rank_list.append(clean_rank)


    if generate_str:
        results = {"score": score_list}
    else:
        results = {"clean_topk": [(K, compute_top_k_accuracy(clean_rank_list, K)) for K in range(1,4)],
                   "clean_rank_list": clean_rank_list}
    if compute_ppl:
        results['clean_ppl'] = np.exp(clean_nll_list).mean()

    if pred_filepath:
        pred_file.close()
    
    return results


# Logic from huggingface `evaluate` library
def normalize_answer(s):
    """Lowercase text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    """Harmonic mean of pred overlap with gold and gold overlap with pred."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    """Only correct if the prediction matches the entire answer."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def first_word_score(prediction, ground_truth):
    """Only correct if the predicted first word matches the answer's first word."""
    prediction = normalize_answer(prediction).split()
    ground_truth = normalize_answer(ground_truth).split()
    if len(prediction) > 0 and len(ground_truth) > 0:
        return prediction[0] == ground_truth[0]
    else:
        return len(prediction) == len(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """Pick maximum score across possible answers."""
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def parse_generation(output_str, target, metric_fn):
    """Parse a generated string for the target, and score using the specified metric"""
    ans_regex = re.compile("([\w. ]+)[\nQ]*")
    parsed_str = ans_regex.findall(output_str)
    if len(parsed_str) > 0:
        parsed_str = parsed_str[0]
        score = metric_max_over_ground_truths(metric_fn, parsed_str, target)
    else:
        score = 0.0
    
    return parsed_str, score

def make_valid_path_name(path: str):
    """
    Returns an updated path name if given name already exists
    """
    file_name, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = file_name + "_(" + str(counter) + ")" + extension
        counter += 1

    return path

def portability_eval(dataset, fv_vector, edit_layer:int, model, model_config, tokenizer, n_eval_templates:int=20, seed:int=42):
    """
    Evaluates the portability of a function vector when used in prompts with other template forms (different from Q:{}\nA:{}\n\n).

    Parameters:
    dataset: ICL dataset
    fv_vector: torch vector extracted from an LM that triggers a task to be executed by the model
    edit_layer: layer at which to add the function vector
    model: huggingface model
    model_config: dict containing model config parameters (n_layers, n_heads, model name, etc.)
    tokenizer: huggingface tokenizer
    n_eval_templates: number of different templates to use for evaluation
    seed: seed for dataset splitting

    Returns:
    fs_res_dict: dict containing results of few-shot performance on different prompt templates
    zs_res_dict: dict containing results on zero-shot prompt templates
    fs_shuffled_res_dict: dict containing results on few-shot shuffled prompt templates
    templates: list of templates used for evaluation, 
    """
    # Pre-define portability template parts
    all_prefixes = [{'input': 'A:', 'output': 'B:', 'instructions': ''},
                    {'input': 'input:', 'output': 'output:', 'instructions': ''},
                    {'input': 'Input:', 'output': 'Output:', 'instructions': ''},
                    {'input': 'In:', 'output': 'Out:', 'instructions': ''},
                    {'input': 'question:', 'output': 'answer:', 'instructions': ''},
                    {'input': 'Question:', 'output': 'Answer:', 'instructions': ''},
                    {'input': '', 'output': ' ->', 'instructions': ''},
                    {'input': '', 'output': ' :', 'instructions': ''},
                    {'input': 'text:', 'output': 'label:', 'instructions': ''},
                    {'input': 'x:', 'output': 'f(x):', 'instructions': ''},
                    {'input': 'x:', 'output': 'y:', 'instructions': ''},
                    {'input': 'X:', 'output': 'Y:', 'instructions': ''}]

    all_separators=[{'input': ' ', 'output': '', 'instructions': ''},
                    {'input': ' ', 'output': '\n', 'instructions': ''},
                    {'input': ' ', 'output': '\n\n', 'instructions': ''},
                    {'input': '\n', 'output': '\n', 'instructions': ''},
                    {'input': '\n', 'output': '\n\n', 'instructions': ''},
                    {'input': '\n\n', 'output': '\n\n', 'instructions': ''},
                    {'input': ' ', 'output': '|', 'instructions': ''},
                    {'input': '\n', 'output': '|', 'instructions': ''},
                    {'input': '|', 'output': '\n', 'instructions': ''},
                    {'input': '|', 'output': '\n\n', 'instructions': ''}]

    # Choose a random subset of n_eval_templates combinations
    all_combinations = list(itertools.product(all_prefixes, all_separators))
    set_seed(seed)
    random_combos = [list(x) for x in np.array(all_combinations)[np.random.choice(np.arange(len(all_combinations)), n_eval_templates, replace=False)]]

    zs_res_dict = {}
    fs_res_dict = {}
    fs_shuffled_res_dict = {}
    templates = []
    for i,(p,s) in enumerate(random_combos):

        template_repr = p['input'] + '{}' + s['input'] + p['output'] + '{}' + s['output']
        templates.append(template_repr)

        set_seed(seed)
        # FS Eval + Filtering
        fs_results = n_shot_eval_no_intervention(dataset=dataset, n_shots=10, model=model, model_config=model_config, tokenizer=tokenizer, compute_ppl=False, prefixes=p, separators=s)
        filter_set = np.where(np.array(fs_results['clean_rank_list']) == 0)[0]
        fs_res_dict[i] = fs_results

        # ZS Eval
        zs_res_dict[i] = n_shot_eval(dataset, fv_vector, edit_layer, 0, model, model_config, tokenizer, filter_set=filter_set, prefixes=p, separators=s)

        # ZS Eval
        fs_shuffled_res_dict[i] = n_shot_eval(dataset, fv_vector, edit_layer, 10, model, model_config, tokenizer, filter_set=filter_set, prefixes=p, separators=s, shuffle_labels=True)
    
    return fs_res_dict, zs_res_dict,fs_shuffled_res_dict,  templates