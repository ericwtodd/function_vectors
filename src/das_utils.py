from datasets import Dataset, concatenate_datasets
from torch.utils.data import DataLoader

import torch
from tqdm import tqdm
import numpy as np
from pyvene.models.configuration_intervenable_model import IntervenableRepresentationConfig, IntervenableConfig
from pyvene.models.intervenable_base import IntervenableModel
from pyvene.models.interventions import BoundlessRotatedSpaceIntervention
from pyvene.models.basic_utils import set_seed, count_parameters, sigmoid_boundary

from utils.prompt_utils import *
from utils.intervention_utils import *
from utils.model_utils import *
from utils.eval_utils import *
from utils.extract_utils import *


def get_rotation_mask(intervenable_model: IntervenableModel):
    
    assert len(intervenable_model.interventions.keys()) == 1
    
    intervention_key = list(intervenable_model.interventions.keys())[0]
    
    intervention = intervenable_model.interventions[intervention_key][0]
    intervention_boundaries = torch.clamp(intervention.intervention_boundaries, 1e-3, 1)
    boundary_mask = sigmoid_boundary(
        intervention.intervention_population.repeat(1, 1),
        0.0,
        intervention_boundaries[0] * int(intervention.embed_dim),
        intervention.temperature
    )
    
    return boundary_mask

def compute_rotation_mask_sparsity(intervenable_model: IntervenableModel):
        
        rotation_mask = get_rotation_mask(intervenable_model)
        return (rotation_mask.sum() / rotation_mask.numel()).item()
    

def compute_metrics(eval_preds, eval_labels):
    total_count = 0
    correct_count = 0
    for eval_pred, eval_label in zip(eval_preds, eval_labels):
        
        for i in range(eval_label.shape[0]):
            label_idxs = eval_label[i].ne(IGNORE_INDEX).nonzero().squeeze(-1)
            
            actual_test_labels = eval_label[i][label_idxs].tolist()
            pred_test_labels = [eval_pred[i][idx].argmax(dim=-1) for idx in label_idxs]
            
            correct = actual_test_labels==pred_test_labels
            
            total_count += 1
            if correct:
                correct_count += 1
                
    accuracy = round(correct_count/total_count, 2)
    return {"accuracy": accuracy}


def evaluate(intervenable_model, dataloader, device="cuda", intervene=False, corrupt=False):
    
    with torch.no_grad():
        
        eval_labels = []
        eval_preds = []
        
        for step, inputs in enumerate(tqdm(dataloader)):
            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)
                    
            if not intervene:
                
                if corrupt:
                    outputs = intervenable_model.model(
                        input_ids=inputs['base_input_ids'],
                        labels=inputs['base_labels'],
                        attention_mask=inputs['base_attention_mask']
                    )
                    eval_labels += [inputs['base_labels']]
                else:
                    outputs = intervenable_model.model(
                        input_ids=inputs['source_input_ids'],
                        labels=inputs['source_labels'],
                        attention_mask=inputs['source_attention_mask']
                    )
                    eval_labels += [inputs['source_labels']]
                
            else:           
                source2base = ([[[idx] for idx in inputs["source_predictive_token_idxs"].tolist()]], [[[idx] for idx in inputs["predictive_token_idxs"].tolist()]])
                
                _, outputs = intervenable_model(
                    {"input_ids": inputs["base_input_ids"], "attention_mask": inputs["base_attention_mask"]},
                    [{"input_ids": inputs["source_input_ids"], "attention_mask": inputs["source_attention_mask"]}],
                    {"sources->base": source2base}
                )
                eval_labels += [inputs['base_labels']]
            eval_preds += [outputs.logits]
        
        eval_metrics = compute_metrics(eval_preds, eval_labels)
        return eval_metrics["accuracy"]


def simple_boundless_das_position_config(model_type, intervention_type, layer):
    intervenable_config = IntervenableConfig(
        intervenable_model_type=model_type,
        intervenable_representations=[
            IntervenableRepresentationConfig(
                layer,             # layer
                intervention_type, # intervention type
                "pos",             # intervention unit
                1                  # max number of unit
            ),
        ],
        intervenable_interventions_type=BoundlessRotatedSpaceIntervention,
    )
    return intervenable_config


def process_dataset(dataset, model_config, tokenizer, n_shots, data_split, prefixes, separators, n_trials=None, ablation_method="zero_shot"):
    
    
    assert ablation_method in ["zero_shot", "noninformative", "none"]
    torch_dataset = []
    
    is_llama = 'llama' in model_config['name_or_path']
    prepend_bos = not is_llama
    
    
    
    if n_trials is None:
        sample_idxs = range(len(dataset[data_split]))
    else:
        sample_idxs = np.random.choice(len(dataset[data_split]), n_trials, replace=True).tolist()
    
    for i in sample_idxs:
        
        data_pair = {}
        
        word_pairs = dataset['train'][np.random.choice(len(dataset['train']), n_shots, replace=False)]  
        word_pairs_test = dataset[data_split][i]
        
        prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair=word_pairs_test, prepend_bos_token=prepend_bos, shuffle_labels=False, prefixes=prefixes, separators=separators)
        
        query = prompt_data['query_target']['input']
        target = prompt_data['query_target']['output']
        
        source_token_labels, prompt_string = get_token_meta_labels(prompt_data, tokenizer, query)
        
        source_batch = preprocess([prompt_string], [target], tokenizer)
        
        data_pair["source_input_ids"] = source_batch["input_ids"]
        data_pair["source_labels"] = source_batch["labels"]
        
        assert source_token_labels[-1][2] == "query_predictive_token"
        source_predictive_token_idxs = source_token_labels[-1][0]
        data_pair["source_predictive_token_idxs"] = source_predictive_token_idxs
        
        if ablation_method == "none":
            pass
        else:
            if ablation_method == "zero_shot":
                base_word_pairs = {'input':[], 'output':[]}
            elif ablation_method == "noninformative":
                base_word_pairs = dataset['train'][np.random.choice(len(dataset['train']), n_shots, replace=False)]
            else:
                raise ValueError(f"ablation_method {ablation_method} is not supported.")

            base_prompt_data = word_pairs_to_prompt_data(base_word_pairs, query_target_pair=word_pairs_test, prepend_bos_token=prepend_bos, shuffle_labels=True, prefixes=prefixes, separators=separators)
            token_labels, base_prompt_string = get_token_meta_labels(base_prompt_data, tokenizer, query)
            
            base_batch = preprocess([base_prompt_string], [target], tokenizer)
            data_pair["base_input_ids"] = base_batch["input_ids"]
            data_pair["base_labels"] = base_batch["labels"]
            
            assert token_labels[-1][2] == "query_predictive_token"
            predictive_token_idxs = token_labels[-1][0]
            data_pair["predictive_token_idxs"] = predictive_token_idxs
            
        torch_dataset.append(data_pair)
            
    torch_dataset = Dataset.from_list(torch_dataset)
    torch_dataset.set_format(type='torch')
    return torch_dataset


def process_dataloader(dataset, model_config, tokenizer, batch_size, n_shots, data_split, prefixes, separators, collate_fn, n_trials=None, ablation_method="zero_shot", shuffle=False):
    
    torch_dataset = process_dataset(dataset, model_config, tokenizer, n_shots, data_split, prefixes, separators, n_trials, ablation_method)
    torch_dataloader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return torch_dataloader


def process_mixed_dataloader(datasets, model_config, tokenizer, batch_size, n_shots, data_split, prefixes, separators, collate_fn, n_trials=None, ablation_method="zero_shot"):
    
    all_dataset = []
    for dataset in datasets:
        torch_dataset = process_dataset(dataset, model_config, tokenizer, n_shots, data_split, prefixes, separators, n_trials, ablation_method)
        all_dataset.append(torch_dataset)
    
    all_dataset = concatenate_datasets(all_dataset)
    torch_dataloader = DataLoader(all_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return torch_dataloader
