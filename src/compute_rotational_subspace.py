from datasets import Dataset

import sys
sys.path.append("../..")

import torch
import seaborn as sns
from tqdm import tqdm, trange

from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from models.configuration_alignable_model import AlignableRepresentationConfig, AlignableConfig
from models.alignable_base import AlignableModel
from models.interventions import BoundlessRotatedSpaceIntervention
from models.llama.modelings_alignable_llama import create_llama
from models.basic_utils import set_seed, count_parameters

from utils.prompt_utils import *
from utils.intervention_utils import *
from utils.model_utils import *
from utils.eval_utils import *
from utils.extract_utils import *

import argparse

def process_no_intervention_eval_dataloader(dataset, model_config, tokenizer, batch_size, n_shots, data_split, prefixes, separators, collate_fn):
    
    torch_dataset = []
    
    is_llama = 'llama' in model_config['name_or_path']
    prepend_bos = not is_llama
    
    
    for i in range(len(dataset[data_split])):
         
        word_pairs = dataset['train'][np.random.choice(len(dataset['train']), n_shots, replace=False)]
        word_pairs_test = dataset[data_split][i]
        prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair=word_pairs_test, prepend_bos_token=prepend_bos, shuffle_labels=False, prefixes=prefixes, separators=separators)
        
        query = prompt_data['query_target']['input']
        target = prompt_data['query_target']['output']
        _, prompt_string = get_token_meta_labels(prompt_data, tokenizer, query)
        
        data_pair = preprocess([prompt_string], [target], tokenizer) 
        torch_dataset.append(data_pair)
        
    torch_dataset = Dataset.from_list(torch_dataset)
    torch_dataset.set_format(type='torch', columns=['input_ids', 'labels'])

    dataloader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return dataloader


def process_train_dataloader(dataset, model_config, tokenizer, batch_size, n_shots, n_trials, prefixes, separators, collate_fn):
    
    torch_dataset = []
    
    is_llama = 'llama' in model_config['name_or_path']
    prepend_bos = not is_llama
    
    for _ in range(n_trials):
        
        noninformative_word_pairs = dataset['train'][np.random.choice(len(dataset['train']), n_shots, replace=False)]
        word_pairs = dataset['train'][np.random.choice(len(dataset['train']), n_shots, replace=False)]
        
        word_pairs_test = dataset['valid'][np.random.choice(len(dataset['valid']), 1, replace=False)]
        
        prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair=word_pairs_test, prepend_bos_token=prepend_bos, shuffle_labels=False, prefixes=prefixes, separators=separators)
        noninformative_prompt_data = word_pairs_to_prompt_data(noninformative_word_pairs, query_target_pair=word_pairs_test, prepend_bos_token=prepend_bos, shuffle_labels=True, prefixes=prefixes, separators=separators)
        
        query = prompt_data['query_target']['input']
        target = prompt_data['query_target']['output']
        
        source_token_labels, prompt_string = get_token_meta_labels(prompt_data, tokenizer, query)
        token_labels, noninformative_prompt_string = get_token_meta_labels(noninformative_prompt_data, tokenizer, query)

        data_pair = preprocess([noninformative_prompt_string], [target], tokenizer)
        data_pair["source_input_ids"] = preprocess([prompt_string], [target], tokenizer)["input_ids"]
        
        assert source_token_labels[-1][2] == "query_predictive_token"
        source_predictive_token_idxs = source_token_labels[-1][0]
        data_pair["source_predictive_token_idxs"] = source_predictive_token_idxs
        
        assert token_labels[-1][2] == "query_predictive_token"
        predictive_token_idxs = token_labels[-1][0]
        data_pair["predictive_token_idxs"] = predictive_token_idxs
        
        torch_dataset.append(data_pair)
    
    torch_dataset = Dataset.from_list(torch_dataset)
    torch_dataset.set_format(type='torch', columns=['input_ids', 'labels', 'source_input_ids', 'source_predictive_token_idxs', 'predictive_token_idxs'])
    train_dataloader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_dataloader


def process_fs_eval_dataloader(dataset, model_config, tokenizer, batch_size, n_shots, data_split, prefixes, separators, collate_fn):
    
    torch_dataset = []
    
    is_llama = 'llama' in model_config['name_or_path']
    prepend_bos = not is_llama
    
    for i in range(len(dataset[data_split])):
        
        noninformative_word_pairs = dataset['train'][np.random.choice(len(dataset['train']), n_shots, replace=False)]
        word_pairs = dataset['train'][np.random.choice(len(dataset['train']), n_shots, replace=False)]

        word_pairs_test = dataset[data_split][i]

        prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair=word_pairs_test, prepend_bos_token=prepend_bos, shuffle_labels=False, prefixes=prefixes, separators=separators)
        noninformative_prompt_data = word_pairs_to_prompt_data(noninformative_word_pairs, query_target_pair=word_pairs_test, prepend_bos_token=prepend_bos, shuffle_labels=True, prefixes=prefixes, separators=separators)

        query = prompt_data['query_target']['input']
        target = prompt_data['query_target']['output']

        source_token_labels, prompt_string = get_token_meta_labels(prompt_data, tokenizer, query)
        token_labels, noninformative_prompt_string = get_token_meta_labels(noninformative_prompt_data, tokenizer, query)

        data_pair = preprocess([noninformative_prompt_string], [target], tokenizer)
        data_pair["source_input_ids"] = preprocess([prompt_string], [target], tokenizer)["input_ids"]
        
        assert source_token_labels[-1][2] == "query_predictive_token"
        source_predictive_token_idxs = source_token_labels[-1][0]
        data_pair["source_predictive_token_idxs"] = source_predictive_token_idxs
        
        assert token_labels[-1][2] == "query_predictive_token"
        predictive_token_idxs = token_labels[-1][0]
        data_pair["predictive_token_idxs"] = predictive_token_idxs
        
        torch_dataset.append(data_pair)
    
    torch_dataset = Dataset.from_list(torch_dataset)
    torch_dataset.set_format(type='torch', columns=['input_ids', 'labels', 'source_input_ids', 'source_predictive_token_idxs', 'predictive_token_idxs'])
    eval_dataloader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return eval_dataloader

def process_zs_eval_dataloader(dataset, model_config, tokenizer, batch_size, n_shots, data_split, prefixes, separators, collate_fn):
    
    torch_dataset = []
    
    is_llama = 'llama' in model_config['name_or_path']
    prepend_bos = not is_llama
    
    for i in range(len(dataset[data_split])):
        
        zs_word_pairs = word_pairs = {'input':[], 'output':[]}
        word_pairs = dataset['train'][np.random.choice(len(dataset['train']), n_shots, replace=False)]

        word_pairs_test = dataset['test'][i]

        prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair=word_pairs_test, prepend_bos_token=prepend_bos, shuffle_labels=False, prefixes=prefixes, separators=separators)
        zs_prompt_data = word_pairs_to_prompt_data(zs_word_pairs, query_target_pair=word_pairs_test, prepend_bos_token=prepend_bos, shuffle_labels=True, prefixes=prefixes, separators=separators)

        query = prompt_data['query_target']['input']
        target = prompt_data['query_target']['output']

        source_token_labels, prompt_string = get_token_meta_labels(prompt_data, tokenizer, query)
        token_labels, zs_prompt_string = get_token_meta_labels(zs_prompt_data, tokenizer, query)

        data_pair = preprocess([zs_prompt_string], [target], tokenizer)
        data_pair["source_input_ids"] = preprocess([prompt_string], [target], tokenizer)["input_ids"]
        
        assert source_token_labels[-1][2] == "query_predictive_token"
        source_predictive_token_idxs = source_token_labels[-1][0]
        data_pair["source_predictive_token_idxs"] = source_predictive_token_idxs
        
        assert token_labels[-1][2] == "query_predictive_token"
        predictive_token_idxs = token_labels[-1][0]
        data_pair["predictive_token_idxs"] = predictive_token_idxs
        
        torch_dataset.append(data_pair)
    
    torch_dataset = Dataset.from_list(torch_dataset)
    torch_dataset.set_format(type='torch', columns=['input_ids', 'labels', 'source_input_ids', 'source_predictive_token_idxs', 'predictive_token_idxs'])
    eval_dataloader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return eval_dataloader


def compute_metrics(eval_preds, eval_labels):
    total_count = 0
    correct_count = 0
    for eval_pred, eval_label in zip(eval_preds, eval_labels):
        
        for i in range(eval_label.shape[0]):
            label_idxs = eval_label[i].ne(IGNORE_INDEX).nonzero().squeeze(-1)
            
            actual_test_labels = eval_label[i][label_idxs].tolist()
            pred_test_labels = [eval_pred[i][idx].argmax(dim=-1) for idx in label_idxs]
            
            correct = (actual_test_labels==pred_test_labels)

            total_count += 1
            if correct:
                correct_count += 1
                
    accuracy = round(correct_count/total_count, 2)
    return {"accuracy": accuracy}


def simple_boundless_das_position_config(model_type, intervention_type, layer):
    alignable_config = AlignableConfig(
        alignable_model_type=model_type,
        alignable_representations=[
            AlignableRepresentationConfig(
                layer,             # layer
                intervention_type, # intervention type
                "pos",             # intervention unit
                1                  # max number of unit
            ),
        ],
        alignable_interventions_type=BoundlessRotatedSpaceIntervention,
    )
    return alignable_config


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset_name', help='Name of the dataset to be loaded', type=str, required=True)
    parser.add_argument('--edit_layer', help='Layer for intervention.', type=int, required=False, default=16)
    parser.add_argument('--model_name', help='Name of model to be loaded', type=str, required=False, default='/data/public_models/llama/llama_hf_weights/llama-7b/')
    parser.add_argument('--root_data_dir', help='Root directory of data files', type=str, required=False, default='../dataset_files')
    parser.add_argument('--save_path_root', help='File path to save to', type=str, required=False, default="../results/ICL-DAS/llama-7b")
    parser.add_argument('--seed', help='Randomized seed', type=int, required=False, default=42)
    parser.add_argument('--device', help='Device to run on',type=str, required=False, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--test_split', help="Percentage corresponding to test set split size", required=False, default=0.3)    
    parser.add_argument('--n_shots', help="Number of shots in each in-context prompt", type=int, required=False, default=10)
    parser.add_argument('--n_trials', help="Number of in-context prompts to average over for indirect_effect", type=int, required=False, default=512)
    parser.add_argument('--prefixes', help='Prompt template prefixes to be used', required=False, default={"input":"Q:", "output":"A:", "instructions":""})
    parser.add_argument('--separators', help='Prompt template separators to be used', required=False, default={"input":"\n", "output":"\n\n", "instructions":""})
    
    # Intervention hyperparameters
    parser.add_argument('--batch_size', help='Batch size of inference and training intervention', type=int, required=False, default=32)
    parser.add_argument('--gradient_accumulation_steps', help='Batch size of inference and training intervention', type=int, required=False, default=4)
    parser.add_argument('--epochs', help='Batch size of inference and training intervention', type=int, required=False, default=25)
    parser.add_argument('--warnup_ratio', help='Batch size of inference and training intervention', type=float, required=False, default=0.1)
    parser.add_argument('--rotate_lr', help='Batch size of inference and training intervention', type=float, required=False, default=1e-3)
    parser.add_argument('--boundary_lr', help='Batch size of inference and training intervention', type=float, required=False, default=1e-2)
    
    parser.add_argument('--temperature_start', help='Batch size of inference and training intervention', type=float, required=False, default=50.0)
    parser.add_argument('--temperature_end', help='Batch size of inference and training intervention', type=float, required=False, default=0.1)
    
    args = parser.parse_args()
    
    dataset_name = args.dataset_name
    edit_layer = args.edit_layer
    model_name = args.model_name
    root_data_dir = args.root_data_dir
    save_path_root = f"{args.save_path_root}/{dataset_name}/L{str(edit_layer)}"
    seed = args.seed
    device = args.device

    test_split = float(args.test_split)
    n_shots = args.n_shots
    n_trials = args.n_trials
    
    prefixes = load_prefixes_or_separators(args.prefixes) 
    separators = load_prefixes_or_separators(args.separators)
    
    batch_size = args.batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    epochs = args.epochs
    warnup_ratio = args.warnup_ratio
    rotate_lr = args.rotate_lr
    boundary_lr = args.boundary_lr
    
    temperature_start = args.temperature_start
    temperature_end = args.temperature_end
    
    results = dict()
    
    print(args)
    
    # Load Model & Tokenizer
    print("Loading Model")
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(model_name, device=device)
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    assert edit_layer < model_config["n_layers"], f"Edit layer {edit_layer} is out of range for model with {model_config['n_layers']} layers."
    
    alignable_config = simple_boundless_das_position_config(type(model), "block_output", edit_layer)
    alignable = AlignableModel(alignable_config, model)
    alignable.set_device(device)
    alignable.disable_model_gradients()
    
    def vanilla_collate_fn(batch):
        input_ids, labels = tuple([data_pair[key] for data_pair in batch] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
        )
        
    def intervention_collate_fn(batch):
        input_ids, labels, source_input_ids, source_predictive_token_idxs, predictive_token_idxs = tuple(
            [data_pair[key] for data_pair in batch] for key in 
            ('input_ids', 'labels', 'source_input_ids', 'source_predictive_token_idxs', 'predictive_token_idxs')
        )
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        
        source_input_ids = torch.nn.utils.rnn.pad_sequence(
            source_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        source_predictive_token_idxs = torch.LongTensor(source_predictive_token_idxs)
        predictive_token_idxs = torch.LongTensor(predictive_token_idxs)
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
            source_input_ids=source_input_ids,
            source_attention_mask=source_input_ids.ne(tokenizer.pad_token_id),
            predictive_token_idxs=predictive_token_idxs,
            source_predictive_token_idxs=source_predictive_token_idxs
        )
    
    def calculate_loss(logits, labels):
        shift_logits = logits[..., :, :].contiguous()
        shift_labels = labels[..., :].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, alignable.model_config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        
        for k, v in alignable.interventions.items():
            boundary_loss = 1. * v[0].intervention_boundaries.sum()
        loss += boundary_loss
        
        return loss
    
    # Load the dataset
    print("Loading Dataset")
    set_seed(seed)
    dataset = load_dataset(dataset_name, root_data_dir=root_data_dir, test_size=test_split, seed=seed)
    
    if not os.path.exists(save_path_root):
        os.makedirs(save_path_root)
        
    print("Processing Dataloaders")
    eval_no_intervention_dataloader = process_no_intervention_eval_dataloader(dataset, model_config, tokenizer, batch_size, n_shots, "valid", prefixes, separators, vanilla_collate_fn)
    train_dataloader = process_train_dataloader(dataset, model_config, tokenizer, batch_size, n_shots, n_trials, prefixes, separators, intervention_collate_fn)
    fs_eval_dataloader = process_fs_eval_dataloader(dataset, model_config, tokenizer, batch_size, n_shots, "test", prefixes, separators, intervention_collate_fn)
    zs_eval_dataloader = process_zs_eval_dataloader(dataset, model_config, tokenizer, batch_size, n_shots, "test", prefixes, separators, intervention_collate_fn)
    
    print(f"Evaluating the model {n_shots}-shots without intervention...")
    
    total_count = 0
    correct_count = 0
    with torch.no_grad():
        for step, inputs in enumerate(tqdm(eval_no_intervention_dataloader)):
            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to(model.device)
                        
            # aligning forward!
            outputs = model(
                input_ids=inputs['input_ids'],
                labels=inputs['labels'],
                attention_mask=inputs['attention_mask']
            )
            
            for i in range(inputs['input_ids'].shape[0]):
                label_idxs = inputs['labels'][i].ne(IGNORE_INDEX).nonzero().squeeze(-1)
                
                actual_test_labels = inputs['labels'][i][label_idxs].tolist()
                pred_test_labels = [outputs.logits[i][idx].argmax(dim=-1) for idx in label_idxs]
                
                correct = (actual_test_labels==pred_test_labels)

                total_count += 1
                if correct:
                    correct_count += 1
                    
    current_acc = round(correct_count/total_count, 2)
    print(f"[WARNING: THIS NEEDS TO BE GOOD!] prealign task accuracy: {current_acc}")
    
    results['prealign_val_task_accuracy'] = current_acc
    
    
    t_total = int(len(train_dataloader) * epochs)
    warm_up_steps = 0.1 * t_total
    optimizer_params = []
    
    for k, v in alignable.interventions.items():
        optimizer_params += [{'params': v[0].rotate_layer.parameters()}]
        optimizer_params += [{'params': v[0].intervention_boundaries, 'lr': boundary_lr}]
    
    optimizer = torch.optim.Adam(
        optimizer_params,
        lr=rotate_lr,
    )
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warm_up_steps,
        num_training_steps=t_total
    )
    
    target_total_step = len(train_dataloader) * epochs
    
    temperature_schedule = torch.linspace(
        temperature_start, temperature_end, target_total_step
    ).to(torch.bfloat16).to(device)
    
    total_step = 0
    alignable.set_temperature(temperature_schedule[total_step])
    
    alignable.model.train() # train enables drop-off but no grads
    print("llama trainable parameters: ", count_parameters(alignable.model))
    print("intervention trainable parameters: ", alignable.count_parameters())
    
    train_iterator = trange(
        0, int(epochs), desc="Epoch"
    )
    
    for epoch in train_iterator:
        epoch_iterator = tqdm(
            train_dataloader, desc=f"Epoch: {epoch}", position=0, leave=True
        )
        for step, inputs in enumerate(epoch_iterator):
            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)
            b_s = inputs["input_ids"].shape[0]

        source2base = ([[[idx] for idx in inputs["source_predictive_token_idxs"].tolist()]], [[[idx] for idx in inputs["predictive_token_idxs"].tolist()]])
        
        _, counterfactual_outputs = alignable(
            {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]},
            [{"input_ids": inputs["source_input_ids"], "attention_mask": inputs["source_attention_mask"]}],
            {"sources->base": source2base}
        )
        
        eval_metrics = compute_metrics(
            [counterfactual_outputs.logits], [inputs['labels']]
        )
        
        loss = calculate_loss(
            counterfactual_outputs.logits, inputs["labels"]
        )
        loss_str = round(loss.item(), 2)
        epoch_iterator.set_postfix({'loss': loss_str, 'acc': eval_metrics["accuracy"]})
        
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
            
        loss.backward()
        if total_step % gradient_accumulation_steps == 0:
            if not (gradient_accumulation_steps > 1 and total_step == 0):
                optimizer.step()
                scheduler.step()
                alignable.set_zero_grad()
                alignable.set_temperature(temperature_schedule[total_step])
        total_step += 1
    
    print("Evaluation the model with intervention...")
    
    eval_labels = []
    eval_preds = []
    with torch.no_grad():
        epoch_iterator = tqdm(fs_eval_dataloader, desc=f"Test")
        for step, inputs in enumerate(epoch_iterator):
            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to("cuda")
            b_s = inputs["input_ids"].shape[0]
            
            source2base = ([[[idx] for idx in inputs["source_predictive_token_idxs"].tolist()]], [[[idx] for idx in inputs["predictive_token_idxs"].tolist()]])
            
            _, counterfactual_outputs = alignable(
                {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]},
                [{"input_ids": inputs["source_input_ids"], "attention_mask": inputs["source_attention_mask"]}],
                {"sources->base": source2base}
            )
            
            eval_labels += [inputs['labels']]
            eval_preds += [counterfactual_outputs.logits]
    eval_metrics = compute_metrics(eval_preds, eval_labels)
    results['fs_shuffled_with_intervention_accuracy'] = eval_metrics["accuracy"]
    
    total_count = 0
    correct_count = 0
    with torch.no_grad():
        for step, inputs in enumerate(tqdm(fs_eval_dataloader)):
            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to(model.device)
                        
            # aligning forward!
            outputs = model(
                input_ids=inputs['input_ids'],
                labels=inputs['labels'],
                attention_mask=inputs['attention_mask']
            )
            
            for i in range(inputs['input_ids'].shape[0]):
                label_idxs = inputs['labels'][i].ne(IGNORE_INDEX).nonzero().squeeze(-1)
                # label_idxs = label_idxs[1: ]
                left_shifted_idxs = label_idxs
                
                actual_test_labels = inputs['labels'][i][label_idxs].tolist()
                pred_test_labels = [outputs.logits[i][idx].argmax(dim=-1) for idx in left_shifted_idxs]
                
                correct = (actual_test_labels==pred_test_labels)

                total_count += 1
                if correct:
                    correct_count += 1
                    
    current_acc = round(correct_count/total_count, 2)
    results['fs_shuffled_no_intervention_accuracy'] = current_acc
    
    # evaluation on the test set
    eval_labels = []
    eval_preds = []
    with torch.no_grad():
        epoch_iterator = tqdm(zs_eval_dataloader, desc=f"Test")
        for step, inputs in enumerate(epoch_iterator):
            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to("cuda")
            b_s = inputs["input_ids"].shape[0]
            
            source2base = ([[[idx] for idx in inputs["source_predictive_token_idxs"].tolist()]], [[[idx] for idx in inputs["predictive_token_idxs"].tolist()]])
            
            _, counterfactual_outputs = alignable(
                {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]},
                [{"input_ids": inputs["source_input_ids"], "attention_mask": inputs["source_attention_mask"]}],
                {"sources->base": source2base}
            )
            
            eval_labels += [inputs['labels']]
            eval_preds += [counterfactual_outputs.logits]
    eval_metrics = compute_metrics(eval_preds, eval_labels)
    results['zs_with_intervention_accuracy'] = eval_metrics["accuracy"]
    
    total_count = 0
    correct_count = 0
    with torch.no_grad():
        for step, inputs in enumerate(tqdm(zs_eval_dataloader)):
            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to(model.device)
                        
            # aligning forward!
            outputs = model(
                input_ids=inputs['input_ids'],
                labels=inputs['labels'],
                attention_mask=inputs['attention_mask']
            )
            
            for i in range(inputs['input_ids'].shape[0]):
                label_idxs = inputs['labels'][i].ne(IGNORE_INDEX).nonzero().squeeze(-1)
                # label_idxs = label_idxs[1: ]
                left_shifted_idxs = label_idxs
                
                actual_test_labels = inputs['labels'][i][label_idxs].tolist()
                pred_test_labels = [outputs.logits[i][idx].argmax(dim=-1) for idx in left_shifted_idxs]
                
                correct = (actual_test_labels==pred_test_labels)

                total_count += 1
                if correct:
                    correct_count += 1
                    
    current_acc = round(correct_count/total_count, 2)
    results['zs_no_intervention_accuracy'] = current_acc
    
    
    print("Few-shot shuffled with intervention accuracy: " + str(results['fs_shuffled_with_intervention_accuracy']))
    print("Few-shot shuffled no intervention accuracy: " + str(results['fs_shuffled_no_intervention_accuracy']))
    print("Zero-shot with intervention accuracy: " + str(results['zs_with_intervention_accuracy']))
    print("Zero-shot no intervention accuracy: " + str(results['zs_no_intervention_accuracy']))
    
    print("Saving results...")
    
    with open(f"{save_path_root}/args.json", "w") as f:
        json.dump(vars(args), f, indent=4)
        f.close()
        
    with open(f"{save_path_root}/results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    assert len(alignable.interventions.keys()) == 1
    key = list(alignable.interventions.keys())[0]
        
    orthogonal_matrix = alignable.interventions[key][0].rotate_layer.weight.detach().cpu()
    torch.save(orthogonal_matrix, f"{save_path_root}/orthogonal_matrix.pt")
    
    print("Done!")
    
    
    
    
    
    
        
        
        
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    