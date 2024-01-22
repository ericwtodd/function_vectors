from datasets import Dataset, concatenate_datasets

import sys
sys.path.append("../..")

import torch
import seaborn as sns
from tqdm import tqdm, trange

from transformers import get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from pyvene.models.configuration_intervenable_model import IntervenableRepresentationConfig, IntervenableConfig
from pyvene.models.intervenable_base import IntervenableModel
from pyvene.models.interventions import BoundlessRotatedSpaceIntervention
from pyvene.models.basic_utils import set_seed, count_parameters


from utils.prompt_utils import *
from utils.intervention_utils import *
from utils.model_utils import *
from utils.eval_utils import *
from utils.extract_utils import *
from das_utils import *

import argparse


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset_names', help='Name of the datasets to be loaded', type=list, required=False, default=["antonym", "capitalize", "country-currency", "english-french", "present-past", "singular-plural"])
    parser.add_argument('--edit_layer', help='Layer for intervention.', type=int, required=False, default=16)
    parser.add_argument('--model_name', help='Name of model to be loaded', type=str, required=False, default='/work/frink/models/llama_7b')
    parser.add_argument('--root_data_dir', help='Root directory of data files', type=str, required=False, default='../dataset_files')
    parser.add_argument('--save_path_root', help='File path to save to', type=str, required=False, default="../results/Aggregate-Two")
    parser.add_argument('--intervention_path_root', help='Path to the trained intervention model', type=str, required=False, default=None)
    parser.add_argument('--seed', help='Randomized seed', type=int, required=False, default=42)
    parser.add_argument('--device', help='Device to run on',type=str, required=False, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--test_split', help="Percentage corresponding to test set split size", required=False, default=0.3)    
    parser.add_argument('--n_shots', help="Number of shots in each in-context prompt", type=int, required=False, default=10)
    parser.add_argument('--n_trials', help="Number of in-context prompts to average over for indirect_effect", type=int, required=False, default=512)
    parser.add_argument('--prefixes', help='Prompt template prefixes to be used', required=False, default={"input":"Q:", "output":"A:", "instructions":""})
    parser.add_argument('--separators', help='Prompt template separators to be used', required=False, default={"input":"\n", "output":"\n\n", "instructions":""})
    
    parser.add_argument('--training_method',type=str, required=False, default='both', choices=['noninformative', 'zero_shot', 'both'])
        
    # Intervention hyperparameters
    parser.add_argument('--batch_size', help='Batch size of inference and training intervention', type=int, required=False, default=32)
    parser.add_argument('--gradient_accumulation_steps', help='Batch size of inference and training intervention', type=int, required=False, default=1)
    parser.add_argument('--epochs', help='Batch size of inference and training intervention', type=int, required=False, default=35)
    parser.add_argument('--warnup_ratio', help='Batch size of inference and training intervention', type=float, required=False, default=0.1)
    parser.add_argument('--rotate_lr', help='Batch size of inference and training intervention', type=float, required=False, default=1e-3)
    parser.add_argument('--boundary_lr', help='Batch size of inference and training intervention', type=float, required=False, default=1e-2)
    
    parser.add_argument('--temperature_start', help='Batch size of inference and training intervention', type=float, required=False, default=50.0)
    parser.add_argument('--temperature_end', help='Batch size of inference and training intervention', type=float, required=False, default=0.1)
    
    parser.add_argument('--evaluate_per_epoch', help='Whether or not to run and save the results of eval during training', required=False, default=True)
    
    args = parser.parse_args()
    
    dataset_names = args.dataset_names
    edit_layer = args.edit_layer
    model_name = args.model_name
    root_data_dir = args.root_data_dir
    save_path_root = f"{args.save_path_root}/L{str(edit_layer)}"
    
    intervention_path_root = args.intervention_path_root
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
    
    evaluate_per_epoch = args.evaluate_per_epoch
    training_method = args.training_method
    
    results = dict()
    
    print(args)
    
    # Load Model & Tokenizer
    print("Loading Model")
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(model_name, device=device)
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    assert edit_layer < model_config["n_layers"], f"Edit layer {edit_layer} is out of range for model with {model_config['n_layers']} layers."
    
    if intervention_path_root is not None:
        print(f"Loading the intervention model from {intervention_path_root}/intervention_model...")
        intervenable = IntervenableModel.load(f"{intervention_path_root}/intervention_model", model=model)
    else:
        intervenable_config = simple_boundless_das_position_config(type(model), "block_output", edit_layer)
        intervenable = IntervenableModel(intervenable_config, model)
        
    intervenable.set_device(device)
    intervenable.disable_model_gradients()
        
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
        base_input_ids, base_labels, source_input_ids, source_labels, source_predictive_token_idxs, predictive_token_idxs = tuple(
            [data_pair[key] for data_pair in batch] for key in 
            ('base_input_ids', 'base_labels', 'source_input_ids', 'source_labels', 'source_predictive_token_idxs', 'predictive_token_idxs')
        )
        
        base_input_ids = torch.nn.utils.rnn.pad_sequence(
            base_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        
        source_input_ids = torch.nn.utils.rnn.pad_sequence(
            source_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        
        base_labels = torch.nn.utils.rnn.pad_sequence(base_labels, batch_first=True, padding_value=IGNORE_INDEX)
        source_labels = torch.nn.utils.rnn.pad_sequence(source_labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        source_predictive_token_idxs = torch.LongTensor(source_predictive_token_idxs)
        predictive_token_idxs = torch.LongTensor(predictive_token_idxs)
        
        return dict(
            base_input_ids=base_input_ids,
            base_labels=base_labels,
            base_attention_mask=base_input_ids.ne(tokenizer.pad_token_id),
            source_input_ids=source_input_ids,
            source_labels=source_labels,
            source_attention_mask=source_input_ids.ne(tokenizer.pad_token_id),
            predictive_token_idxs=predictive_token_idxs,
            source_predictive_token_idxs=source_predictive_token_idxs
        )
    
    def calculate_loss(logits, labels):
        shift_logits = logits[..., :, :].contiguous()
        shift_labels = labels[..., :].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, intervenable.model_config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        
        for k, v in intervenable.interventions.items():
            boundary_loss = 1. * v[0].intervention_boundaries.sum()
        loss += boundary_loss
        
        return loss
    
    # Load the dataset
    print("Loading Dataset")
    set_seed(seed)
    datasets = [load_dataset(dataset_name, root_data_dir=root_data_dir, test_size=test_split, seed=seed) for dataset_name in dataset_names]
    
    if not os.path.exists(save_path_root):
        os.makedirs(save_path_root)
        
    print("Processing Dataloaders")
    
    eval_no_intervention_dataloader = process_mixed_dataloader(datasets, model_config, tokenizer, batch_size, n_shots, "valid", prefixes, separators, intervention_collate_fn, ablation_method="zero_shot")
    if training_method == "both":
        
        all_datasets = []
        for method in ["zero_shot", "noninformative"]:
            for dataset in datasets:
                all_datasets.append(process_dataset(dataset, model_config, tokenizer, n_shots, "train", prefixes, separators, n_trials=n_trials, ablation_method=method))
        train_dataset = concatenate_datasets(all_datasets)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=intervention_collate_fn)
    else:
        train_dataloader = process_mixed_dataloader(datasets, model_config, tokenizer, batch_size, n_shots, "train", prefixes, separators, intervention_collate_fn, n_trials=n_trials, ablation_method=training_method)
        
    fs_eval_dataloader = process_mixed_dataloader(datasets, model_config, tokenizer, batch_size, n_shots, "valid", prefixes, separators, intervention_collate_fn, ablation_method="noninformative")
    zs_eval_dataloader = process_mixed_dataloader(datasets, model_config, tokenizer, batch_size, n_shots, "valid", prefixes, separators, intervention_collate_fn, ablation_method="zero_shot")
    
    print(f"Evaluating the model {n_shots}-shots without intervention...")
    eval_accuracy = evaluate(intervenable, eval_no_intervention_dataloader, device=model.device, intervene=False, corrupt=False)
    print(f"[WARNING: THIS NEEDS TO BE GOOD!] prealign task accuracy: {eval_accuracy}")
    results['prealign_val_task_accuracy'] = eval_accuracy
    
    t_total = int(len(train_dataloader) * epochs)
    warm_up_steps = 0.1 * t_total
    optimizer_params = []
    
    for k, v in intervenable.interventions.items():
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
    intervenable.set_temperature(temperature_schedule[total_step])
    
    intervenable.model.train() # train enables drop-off but no grads
    print("llama trainable parameters: ", count_parameters(intervenable.model))
    print("intervention trainable parameters: ", intervenable.count_parameters())
    
    train_iterator = trange(
        0, int(epochs), desc="Epoch"
    )
    
    training_log_dicts = None
    
    if intervention_path_root is None:
        
        os.makedirs(os.path.join(save_path_root, "checkpoints"), exist_ok=True)
        
        training_log_dicts = []
                
        for epoch in train_iterator:
            
            log_dicts = []
            ckpt_path = os.path.join(save_path_root, "checkpoints", f"epoch_{epoch}")
            
            epoch_iterator = tqdm(
                train_dataloader, desc=f"Epoch: {epoch}", position=0, leave=True
            )
            
            for step, inputs in enumerate(epoch_iterator):
                for k, v in inputs.items():
                    if v is not None and isinstance(v, torch.Tensor):
                        inputs[k] = v.to(device)
                b_s = inputs["base_input_ids"].shape[0]

                source2base = ([[[idx] for idx in inputs["source_predictive_token_idxs"].tolist()]], [[[idx] for idx in inputs["predictive_token_idxs"].tolist()]])
                
                _, counterfactual_outputs = intervenable(
                    {"input_ids": inputs["base_input_ids"], "attention_mask": inputs["base_attention_mask"]},
                    [{"input_ids": inputs["source_input_ids"], "attention_mask": inputs["source_attention_mask"]}],
                    {"sources->base": source2base}
                )
                
                eval_metrics = compute_metrics(
                    [counterfactual_outputs.logits], [inputs['base_labels']]
                )
                
                loss = calculate_loss(
                    counterfactual_outputs.logits, inputs["base_labels"]
                )
                loss_str = round(loss.item(), 2)
                
                log_dict = {'loss': loss_str, 'acc': eval_metrics["accuracy"], 'sparsity': compute_rotation_mask_sparsity(intervenable)}
                epoch_iterator.set_postfix(log_dict)
                
                log_dicts.append(log_dict)
                
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps
                    
                loss.backward()
                if total_step % gradient_accumulation_steps == 0:
                    if not (gradient_accumulation_steps > 1 and total_step == 0):
                        optimizer.step()
                        scheduler.step()
                        intervenable.set_zero_grad()
                        intervenable.set_temperature(temperature_schedule[total_step])
                        
                total_step += 1
            
            ave_loss = round(sum([log_dict['loss'] for log_dict in log_dicts])/len(log_dicts), 4)
            ave_acc = round(sum([log_dict['acc'] for log_dict in log_dicts])/len(log_dicts), 4)
            ave_sparsity = round(sum([log_dict['sparsity'] for log_dict in log_dicts])/len(log_dicts), 4) 
            
            epoch_training_log = {'loss': ave_loss, 'acc': ave_acc, 'sparsity': ave_sparsity}
            print("Epoch " + str(epoch) + " finished! Training loss: " + str(ave_loss) + ", training acc: " + str(ave_acc) + ", sparsity: " + str(ave_sparsity))
            
            if evaluate_per_epoch:
                
                fs_shuffled_acc = evaluate(intervenable, fs_eval_dataloader, device=model.device, intervene=True)
                epoch_training_log['fs_shuffled_with_intervention_accuracy'] = fs_shuffled_acc
                
                zs_intervention_acc = evaluate(intervenable, zs_eval_dataloader, device=model.device, intervene=True)
                epoch_training_log['zs_with_intervention_accuracy'] = zs_intervention_acc
                
                print("Few-shot shuffled with intervention accuracy: " + str(epoch_training_log['fs_shuffled_with_intervention_accuracy']))
                print("Zero-shot with intervention accuracy: " + str(epoch_training_log['zs_with_intervention_accuracy']))
            
            intervenable.save(ckpt_path)
            training_log_dicts.append(epoch_training_log)
        
    print("Evaluation the model with intervention...")
    
    fs_shuffled_acc = evaluate(intervenable, fs_eval_dataloader, device=model.device, intervene=True)
    results['fs_shuffled_with_intervention_accuracy'] = fs_shuffled_acc
    
    fs_shuffled_no_intervention_acc = evaluate(intervenable, fs_eval_dataloader, device=model.device, intervene=False, corrupt=True)
    results['fs_shuffled_no_intervention_accuracy'] = fs_shuffled_no_intervention_acc
    
    zs_intervention_acc = evaluate(intervenable, zs_eval_dataloader, device=model.device, intervene=True)
    results['zs_with_intervention_accuracy'] = zs_intervention_acc
        
    zs_no_intervention_acc = evaluate(intervenable, zs_eval_dataloader, device=model.device, intervene=False, corrupt=True)
    results['zs_no_intervention_accuracy'] = zs_no_intervention_acc
    
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
        
    if training_log_dicts is not None:
        with open(f"{save_path_root}/training_log.json", "w") as f:
            json.dump(training_log_dicts, f, indent=4)
            f.close()
            
    intervenable.save(f"{save_path_root}/intervention_model")
    print("Done!")
    