import os
import torch, numpy as np
import argparse

# Include prompt creation helper functions
from utils.prompt_utils import load_dataset
from utils.extract_utils import get_mean_head_activations, compute_universal_function_vector
from utils.eval_utils import n_shot_eval_no_intervention, n_shot_eval
from utils.model_utils import load_gpt_model_and_tokenizer, set_seed

def optim_loop(v_n, target, decoder, loss_fn, optimizer, n_steps:int=1000, verbose:bool=False, restrict_vocab:int=50400):
    if target.shape[-1] != restrict_vocab:
        inds = torch.topk(target, restrict_vocab).indices[0]
        Z = torch.zeros(target.size()).cuda()
        Z[:,inds] = target[:,inds]
    else:
        Z = target
            
    for i in range(n_steps):
        loss = loss_fn(decoder(v_n),Z)
        loss.backward()
        if verbose:
            print(f"Loss:{loss.item()}, iter:{i}")
        optimizer.step()
        optimizer.zero_grad()
    return v_n

def vocab_reconstruction(datasets, n_steps:int=1000, lr:float=0.5, n_seeds:int=5, n_trials:int=100, n_shots:int=10, restrict_vocab_list=[100,50400], return_vecs:bool=False):
    """
    Computes and evaluates a function vector reconstruction which matches its output vocabulary distribution.
    
    Parameters:
    n_steps: number of optimization steps
    lr: adam learning rate
    n_seeds: number of seeds to run
    n_trials: number of prompts to compute task-conditioned mean head activations over
    n_shots: number of shots for task-conditioned mean prompts
    restrict_vocab_list: list of ints determining how many vocab words to match. Defaults to 100 & full-vocab (which is 50400 for GPT-J)
    return_vecs: whether to return the function vectors and their corresponding vocab-optimized reconstruction vectors

    Returns:
    orig_results: FV results
    zs_results: 
    kl_divs: kl divergences between the distribution of the FV and its reconstruction
    fvs: (optional) the function vectors used
    vns: (optional) the vocab-optimized reconstruction vectors
    """

    seeds = {k:[] for k in datasets}
    orig_results = {k:[] for k in datasets}
    fvs = {k:[] for k in datasets}
    vns = {k:{j:[] for j in range(len(restrict_vocab_list))} for k in datasets}
    zs_results = {k:{j:[] for j in range(len(restrict_vocab_list))} for k in datasets}
    kl_divs = {k:{j:[] for j in range(len(restrict_vocab_list))} for k in datasets}


    for dataset_name in datasets:
        print(f"Dataset: {dataset_name}")
        
        for i in range(n_seeds):
            seed = np.random.randint(100000)
            print(f"seed:{seed}")
            seeds[dataset_name].append(seed)
            set_seed(seed)
            
            # Disable gradients when extracting activations & computing FV 
            torch.set_grad_enabled(False)

            dataset = load_dataset(dataset_name, seed=seed, root_data_dir=root_data_dir)

            fs_results_validation = n_shot_eval_no_intervention(dataset=dataset, n_shots=n_shots, model=model, model_config=model_config, tokenizer=tokenizer, compute_ppl=False, test_split='valid')
            filter_set_validation = np.where(np.array(fs_results_validation['clean_rank_list']) == 0)[0]
            
            mean_activations = get_mean_head_activations(dataset, model=model, model_config=model_config, tokenizer=tokenizer, n_icl_examples=n_shots,
                                                        N_TRIALS=n_trials, filter_set=filter_set_validation)

            fv, _ = compute_universal_function_vector(mean_activations, model, model_config)
            
            fs_results = n_shot_eval_no_intervention(dataset=dataset, n_shots=n_shots, model=model, model_config=model_config, tokenizer=tokenizer, compute_ppl=False, test_split='test')
            filter_set = np.where(np.array(fs_results['clean_rank_list']) == 0)[0]
            
            fv_results = n_shot_eval(dataset=dataset, fv_vector=fv, edit_layer=9, n_shots=0,
                                    model=model, model_config=model_config, tokenizer=tokenizer, filter_set=filter_set)
            
            orig_results[dataset_name].append(fv_results)
            fvs[dataset_name].append(fv)
            
            for j, vocab_size in enumerate(restrict_vocab_list):
                # Enable Gradients for Optimization
                torch.set_grad_enabled(True)
                v_n = torch.randn(fv.size()).cuda()
                v_n.requires_grad=True

                # Optim setup
                loss_fn = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam([v_n], lr=lr)
                decoder = torch.nn.Sequential(model.transformer.ln_f, model.lm_head).to(model.device)

                decoder.requires_grad=True
                for p in decoder.parameters():
                    p.requires_grad = True

                target = torch.nn.functional.softmax(decoder(fv), dim=-1).detach()

                computed_vn = optim_loop(v_n, target, decoder, loss_fn, optimizer, verbose=False, n_steps=n_steps, restrict_vocab=vocab_size)
                
                scaled_vn = computed_vn / torch.linalg.norm(computed_vn) * torch.linalg.norm(fv)

                zs_reconstruction_results = n_shot_eval(dataset=dataset, fv_vector=scaled_vn, edit_layer=9, n_shots=0,
                                        model=model, model_config=model_config, tokenizer=tokenizer, filter_set=filter_set)
                
                zs_results[dataset_name][j].append(zs_reconstruction_results)
                vns[dataset_name][j].append(scaled_vn.detach())
                
                # Compute kl divergence between two distributions
                if vocab_size != 50400:
                    tp = torch.softmax(decoder(fvs[dataset_name][i]), dim=-1)
                    inds = torch.topk(tp, vocab_size).indices[0]
                    vn_ps = torch.softmax(decoder(vns[dataset_name][j][i]), dim=-1)[:,inds]

                    log_probs = torch.log(vn_ps / vn_ps.sum())
                    target_probs = tp[:,inds] / tp[:,inds].sum()
                else:
                    log_probs = torch.log(torch.softmax(decoder(vns[dataset_name][j][i]), dim=-1))
                    target_probs = torch.softmax(decoder(fvs[dataset_name][i]), dim=-1)

                kl_divs[dataset_name][j].append(torch.nn.functional.kl_div(log_probs, target_probs, reduction='batchmean').item())

    if return_vecs:
        return orig_results, zs_results, kl_divs, fvs, vns
    else:   
        return orig_results, zs_results, kl_divs


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', help='Name of model to be loaded', type=str, required=False, default='EleutherAI/gpt-j-6b')
    parser.add_argument('--root_data_dir', help='Root directory of data files', type=str, required=False, default='../dataset_files')
    parser.add_argument('--save_path_root', help='File path to save mean activations to', type=str, required=False, default='../results')
    parser.add_argument('--n_seeds', help='Number of seeds', type=int, required=False, default=5)
    parser.add_argument('--n_trials', help='Number of trials to use for computing task-conditioned mean head activations', type=int, required=False, default=100)
    parser.add_argument('--n_shots', help='Number of shots to use for prompts when computing task-conditioned mean head activations', type=int, required=False, default=10)
    parser.add_argument('--lr', help="Learning Rate for Adam Optimizer", type=int, required=False, default=0.5)
    parser.add_argument('--n_steps', help="Learning Rate for Adam Optimizer", type=int, required=False, default=1000)
        
    args = parser.parse_args()
    
    # Gather inputs
    model_name = args.model_name
    root_data_dir = args.root_data_dir
    save_path_root = args.save_path_root
    n_seeds = args.n_seeds
    n_trials = args.n_trials
    n_shots = args.n_shots
    lr = args.lr
    n_steps = args.n_steps
    

    # Load Model & Tokenizer
    torch.set_grad_enabled(False)
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(model_name)

    datasets = ['antonym', 'english-french', 'capitalize', 'present-past', 'singular-plural', 'country-capital']
    args.datasets = datasets

    # Test Loop:
    orig_results, zs_results, kl_divs = vocab_reconstruction(datasets, n_steps=n_steps, lr=lr, n_seeds=n_seeds, n_trials=n_trials, n_shots=n_shots, restrict_vocab_list=[100,50400])
  

    # Extract Summary Results:
    os.makedirs(os.path.join(save_path_root), exist_ok=True)
    with open(os.path.join(save_path_root, 'reconstruction_results.txt'), 'w') as out_file:

        for dataset_name in datasets:
            print(f"{dataset_name.title()}:", file=out_file)
            fv_acc = [orig_results[dataset_name][i]['intervention_topk'][0][1] for i in range(n_seeds)]
            v100_acc = [zs_results[dataset_name][0][i]['intervention_topk'][0][1] for i in range(n_seeds)]
            kl100_val = kl_divs[dataset_name][0]
            
            vfull_acc = [zs_results[dataset_name][1][i]['intervention_topk'][0][1] for i in range(n_seeds)]
            klfull_val = kl_divs[dataset_name][1]
            
            print("fv results:", np.mean(fv_acc).round(3)*100, '% +/-', np.std(fv_acc).round(3)*100, file=out_file)
            print("v_100 results:", np.mean(v100_acc).round(3)*100, '% +/-', np.std(v100_acc).round(3)*100, file=out_file)
            print("KL100:", np.mean(kl100_val).round(5), '+/-', np.std(kl100_val).round(5), file=out_file)
            print("v_full results:", np.mean(vfull_acc).round(3)*100, '% +/-', np.std(vfull_acc).round(3)*100, file=out_file)
            print("KLFull:", np.mean(klfull_val).round(5), '+/-', np.std(klfull_val).round(5), '\n', file=out_file)
    
    with open(os.path.join(save_path_root, 'reconstruction_args.txt'), 'w') as arg_file:
        print(args.__dict__, file=arg_file)