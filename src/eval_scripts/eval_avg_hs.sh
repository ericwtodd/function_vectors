#!/bin/bash
datasets=('antonym' 'capitalize' 'country-capital' 'english-french' 'present-past' 'singular-plural')
cd ../

for d_name in "${datasets[@]}"
do
    echo "Running Script for: ${d_name}"
    python compute_avg_hidden_state.py --dataset_name="${d_name}" --save_path_root="results/gptj_avg_hs" --model_name='EleutherAI/gpt-j-6b'
done
