#!/bin/bash
datasets=('antonym')
# datasets=('antonym' 'capitalize' 'country-capital' 'english-french' 'present-past' 'singular-plural')
cd ../

for d_name in "${datasets[@]}"
do
    echo "Running Script for: ${d_name}"
    python evaluate_function_vector.py --dataset_name="${d_name}" --save_path_root="results/gptj" --model_name='EleutherAI/gpt-j-6b'
done