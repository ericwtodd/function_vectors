#!/bin/bash
datasets=('antonym' 'capitalize' 'country-capital' 'english-french' 'present-past' 'singular-plural')
cd ../

for d_name in "${datasets[@]}"
do
    echo "Running Script for: ${d_name}"
    python test_numheads.py --dataset_name="${d_name}" --model_name='EleutherAI/gpt-j-6b'
done