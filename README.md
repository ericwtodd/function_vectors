# Function Vectors in Large Language Models
### [Project Website](https://functions.baulab.info)

This repository contains data and code for the paper: [Function Vectors in Large Language Models](https://arxiv.org/abs/2310.15213).

<p align="left">
<img src="https://functions.baulab.info/images/Paper/fv-new-overview.png" style="width:100%;"/>
</p> 

## Setup

We recommend using conda as a package manager. 
The environment used for this project can be found in the `fv_environment.yml` file.
To install, you can run: 
```
conda env create -f fv_environment.yml
conda activate fv
```

## Demo Notebook
Checkout `notebooks/fv_demo.ipynb` for a jupyter notebook with a demo of how to create a function vector and use it in different contexts.

## Data
The datasets used in our project can be found in the `dataset_files` folder.

## Code
Our main evaluation scripts are contained in the `src` directory with sample script wrappers in `src/eval_scripts`.

Other main code is split into various util files:
- `eval_utils.py` contains code for evaluating function vectors in a variety of contexts
- `extract_utils.py`  contains functions for extracting function vectors and other relevant model activations.
- `intervention_utils.py` contains main functionality for intervening with function vectors during inference
- `model_utils.py` contains helpful functions for loading models & tokenizers from huggingface
- `prompt_utils.py` contains data loading and prompt creation functionality

## Citing our work
The preprint can be cited as follows

```bibtex
@article{todd2023function,
    title={Function Vectors in Large Language Models}, 
    author={Eric Todd and Millicent L. Li and Arnab Sen Sharma and Aaron Mueller and Byron C. Wallace and David Bau},
    year={2023},
    eprint={2310.15213},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
