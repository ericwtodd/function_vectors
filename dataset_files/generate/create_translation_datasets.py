from transformers import AutoTokenizer
from typing import *
from typing import TextIO

import json
import os

def verify_word_length(word: str, tokenizer: object) -> bool:
    """
    Verifies whether a word can be tokenized into a single word or not

    Parameters:
        word: the word that we're checking
        tokenizer: the tokenizer we use to tokenize the word

    Return: a boolean denoting whether the word is within the required 1 or not
    """

    return len(tokenizer(word)['input_ids']) == 1

if __name__ == "__main__":

    d_names = {'en-de':"english-german", 'en-es':"english-spanish",'en-fr':"english-french"}
    path_exists = [os.path.exists(f'./translation/{lang_id}.0-5000.txt') for lang_id in d_names.keys()] + [os.path.exists(f'./translation/{lang_id}.5000-6500.txt') for lang_id in d_names.keys()]
    
    assert all(path_exists), "Original data missing! Please download corresponding 'train' and 'test' files from https://github.com/facebookresearch/MUSE#ground-truth-bilingual-dictionaries in order to re-generate translation_datasets."

    model_name = r"EleutherAI/gpt-j-6b"

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    out_dir = '../abstractive'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    for lang_id in d_names.keys():
        valid = []
        for d_base in [f'./translation/{lang_id}.0-5000.txt', f'./translation/{lang_id}.5000-6500.txt']:
            with open(d_base, 'r', encoding="utf-8") as f:
                lines = f.read()

            word_pairs = list(set([tuple(x.split()) for x in lines.splitlines()]))
            word_pairs = [{'input':w1, 'output':w2} for (w1,w2) in word_pairs]

            for i, x in enumerate(word_pairs):
                if (x['input'] != x['output']): # Filter pairs that are exact copies
                    valid.append(word_pairs[i])

        json.dump(valid, open(os.path.join(out_dir, f'{d_names[lang_id]}.json'), 'w'))


