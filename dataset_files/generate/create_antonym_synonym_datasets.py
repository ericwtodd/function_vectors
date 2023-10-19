from transformers import AutoTokenizer
from typing import *
from typing import TextIO

import json
import os
import random

def verify_word_length(word: str, tokenizer: object) -> bool:
    """
    Verifies whether a word can be tokenized into a single token

    Parameters:
        word: the word that we're checking
        tokenizer: the tokenizer we use to tokenize the word

    Return: a boolean denoting whether the word is within the required 1 or not
    """

    return len(tokenizer(word)['input_ids']) == 1

def parse_file(
    f_in: TextIO,
    ant_list: List[Dict],
    syn_list: List[Dict],
    seen: Set,
    tokenizer: object
):
    """
    Parses the input file into synonym and antonym categories

    Parameters:
        f_in: the input file from where the data will be taken from
        ant_list: the list of antonyms
        syn_list: the list of synonyms
        seen: the seen set of tuples to check against for duplicates
        tokenizer: the tokenizer we use to tokenize the word
    """
    for line in f_in:
        word1, word2, t = line.split()
        t = int(t)

        word1_bool = verify_word_length(" " + word1, tokenizer)
        word2_bool = verify_word_length(" " + word2, tokenizer)

        if word1_bool and word2_bool:
            d = {"input": word1, "output": word2}
            words = (word1, word2)
            if words not in seen:
                seen.add(words)
            else:
                continue
            # Synonym
            if t == 0:
                syn_list.append(d)
            # Antonym
            else:
                ant_list.append(d)
        else:
            continue


if __name__ == "__main__":
    # Seed for dataset generation
    random.seed(42)
    model_name = r"EleutherAI/gpt-j-6B"

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    assert os.path.exists('./AntSynNET/dataset'), "Original dataset missing! Please first clone https://github.com/nguyenkh/AntSynNET into this folder in order to re-generate antonym and synonym datasets."

    out_dir = "../abstractive"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    input_data_dir = "./AntSynNET/dataset"
    splits = ["train", "val", "test"]
    types = ["adjective", "noun", "verb"]

    ant_list = []
    syn_list = []
    filename_ant = "antonym.json"
    filename_syn = "synonym.json"
    seen = set()

    ant_path = os.path.join(out_dir, filename_ant)
    syn_path = os.path.join(out_dir, filename_syn)
    f_ant = open(ant_path, "w")
    f_syn = open(syn_path, "w")
    for s in splits:
        for t in types:
            path = t + "-pairs." + s
            full_path = os.path.join(input_data_dir, path)
            input_file = open(full_path, "r")
            parse_file(input_file, ant_list, syn_list, seen, tokenizer)

    json.dump(ant_list, f_ant)
    json.dump(syn_list, f_syn)
    f_ant.close()
    f_syn.close()
