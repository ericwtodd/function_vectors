"""
Each modeling file in this library is a mapping between
abstract naming of intervention anchor points and actual
model module defined in the huggingface library.

We also want to let the intervention library know how to
config the dimensions of intervention based on model config
defined in the huggingface library.
"""


from models.constants import CONST_INPUT_HOOK, CONST_OUTPUT_HOOK, CONST_QKV_INDICES


"""gpt2 base model"""
gpt2_type_to_module_mapping = {
    'block_input': ("h[%s]", CONST_INPUT_HOOK), 
    'block_output': ("h[%s]", CONST_OUTPUT_HOOK), 
    'mlp_activation': ("h[%s].mlp.act", CONST_OUTPUT_HOOK), 
    'mlp_output': ("h[%s].mlp", CONST_OUTPUT_HOOK), 
    'mlp_input': ("h[%s].mlp", CONST_INPUT_HOOK), 
    'attention_value_output': ("h[%s].attn.c_proj", CONST_INPUT_HOOK),
    'head_attention_value_output': ("h[%s].attn.c_proj", CONST_INPUT_HOOK),
    'attention_output': ("h[%s].attn", CONST_OUTPUT_HOOK),
    'attention_input': ("h[%s].attn", CONST_INPUT_HOOK),
    'query_output': ("h[%s].attn.c_attn", CONST_OUTPUT_HOOK),
    'key_output': ("h[%s].attn.c_attn", CONST_OUTPUT_HOOK),
    'value_output': ("h[%s].attn.c_attn", CONST_OUTPUT_HOOK),
    'head_query_output': ("h[%s].attn.c_attn", CONST_OUTPUT_HOOK),
    'head_key_output': ("h[%s].attn.c_attn", CONST_OUTPUT_HOOK),
    'head_value_output': ("h[%s].attn.c_attn", CONST_OUTPUT_HOOK),
}


gpt2_type_to_dimension_mapping = {
    'block_input': ("n_embd", ), 
    'block_output': ("n_embd", ), 
    'mlp_activation': ("n_inner", "n_embd*4", ), 
    'mlp_output': ("n_embd", ), 
    'mlp_input': ("n_embd", ), 
    'attention_value_output': ("n_embd", ),
    'head_attention_value_output': ("n_embd/n_head", ),
    'attention_output': ("n_embd", ),
    'attention_input': ("n_embd", ),
    'query_output': ("n_embd", ),
    'key_output': ("n_embd", ),
    'value_output': ("n_embd", ),
    'head_query_output': ("n_embd/n_head", ),
    'head_key_output': ("n_embd/n_head", ),
    'head_value_output': ("n_embd/n_head", ),
}


"""gpt2 model with LM head"""
gpt2_lm_type_to_module_mapping = {}
for k, v in gpt2_type_to_module_mapping.items():
    gpt2_lm_type_to_module_mapping[k] = (f"transformer.{v[0]}", v[1])


gpt2_lm_type_to_dimension_mapping = gpt2_type_to_dimension_mapping


def create_gpt2(name="gpt2", cache_dir="../../.huggingface_cache"):
    """Creates a GPT2 model, config, and tokenizer from the given name and revision"""
    from transformers import GPT2Model, GPT2Tokenizer, GPT2Config
    
    config = GPT2Config.from_pretrained(name)
    tokenizer = GPT2Tokenizer.from_pretrained(name)
    gpt = GPT2Model.from_pretrained(name, config=config, cache_dir=cache_dir)
    print("loaded model")
    return config, tokenizer, gpt

def create_gpt2_lm(name="gpt2", config=None, cache_dir="../../.huggingface_cache"):
    """Creates a GPT2 LM, config, and tokenizer from the given name and revision"""
    from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if config is None:
        config = GPT2Config.from_pretrained(name)
        gpt = GPT2LMHeadModel.from_pretrained(name, config=config, cache_dir=cache_dir)
    else:
        gpt = GPT2LMHeadModel(config=config)
    print("loaded model")
    return config, tokenizer, gpt