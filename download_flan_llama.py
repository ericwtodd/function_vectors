device = 0 # Define your GPU device here
llama_path = '/data/public_models/llama/llama_hf_weights/llama-7b' # Define your original llama-7b load path here (huggingface checkpoint)
save_path = './flan-llama-7b' # Define your save path here


from transformers import LlamaForCausalLM, LlamaTokenizer
from collections import OrderedDict

model_llama = LlamaForCausalLM.from_pretrained(llama_path)
tokenizer = LlamaTokenizer.from_pretrained(llama_path)

model_flan_llama = LlamaForCausalLM.from_pretrained("NTU-NLP-sg/flan-llama-7b-10m-delta")
flan_tokenizer = LlamaTokenizer.from_pretrained("NTU-NLP-sg/flan-llama-7b-10m-delta")

model_llama.resize_token_embeddings(len(flan_tokenizer))

model_state_dict = []
for key in model_flan_llama.state_dict().keys():
    model_state_dict.append((key, model_flan_llama.state_dict()[key]+model_llama.state_dict()[key]))
model_state_dict = OrderedDict(model_state_dict)
model_flan_llama.load_state_dict(model_state_dict)

model_flan_llama = model_flan_llama.to(device)
model_flan_llama.eval()

def generate(prompt, model, device):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    gen_output = model.generate(input_ids.to(device), max_new_tokens=512, early_stopping=True)[0]
    answer_cot = tokenizer.decode(gen_output, skip_special_tokens=True)
    return answer_cot

prompt = "Can Geoffrey Hinton have a conversation with George Washington? Give the rationale before answering."
print(generate(prompt, model_flan_llama, device))

flan_tokenizer.save_pretrained(save_path)
model_flan_llama.save_pretrained(save_path)

