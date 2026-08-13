from datasets import load_dataset
from transformers import GPT2TokenizerFast
from pathlib import Path
import torch
import numpy as np
from transformers import GPT2Model

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
# 1. Load and process the dataset
wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train[:20000]")

# Explore the dataset
print("Number of docs: ", len(wiki)) # 20000
# print("First doc: ", wiki[0])
# print("Tokens: ", tokenizer.encode(wiki[0]["text"]))
# print("Token count: ", len(tokenizer.encode(wiki[0]["text"])))

# Loop over each doc, tokenize it, check if there are geq 512 tokens, if it does, truncate it to 512 tokens
# Add it to a list
# Save the list to a file (n_docs, 512)
def process(batch, idx):
    enc = tokenizer(batch["text"], truncation=True, max_length=512)
    return {"ids": enc["input_ids"], "doc_id": idx}
    
if Path("tokenized.jsonl").exists():
    wiki = load_dataset("json", data_files="tokenized.jsonl", split="train")
else:
    wiki = wiki.map(process, batched=True, with_indices=True, batch_size=1000, num_proc=8, remove_columns=wiki.column_names)
    wiki = wiki.filter(lambda x: len(x["ids"]) == 512, num_proc=8)
    wiki.to_json("tokenized.jsonl")

print("first row: ", wiki[0])

# 2. Extract the activations at layer N_LAYERS * 2/3
model = GPT2Model.from_pretrained("gpt2").eval()
N_LAYERS = model.config.n_layer  # 12 for gpt2
LAYER = N_LAYERS * 2 // 3

# For each batch of docs, forward pass the token ids through the model, 
# and extract the activations at layer N_LAYERS * 2/3 (residual stream) for each token
# Save the list to a file (n_docs, 512, d)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

acts = []
with torch.no_grad():
    for i in range(0, len(wiki), 32):
        print("Processing: ", i, "of", len(wiki))
        batch = torch.tensor(wiki[i:i+32]["ids"], device=device)
        out = model(batch, output_hidden_states=True)
        acts.append(out.hidden_states[LAYER].cpu().numpy())

acts = np.concatenate(acts, axis=0)  # (n_docs, 512, d)
print("Activations shape: ", acts.shape)
np.save("activations.npy", acts)