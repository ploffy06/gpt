import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
from model.v1 import GPTv1
from model.v2 import GPTv2

if len(sys.argv) != 3:
    print("Incorrect usage")
    exit(0)

version = sys.argv[1]
prompt = sys.argv[2]

if version not in ["v1", "v2"]:
    print("Invalid version specified")
    exit(0)


with open('sonnets.txt', 'r', encoding='utf-8') as file:
    text = file.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {s:i for i, s in enumerate(chars)}
itos = {i:s for i, s in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[c] for c in l])

n_embed = 192
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if version == "v1":
    model = GPTv1(vocab_size, n_embed)
    model.load_state_dict(torch.load("model/v1"))
elif version == "v2":
    model = GPTv2(vocab_size, n_embed)
    model.load_state_dict(torch.load("model/v2"))

model.eval()

if prompt == "":
    data_input = torch.zeros((1, 1), dtype=torch.long, device=device)
else:
    data_input = torch.tensor([encode(prompt)], dtype=torch.long)

print(decode(model.generate(data_input, max_new_tokens=500)[0].tolist()))