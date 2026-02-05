import os
import pandas as pd
import numpy as np
import pickle
import json
import gc
import time
from scipy.spatial import distance
from transformers import AutoTokenizer, AutoModel
import torch


tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")
model.load_state_dict(torch.load("./CodeXGLUE/Code-Code/Defect-detection/code/saved_models/checkpoint-best-acc/model.bin"), strict=False)

import json
import pandas as pd

# List of JSONL files to merge
jsonl_files = ["train2.jsonl", "valid2.jsonl", "sven.jsonl"]

# Create an empty list to store the JSON objects
data = []

# Read and merge JSONL files
for file in jsonl_files:
    with open(f"raw/{file}", "r") as f:
        for line in f:
            data.append(json.loads(line))

# Create DataFrame from the merged data
dfNew = pd.DataFrame(data)

dfTest = pd.DataFrame(dfNew, columns=['target', 'func'])

from tqdm import tqdm
sentences1 = dfNew['func']
all_data = None
for i, sentence in tqdm(enumerate(sentences1)):
    inputs = tokenizer(sentence, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    if i == 0:
        all_data = outputs.pooler_output.cpu()
    else:
        all_data = torch.cat([all_data, outputs.pooler_output.cpu()], dim=0)

torch.save(all_data, "embedding_codebert.pt")
