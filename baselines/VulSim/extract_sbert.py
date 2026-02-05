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

from sentence_transformers import SentenceTransformer,util
model = SentenceTransformer('all-MiniLM-L6-v2')

# Two lists of sentences
# sentences1 = ['The cat sits outside',
#               'The cat sits inside',
#               'The new movie is so great']

sentences1 = dfNew['func']
#sentences1 = Desc
#print(sentences1)

# sentences2 = ['The cat sits outside',
#               'The cat sits inside',
#               'The new movie is so great']

#Compute embedding for both lists

embeddings1 = model.encode(sentences1, convert_to_tensor=True)

# embeddings2 = model.encode(sentences2, convert_to_tensor=True)

#Compute cosine-similarities
cosine_scores = util.cos_sim(embeddings1, embeddings1)
npArr = cosine_scores.cpu().detach().numpy()
# print(npArr)

#Output the pairs with their score
# for i in range(len(sentences1-1)):
#     print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences1[i+], cosine_scores[i][j]))

f= open('outputEmbedding.txt','w')
for i in range(len(npArr)):
  for j in range(len(npArr)):
    f.write(str(npArr[i][j])+" ")
  f.write("\n")
f.close()
