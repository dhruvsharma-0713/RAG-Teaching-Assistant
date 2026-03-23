import requests
import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/v1/embeddings", json={"model": "bge-m3",
                                                                    "input": text_list})
    resp = r.json()

    embeddings = [item['embedding'] for item in resp.get('data', [])]
    if len(embeddings) != len(text_list):
        raise ValueError(f"Expected {len(text_list)} embeddings, got {len(embeddings)}. Response: {resp}")
    return embeddings

jsons = os.listdir("jsons")
my_dicts = []
chunk_id = 0

for json_file in jsons:
    with open(f"jsons/{json_file}") as f:
        content = json.load(f)
    print(f"Processing file: {json_file} with {len(content['chunks'])} chunks")
    embeddings = create_embedding([c['text'] for c in content['chunks']])

    for i, chunk in enumerate(content['chunks']):
        chunk['chunk_id'] = chunk_id
        chunk['embedding'] = embeddings[i]
        chunk_id += 1
        my_dicts.append(chunk)

# print(f"Total chunks: {len(my_dicts)}")
# print(my_dicts)

df = pd.DataFrame.from_records(my_dicts)
# save this dataframe
joblib.dump(df, 'embeddings.joblib')