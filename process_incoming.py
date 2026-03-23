import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import requests

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/v1/embeddings", json={"model": "bge-m3",
                                                                    "input": text_list})
    resp = r.json()

    embeddings = [item['embedding'] for item in resp.get('data', [])]
    if len(embeddings) != len(text_list):
        raise ValueError(f"Expected {len(text_list)} embeddings, got {len(embeddings)}. Response: {resp}")
    return embeddings

def inference(prompt, model):
    r = requests.post("http://localhost:11434/api/generate", json={
        # "model": "deepseek-r1",
        # "model": "llama3.2",
        "model": "llama3",
        "prompt": prompt,
        "stream": False})
    response = r.json()
    print(response)
    return response

df = joblib.load('embeddings.joblib')


incoming_query = input("Ask a Question: ")
question_embedding = create_embedding([incoming_query])[0]

# Find similiarities of question embedding with other embeddings
# print(np.vstack(df['embedding'].values))
# print(np.vstack(df['embedding'].shape))

similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
print(similarities)
top_results = 30
max_indx = similarities.argsort()[::-1][0:top_results]
# print(max_indx)
new_df = df.loc[max_indx]
# print(new_df[['number', 'title', 'start', 'end', 'text']])

prompt = f''' Harry is teaching web development using sigma web development course.here are video chunks with their respective video number, title, start and end timestamps, and the text content of the chunk:

{new_df[['number', 'title', 'start', 'end', 'text']].to_json(orient='records')}
------------------------------------------------
"{incoming_query}"
user asked this question related to the video chunks, you have to answer where and how much content is taught where (in which video and at what timestamp) and guide the user to go to that video and timestamp to learn the answer. You have to answer in a concise way, you should not give the whole content of the chunk, you should only give the information about which video and at what timestamp the user can find the answer. You should not give any other information except that.
if user ask unrelated question, tell him that you are not sure about the answer and suggest him to ask questions related to the course.
'''
with open ("prompt.txt", "w") as f:
    f.write(prompt)

response = inference(prompt, model="llama3")["response"]
print(response)


with open ("response.txt", "w") as f:
    f.write(response)
