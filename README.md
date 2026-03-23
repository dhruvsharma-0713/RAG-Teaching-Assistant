# RAG-Teaching-Assistant
A RAG-based pipeline that transforms video lectures into an interactive AI assistant using OpenAI Whisper, Vector Embeddings, and LLMs for semantic retrieval.

This project is a Retrieval-Augmented Generation (RAG) Teaching Assistant designed to help students interact with video lecture content. It automates the entire pipeline: converting video to audio, transcribing speech to text via Whisper, generating Vector Embeddings for semantic search, and using an LLM to provide context-aware answers to student queries.

# How to use this RAG AI Teaching assistant on your own data
## Step 1 - Collect your videos
Move all your video files to the videos folder.

## Step 2 - Convert to mp3
Convert all the video files to mp3 by running video_to_mp3.py

## Step 3 - Convert mp3 to json
Convert all the mp3 files to json by running mp3_to_json.py

## Step 4 - Convert the json files to vectors
Use the file preprocess_json to convert the json files to a dataframe with Embeddings and save it as a joblib pickle

## Step 5 - Prompt generation and feeding to LLM
Read the joblib file and load it into the memory. Then create a relevent prompt as per the user query and feed it to the LLM.
