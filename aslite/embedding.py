"""
Database support functions.
The idea is that none of the individual scripts deal directly with the file system.
Any of the file system I/O and the associated settings are in this single file.
"""

import os
import pickle
from contextlib import contextmanager
import sqlite3, zlib, pickle, tempfile
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import google.generativeai as genai
import google.ai.generativelanguage as glm

def embed_text(q_list, method="local"):
    embeddings = []
    if method == "gemini":
        for q in tqdm(q_list):
            model = 'models/embedding-001'
            title = "The next generation of AI for developers and Google Workspace"
            embedding = genai.embed_content(model=model,
                                            content=q,
                                            task_type="retrieval_document",
                                            title=title)
        embeddings.append(embedding)
    elif method == "openai":
        client = OpenAI()
        model="text-embedding-3-small"
        embeddings = client.embeddings.create(input = q_list, model=model).data
    else:
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("Encodding")
        embeddings = model.encode(q_list, show_progress_bar=True)

    return embeddings
