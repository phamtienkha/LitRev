import os
from dotenv import load_dotenv

from aslite.db import get_papers_db
from aslite.embedding import embed_text
from .utils import generate_response_api, load_model_tokenizer, preprocess_text, get_bigrams
import string
import numpy as np
from nltk.corpus import wordnet as wn
import inflect
from pinecone import Pinecone

def search_rank(q: str = '', top_k=10): 
    def _all_apear_in(s, q, excludes=[]):
        return all([f" {qp} " in f" {s} " for qp in q if qp not in excludes])
                                                                                                                                                                                                   
    if not q:
        return [], [] # no query? no results
    
    # preprocess the query
    q = shorten(q)
    q = preprocess_text(q)
    q_embed = embed_text([q])[0]

    pc = Pinecone(api_key=os.getenv("PINECONE_API"))
    index = pc.Index("litrev")
    namespace = "summary"
    queried = index.query(
        namespace=namespace,
        vector=q_embed,
        top_k=top_k,
    )
    
    pids = [x["id"] for x in queried]
    scores = [x["score"] for x in queried]

    return pids, scores

def search_deeplearning(q: str= ''):
    pass

def shorten(text):
    if len(text.split()) <= 6:
        return text
    prompt = f"""summarize into 4-6 words, as a research topic: {text}
    Output the summarization only.
    """
    model_path = 'claude-sonnet'
    model, _ = load_model_tokenizer(model_path=model_path)

    short_text = generate_response_api(prompt=prompt, model=model, model_path=model_path)
    print('Shortened query:', short_text)
    return short_text

