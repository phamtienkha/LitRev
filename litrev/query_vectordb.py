from aslite.db import get_papers_db
from aslite.embedding import get_papers_db_embedding
from .utils import generate_response_api, load_model_tokenizer, preprocess_text, get_bigrams
import string
import numpy as np
from nltk.corpus import wordnet as wn
import inflect

def search_rank(q: str = ''): 
    def _all_apear_in(s, q, excludes=[]):
        return all([f" {qp} " in f" {s} " for qp in q if qp not in excludes])
                                                                                                                                                                                                   
    if not q:
        return [], [] # no query? no results
    
    # preprocess the query
    q = shorten(q)
    q = preprocess_text(q)
    

    # search in the database
    pdb = get_papers_db()
    pdb_embeddigns = get_papers_db_embedding(pdb)
    raise Exception


    match = lambda s: sum(f" {s} ".lower().count(f" {qp} ") * 1/np.log(imp_w_freqs[qp]+1) for qp in q_words)
    match2 = lambda s: sum(f" {s} ".lower().count(f" {qp} ") * 1/5 for qp in q_bigrams)
    pairs = []
    for pid, p in pdb.items():
        score = 0.0
        title = preprocess_text(p['title'].lower())
        summary = preprocess_text(p['summary'].lower())

        score += 10.0 * match(' '.join([a['name'].lower() for a in p['authors']]))

        # match the query words
        score += 20.0 * match(title)
        score += 1.0 * match(summary)

        # match the bigrams
        score += 20 * match2(title)
        score += 2.0 * match2(summary)

        # if the score is positive and all important words appear in the summary
        if score > 0 and _all_apear_in(' '.join([title, summary]), q_words, excludes=excludes):
            pairs.append((score, pid))

    pairs.sort(reverse=True)
    pids = [p[1] for p in pairs]
    scores = [p[0] for p in pairs]
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

