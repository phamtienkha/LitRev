import numpy as np
from aslite.db import get_papers_db
from litrev.utils import preprocess_text

def compute_freq():
    pdb = get_papers_db()
    abss = [p['summary'] for p in pdb.values()]
    abss = [preprocess_text(abs) for abs in abss]

    all_words = []
    for abs in abss:
        all_words.extend(abs.split())

    word_counts = {}
    for word in all_words:
        word_counts[word] = word_counts.get(word, 0) + 1

    word_counts = dict(sorted(word_counts.items(), key=lambda item: item[1], reverse=True))
    
    # save the dict
    np.save('./data/wordfreqs.npy', word_counts)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    compute_freq()
