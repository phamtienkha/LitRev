from aslite.db import get_papers_db
from .utils import generate_response_api, load_model_tokenizer
import string

def search_rank(q: str = ''):
    def _all_appear(q, s):
        return all(f" {qp} " in f" {s} " for qp in q)                                                                                                                                                                                                  

    if not q:
        return [], [] # no query? no results
    
    # get related words and important words
    related_words = get_related_words(q)
    print(related_words)
    important_words = []
    for w in related_words:
        w = remove_punctuation(w)
        important_words += get_important_words(w)
    important_words = list(set(important_words))
    print(important_words)

    # search in the database
    pdb = get_papers_db()
    match = lambda s: sum(min(3, f" {s} ".lower().count(f" {qp} ")) for qp in important_words)
    matchu = lambda s: sum(int(f" {s} ".lower().count(f" {qp} ") > 0) for qp in important_words)
    pairs = []
    for pid, p in pdb.items():
        score = 0.0
        title = remove_punctuation(p['title'].lower())
        summary = remove_punctuation(p['summary'].lower())

        score += 10.0 * matchu(' '.join([a['name'].lower() for a in p['authors']]))
        score += 20.0 * matchu(title)
        score += 1.0 * match(summary)

        # if the score is positive and all important words appear in the summary
        if score > 0:
            pairs.append((score, pid))

    pairs.sort(reverse=True)
    pids = [p[1] for p in pairs]
    scores = [p[0] for p in pairs]
    return pids, scores

def remove_punctuation(text):
    """Removes punctuation from a string using list comprehension."""
    text_nopunc = ''
    for char in text:
        if char not in string.punctuation:
            text_nopunc += char 
        else:
            text_nopunc += ' '

    # Remove all double spaces
    text_nopunc = ' '.join(text_nopunc.split())
    return text_nopunc

def get_important_words(text):
    prompt = f"""List important words in this string, so that a search engine must not ignore those words: "{text.lower()}". 
    Output should be in one line, including only the words separated by comma. 
    You may correct any word if you think it is a typo.
    """
    model_path = 'claude-sonnet'
    model, _ = load_model_tokenizer(model_path=model_path)

    response = generate_response_api(prompt=prompt, model=model, model_path=model_path)
    return [word.strip() for word in response.split(',')]

def get_related_words(text):
    prompt = f"""give me a list of 4 most related keywords to "{text.lower()}", including abbreviations, plural/singualar form, noun/verb equivalence, etc. 
    Output the keywords only, in one line, separated by comma. 
    You may correct any word if you think it is a typo.
    """
    model_path = 'claude-sonnet'
    model, _ = load_model_tokenizer(model_path=model_path)

    response = generate_response_api(prompt=prompt, model=model, model_path=model_path)
    return [word.strip() for word in response.split(',')]
