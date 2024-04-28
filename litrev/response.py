from .utils import generate_response_api
from .const import API_MODELS
import pandas as pd
from sentence_transformers import SentenceTransformer
import hdbscan
import random

def summarize_probs(response, model, model_path):
    probs = get_problem(response)
    # print(probs)
    # probs = rearrange_probs(probs)
    random.seed(2024)
    random.shuffle(probs)
    
    prompt = 'Here are the problems mentioned in the papers:\n' + '\n'.join(probs)
    prompt += '\n\nNow, based on the above information, answer this question: What important research problems are addressed in the papers?'
    prompt += f'Imagine you are writing a paragraph of a survey paper. Write about {20 * len(probs)} words. Try to be concise. Make it coherent.'
    prompt += "IMPORTANT: Include all papers I gave you in your response!"
    prompt += "Don't forget to cite the papers you mention, using the arxiv ids of the papers."

    if model_path in API_MODELS:
        response = generate_response_api(model=model, 
                                         model_path=model_path,
                                         prompt=prompt)
    else:
        response = None
        raise ValueError(f"Invalid model_path. Choose from {API_MODELS}.")

    response = add_hyperlink(response)  # Add hyperlink to arXiv IDs
    return response

def add_hyperlink(text):
    arxiv_ids = set(detect_arxiv_id(text))
    for aid in arxiv_ids:
        text = text.replace(aid, f"[{aid}](https://arxiv.org/abs/{aid})")
    return text

def detect_arxiv_id(text):
    import re
    arxiv_id = re.findall(r'\d+\.\d+(?:v\d+)?', text)
    return arxiv_id

def get_problem(text):
    lines = text.strip().split('\n')
    probs = []
    for i, line in enumerate(lines):
        if 'Problem' in line:
            # Find associated paper
            paper = None                                                                                                                                                                
            for j in range(1, i+1):                                                                                                                                                   
                if 'Paper' in lines[i-j]:
                    paper = lines[i-j]
                    break
            if paper: # If paper is found
                prob = f'{paper}. {line}'
                prob = prob.replace('-', '').strip()
                prob = ' '.join(prob.split())
                probs.append(prob)
    return probs

def rearrange_probs(probs):
    # Load the sentence embedding model
    model_name = 'all-mpnet-base-v2'  # You can experiment with other models
    embedder = SentenceTransformer(model_name) 

    # Create sentence embeddings
    probs_clean = [prob.split(':')[-1].strip() for prob in probs]
    # print(probs_clean)
    sentence_embeddings = embedder.encode(probs_clean)

    # Cluster the embeddings using HDBSCAN
    clusterer = hdbscan.HDBSCAN()
    clusterer.fit(sentence_embeddings)

    # Analyze the results
    labels = clusterer.labels_
    print('Number of clusters:', len(set(labels)))

    if len(set(labels)) > 1:
        # Create a DataFrame for easy manipulation
        df = pd.DataFrame({'sentence': probs, 'cluster': labels})

        # Sort sentences within each cluster
        df = df.sort_values(by=['cluster', 'sentence'])

        # Extract the rearranged list of sentences
        rearranged_probs = df['sentence'].tolist()
    else:
        random.seed(2024)
        rearranged_probs = probs
        random.shuffle(rearranged_probs)
    return rearranged_probs