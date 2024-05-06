from .utils import load_model_tokenizer, generate_response_api, detect_arxiv_id
from aslite.db import get_papers_db

def cluster_papers(response, model_path='claude-sonnet'):
    num_ids = len(detect_arxiv_id(response))
    titles = get_titles(response)
    prompt = f"""Given the following paper titles:
    "{titles}"

    Cluster the papers based on research ideas or research questions into {num_ids//10}-{num_ids//5} clusters. Follows these guidelines:
    - The cluster sizes should be balanced in average.
    - Include ALL {num_ids} paper ids given in the above passage.
    - Each cluster should have a clear theme or topic.
    - Within a cluster, more related papers are placed closer to each other.
    
    Return the paper ids only. The format should be as follows:
    Cluster 1: [name of Cluster 1]
    - Paper ID 1
    - Paper ID 2
    Cluster 2: [name of Cluster 2]
    - Paper ID 3
    - Paper ID 4
    ...

    No need for any further explanation or introduction.
    """
    model, _ = load_model_tokenizer(model_path=model_path)

    response_cluster = generate_response_api(prompt=prompt, model=model, model_path=model_path)
    clusters = extract_cluster(response_cluster, 
                               response=response)
    return clusters

def get_titles(response):
    # Extract arXiv IDs and load paper db
    arxiv_ids = detect_arxiv_id(response)
    pdb = get_papers_db()

    # Get titles
    titles = ''
    for aid in arxiv_ids:
        if aid in pdb:
            titles += f"Paper {aid}: {pdb[aid]['title']}\n\n"
    return titles

def extract_cluster(response_cluster, response):
    print(response_cluster)
    lines = response_cluster.strip().split('\n')
    results = {}

    current_cluster = None
    for line in lines:
        if "Cluster" in line:  # Check if it's a cluster header
            current_cluster = line  # Extract cluster name
            results[current_cluster] = []
        elif line.strip():  # Check for non-empty lines (i.e., paper IDs)
            arxiv_ids = detect_arxiv_id(line)
            for arxix_id in arxiv_ids:
                if arxix_id in response:
                    results[current_cluster].append(arxix_id)

    return results