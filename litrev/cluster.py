from .utils import detect_arxiv_id, preprocess_text
from aslite.db import get_papers_db
from pprint import pprint
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import numpy as np
import random

def cluster_papers_embedding(response):
    contents = get_content(response)
    if len(contents.keys()) <= 3:
        return {0: list(contents.keys())}
    CONTENT_LIST = [item.replace('\n', '') for item in list(contents.values())]
    NUM_CLUSTER_MIN = 3

    # Sentence encocoding
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Find best number of clusters for KMeans
    # Run KMeans multiple times and get the most common elbow index to ensure stability
    elbow_indices = {}
    indices = list(range(len(contents)))
    for _ in range(100):
        random.shuffle(indices)
        content_list = [CONTENT_LIST[i] for i in indices]

        X = model.encode(content_list)
        inertia = []
        for k in range(NUM_CLUSTER_MIN, min(15, len(contents.keys())+1)):
            kmeans = KMeans(n_clusters=k, random_state=28).fit(X)
            inertia.append(kmeans.inertia_)
        diff = np.diff(inertia)     # Calculate the difference between consecutive inertia values
        diff_diff = np.diff(diff)   # Calculate the rate of change of the difference
        elbow_index = np.argmax(diff_diff) + NUM_CLUSTER_MIN      # Find the index of the maximum rate of change of the difference
        
        if elbow_index not in elbow_indices:
            elbow_indices[elbow_index] = [indices]
        else:
            elbow_indices[elbow_index].append(indices)

    # Get the most common elbow index
    print({i: len(elbow_indices[i]) for i in elbow_indices})
    elbow_index = max(elbow_indices, key=lambda k: len(elbow_indices[k]))  # Get key with longest values

    # Run KMeans
    print('Number of clusters:', elbow_index)
    indices = elbow_indices[elbow_index][0]
    contents = {list(contents.keys())[i]: list(contents.values())[i] for i in indices}
    content_list = [preprocess_text(item).replace('\n', '') for item in list(contents.values())] 
    X = model.encode(content_list)
    clusters = KMeans(n_clusters=elbow_index, random_state=0).fit_predict(X)

    # Get items of each cluster
    cluster_items = {}
    for i, cluster in enumerate(clusters):
        if cluster not in cluster_items:
            cluster_items[cluster] = []
        cluster_items[cluster].append(list(contents.keys())[i])
    # print(cluster_items)
    
    # # Get titles of items in each cluster
    # cluster_titles = {}
    # pdb = get_papers_db()
    # for cluster, items in cluster_items.items():
    #     cluster_titles[cluster] = [pdb[item]['title'].replace('\n', '') for item in items]
    # pprint(cluster_titles)
    # exit()
    return cluster_items

def get_content(response):
    # Extract arXiv IDs and load paper db
    arxiv_ids = detect_arxiv_id(response)
    arxiv_ids = sorted(arxiv_ids)   # sort alphabetically
    pdb = get_papers_db()

    # Get titles
    contents = {}
    for aid in arxiv_ids:
        if aid in pdb:
            contents[aid] = f"{pdb[aid]['title']}"
    return contents

def extract_cluster(response_cluster, response):
    print(response_cluster)
    lines = response_cluster.strip().split('\n')
    results = {}

    current_cluster = None
    for line in lines:
        if "Cluster" in line:  # Check if it's a cluster header
            current_cluster = line  # Extract cluster name
            results[current_cluster] = {}
        elif line.strip():  # Check for non-empty lines (i.e., paper IDs)
            topic = line.split(':')[1].strip()
            results[current_cluster][topic] = []
            arxiv_ids = detect_arxiv_id(line)
            for arxix_id in arxiv_ids:
                if arxix_id in response:
                    results[current_cluster][topic].append(arxix_id)
    return results