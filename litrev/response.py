from .utils import generate_response_api, detect_arxiv_id, get_survey_prompt, load_model_tokenizer
from .const import API_MODELS, NO_PAPER_RESPONSE
from .cluster import cluster_papers_embedding
from pprint import pprint

def summarize_content(response, 
                      q, 
                      model, 
                      model_path, 
                      max_new_tokens=2048):
    if response == NO_PAPER_RESPONSE:
        return NO_PAPER_RESPONSE
    
    # Get clusters and cluster responses
    clusters = cluster_papers_embedding(response)
    cluster_responses = get_cluster_responses(response, clusters=clusters)

    # Refine the cluster responses and generate the main contents of the survey
    main_content = ""
    for i in range(len(clusters.keys())):
        content = cluster_responses[i]
        # Refine the cluster response
        gen_step = 0
        while True:
            cluster_refined_content = refine_response(response=content,
                                                      model_path=model_path)
            gen_step += 1
            if len(set(detect_arxiv_id(cluster_refined_content))) == len(set(detect_arxiv_id(content))):
                break
            elif gen_step > 3:
                break   # break if the response is not refined after 3 steps
        
        # Generate cluster content
        cluster_content = f"**{i+1}.** {cluster_refined_content}\n\n"
        main_content += cluster_content
    
    # Extract arXiv IDs
    arxiv_ids = detect_arxiv_id(main_content)
    
    # Generate response
    prompt = get_survey_prompt(main_content, 
                               q=q)
    if model_path in API_MODELS:
        response = generate_response_api(model=model, 
                                         model_path=model_path,
                                         prompt=prompt, 
                                         max_new_tokens=max_new_tokens
                                         )
    else:
        response = None
        raise ValueError(f"Invalid model_path. Choose from {API_MODELS}.")
    
    # Extract intro, conclusion, and title
    intro = extract_intro(response)
    conclusion = extract_conclusion(response)
    title = extract_title(response)

    # Refine response
    response = f"""
#### {title}

{intro}

{main_content}

{conclusion}
"""
    response = add_hyperlink(response, arxiv_ids)  # Add hyperlink to arXiv IDs
    return response

def get_cluster_responses(response, clusters):
    """
    Generate responses for each cluster based on the paper content.
    """

    # Split the response into individual papers
    paper_contents = response.split('\n\n')

    # Extract paper content and store in dictionary
    papers = {}
    for content in paper_contents:
        try:
            pid = detect_arxiv_id(content)[0]
            papers[pid] = content
        except IndexError:
            print(content)
    
    # Generate response for each cluster
    cluster_responses = {}
    for cluster, pids in clusters.items():
        cluster_response = ""
        for pid in pids:
            cluster_response += papers[pid] + ' '
        cluster_responses[cluster] = cluster_response
    return cluster_responses

def refine_response(response,
                    model_path):
    """
    Refine the cluster response to generate a coherent content.
    """
    prompt = f"""Refine the following paragraph to maintain smooth transitions between sentences in this paragraph. 
    Try to point out key similarities and differences in approaches and findings among these papers.
    You can also revise typos and wrong formatting. 
    Here is the paragraph:
{response}

    Your output should be as follows:
    "**Title of the paragraph (in 4-6 words, be specific and direct; don't use general words like deep learning, machine learning, etc.)**

    The refined paragraph"

    Write in a single paragraph. Don't include sentence like "The refined paragraph...". Remember to bold the paper IDs in the format **Paper 2012.12345** in the revision.
"""
    model, _ = load_model_tokenizer(model_path=model_path)
    if model_path in API_MODELS:
        refined_response = generate_response_api(model=model, 
                                                 model_path=model_path,
                                                 prompt=prompt, 
                                                 max_new_tokens=2048
                                                )
    else:
        refined_response = None
        raise ValueError(f"Invalid model_path. Choose from {API_MODELS}.")
    return refined_response

def extract_intro(text):
    """
    Extract the introduction section from response of AI.
    """
    lines = text.strip().split('\n')
    intro = ""
    for line in lines:
        if "Introduction" in line:
            intro = line
            break
    intro = intro.replace("Introduction:", "").strip()
    return intro

def extract_conclusion(text):
    """
    Extract the conclusion section from response of AI.
    """
    lines = text.strip().split('\n')
    conclusion = ""
    for line in lines:
        if "Conclusion" in line:
            conclusion = line
            break
    conclusion = conclusion.replace("Conclusion:", "").strip()
    return conclusion

def extract_title(text):
    """
    Extract the title section from response of AI.
    """
    lines = text.strip().split('\n')
    title = ""
    for line in lines:
        if "Title" in line:
            title = line
            break
    title = title.replace("Title:", "").strip()
    return title

def add_hyperlink(text, arxiv_ids):
    for aid in arxiv_ids:
        text = text.replace(aid, f"[{aid}](https://arxiv.org/abs/{aid})")
    return text

