from .utils import generate_response_api, detect_arxiv_id, get_survey_prompt, load_model_tokenizer
from .const import API_MODELS, NO_PAPER_RESPONSE
from .cluster import cluster_papers
from pprint import pprint

def summarize_content(response, 
                      q, 
                      model, 
                      model_path, 
                      max_new_tokens=2048):
    if response == NO_PAPER_RESPONSE:
        return NO_PAPER_RESPONSE
    
    # Get clusters and cluster responses
    clusters = cluster_papers(response)
    cluster_responses = get_cluster_responses(response, clusters=clusters)

    # Refine the cluster responses and generate the main contents of the survey
    main_content = ""
    for cluster, content in cluster_responses.items():
        # Extract cluster title
        cluster_title = cluster.replace('**', '').replace('[', '').replace(']', '').split(':')[1].strip()

        # Refine the cluster response
        refined_content = refine_cluster_response(cluster_response=content, 
                                                  cluster_title=cluster_title,
                                                  model_path=model_path)

        # Generate cluster content
        cluster_content = f"**{cluster_title}**\n\n{refined_content}\n\n"
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
    cluster_responses = {}
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
    for cluster, arxiv_ids in clusters.items():
        cluster_response = ""
        for pid in arxiv_ids:
            cluster_response += papers[pid] + ' '
        cluster_responses[cluster] = cluster_response
    return cluster_responses

def refine_cluster_response(cluster_response, 
                            cluster_title, 
                            model_path):
    """
    Refine the cluster response to generate a coherent content.
    """
    prompt = f"""This section is about {cluster_title}. Use this information, improve the transition between sentences in this paragraph. 
    Try to point out key similarities and differences in approaches and findings among these papers.
    Return the revision only, no need for any further introduction. You can also revise typos and wrong formatting. Don't add words like "revised paragraph" or so. Here is the paragraph:
{cluster_response}

You can break a long paragraph into 2-3 paragraphs if necessary, but don't break too much. Also note to improve the transition between paragraphs. It's best to point out the similarity of difference between two papers when you transition between them.

Remember to bold the paper IDs in the format **Paper 2012.12345** in the revision.

**Important**: Don't add any new paper, especially when there is only one paper in the original text.
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
    
    # Further refine the response
    # refined_response = refined_response.replace('\n', '')
    if ':' in refined_response:
        refined_response = ' '.join(refined_response.split(':')[1:])
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

