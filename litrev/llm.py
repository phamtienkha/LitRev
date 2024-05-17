import torch
from .query import search_rank
from .query_vectordb import search_rank as search_rank_dl
from dotenv import load_dotenv
from .const import API_MODELS, NO_PAPER_RESPONSE
from .utils import generate_response_api, detect_arxiv_id
from aslite.db import get_papers_db
import math
import random

load_dotenv()  # Loads from '.env' file 

def generate_response(q,
                      model,
                      model_path, 
                      tokenizer, 
                      k_search,
                      temperature=1, 
                      max_new_tokens=2048, 
                      batch_size=1):
    summaries = get_summaries(q, top_k=k_search)
    print(f'There are {len(summaries)} papers found for your query: {q}\n')
    if len(summaries) == 0:
        return NO_PAPER_RESPONSE

    INSTRUCTION = f"""\n\nImagine you are writing a survey paper.
            You must mention all the papers in the summaries above. 
            Each paper is written in a single paragraph. You must mention the paper ID first, in bold with normal size in markdown. There is no bullet on the line of paper ID. For example: "**Paper 2012.05616**", not "**-Paper 2012.05616**". 
            Then, in the same paragraph, mention the research problem of the paper, and highlight the main contributions of the paper.

            Example: "**Paper 0000.11111** studies the problem of X. The main contributions of the paper are A, B, and C (write in the same paragraph, no bullets).
            
            **Paper 0000.22222** studies the problem of Y. It proposes the method of...

            ...
            "

            Follow the example, but you don't need to use the same wording. Be creative and diverse in your writing, but still accurate and informative. Avoid using "this paper" and bullets. Just use the paper ids as the subjects of the sentences.
            """

    # API models
    if model_path in API_MODELS:
        responses = ''
        for i in range(math.ceil(len(summaries)/10)):
            prompt = '\n\n'.join(summaries[i*10:(i+1)*10])
            prompt += INSTRUCTION
            response = generate_response_api(model=model, 
                                             model_path=model_path,
                                             prompt=prompt, 
                                             max_new_tokens=max_new_tokens
                                             )
            # Refine the response to remove hallucinated papers
            response = refine_llm_response(response, summaries)
            responses += response + '\n\n'
        return responses
    
    # Local models
    else:
        prompt = '\n'.join(summaries)
        prompt += INSTRUCTION

        if type(prompt) == str:
            input_ids = tokenizer([prompt]).input_ids
        else:
            tokenizer.pad_token = tokenizer.eos_token
            input_ids = tokenizer(prompt, padding=True).input_ids

        output_ids = []
        num_iter = len(input_ids) // batch_size + 1 if len(input_ids) % batch_size != 0 else len(input_ids) // batch_size
        for i in range(num_iter):
            input_ids_cur = input_ids[i*batch_size:(i+1)*batch_size]
            output_ids_cur = model.generate(
            torch.as_tensor(input_ids_cur).cuda(),
                do_sample=False,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
            output_ids.append(output_ids_cur)
        output_ids = torch.cat(output_ids, dim=0)

        if type(prompt) == str:
            output_ids = output_ids[0][len(input_ids[0]):]
            outputs = tokenizer.decode(output_ids, skip_special_tokens=True)
        else:
            outputs = []
            for i in range(len(output_ids)):
                output_ids_cur = output_ids[i, len(input_ids[i]):]
                output = tokenizer.decode(output_ids_cur, skip_special_tokens=True)
                outputs.append(output)
        return outputs

def get_summaries(q, type="dl", top_k=30):
    # Search for papers
    pdb = get_papers_db()
    if type == "rule":
        pids, _ = search_rank(q)
    else:
        pids, _ = search_rank_dl(q)
    top_pids = pids[:top_k]
    
    # Shuffle the top_pids
    top_pids = list(set(top_pids))
    random.seed(1234)
    random.shuffle(top_pids)

    # Get summaries
    summaries = []
    for pid in top_pids:
        summaries.append(f'Paper {pid}: ' + pdb[pid]["summary"])
    return summaries

def refine_llm_response(response, summaries):
    lines = response.strip().split('\n\n')
    refined_response = ''
    for line in lines:
        arxiv_id = detect_arxiv_id(line)[0]
        exists = [arxiv_id in summary for summary in summaries]
        if any(exists):
            refined_response += line + '\n\n'
    return refined_response
    