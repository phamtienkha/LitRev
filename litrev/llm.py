import torch
from .query import search_rank
from dotenv import load_dotenv
from .const import API_MODELS
from .utils import generate_response_api
from aslite.db import get_papers_db
import math 

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

    INSTRUCTION = f"""\n\nLet's think step by step. Imagine you are writing a survey paper.
            You must mention all the papers in the summaries above, and summarize its contributions in 3-4 sentences for each paper.
            The format for each paper should be in four bullets as follows:
                - Paper ID: What is the ID of the paper? (e.g., Paper 2103.12345)
                - Problem: What is the main problem addressed in the paper? How this problem is related to {q}. Answer in two to three sentences.
                - Contribution: What is the main contribution of tda actihe paper? Answer in one line.
                - Limitation: What is possible limitation? Answer in one line.
            Remember to mention the paper when you refer to its content. You can mention the papers in an arbitrary order, as long as the flow is smooth.
            """

    # API models
    if model_path in API_MODELS:
        responses = ''
        for i in range(math.ceil(len(summaries)/10)):
            prompt = '\n'.join(summaries[i*10:(i+1)*10])
            prompt += INSTRUCTION
            response = generate_response_api(model=model, 
                                        model_path=model_path,
                                        prompt=prompt)
            responses += response + '\n'
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

def get_summaries(q, top_k=3):
    pdb = get_papers_db()
    pids, _ = search_rank(q)
    summaries = []
    for pid in pids[:top_k]:
        summaries.append(f'Paper {pid}: ' + pdb[pid]["summary"])
    return summaries
    