import torch
from .query import get_summaries
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from dotenv import load_dotenv
from .const import API_MODELS
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

    INSTRUCTION = """\n\nLet's think step by step. Imagine you are writing a survey paper.
            You must mention all the papers in the summaries above, and summarize its contributions in 3-4 sentences for each paper.
            The format for each paper should be as follows:
                - Problem: What is the main problem addressed in the paper? Be as detailed as possible about the problem.
                - Contribution: What is the main contribution of tda actihe paper?
                - Limitation: What is possible limitation?
            Remember to mention the paper when you refer to its content. You can mention the papers in an arbitrary order, as long as the flow is smooth.
            """

    # API models
    if model_path in API_MODELS:
        assert len(summaries) % 10 == 0, 'Number of papers must be divisible by 10'
        responses = ''
        for i in range(len(summaries)//10):
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
    
    
def generate_response_api(model, model_path, prompt):
    if model_path == 'gemini':
        response = model.generate_content(prompt).text
    elif 'gpt' in model_path:
        if model_path == 'gpt3.5':
            gpt_version = "gpt-3.5-turbo"
        elif model_path == 'gpt4':
            gpt_version = "gpt-4-turbo"
        else:
            raise ValueError('Invalid model. Choose from (gpt3.5, gpt4).')
        completion = model.chat.completions.create(
                        model=gpt_version,
                        messages=[
                            {"role": "system", "content": "You are researcher writing a survey paper."},
                            {"role": "user", "content": prompt}
                        ]
                        )
        response = completion.choices[0].message.content
    elif 'claude' in model_path:
        if model_path == 'claude-haiku':
            claude_version = 'claude-3-haiku-20240307'
        elif model_path == 'claude-sonnet':
            claude_version = 'claude-3-sonnet-20240229'
        else:
            raise ValueError('Invalid model. Choose from (claude-haiku, claude-sonnet).')
        message = model.messages.create(
                        model=claude_version,
                        max_tokens=2048,
                        temperature=0.3,
                        system="You are researcher writing a survey paper.",
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )
        response = message.content[0].text
    else:
        raise ValueError(f'Invalid model. Choose from {API_MODELS}.')
    return response

    
def load_model_tokenizer(model_path='gemini'):
    # Gemini model
    if model_path == 'gemini':
        import google.generativeai as genai

        GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-pro')
        tokenizer = None

    # OpenAI models
    elif model_path == 'gpt3.5' or model_path == 'gpt4':
        from openai import OpenAI

        model = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        tokenizer = None

    # Claude models
    elif 'claude' in model_path:
        import anthropic

        model = anthropic.Anthropic(
            api_key=os.getenv('ANTHROPIC_API_KEY'),
        )
        tokenizer = None
    
    # Local models
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, 
                                                device_map="auto",
                                                trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, 
                                                    device_map="auto", 
                                                    trust_remote_code=True)
    return model, tokenizer