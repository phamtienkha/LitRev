from .const import API_MODELS
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

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
                        max_tokens=4096,
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