from .const import API_MODELS
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams

nltk.download('stopwords')  # Download stopwords if you haven't already
nltk.download('punkt')  # Download for tokenization

def generate_response_api(model, model_path, prompt, max_new_tokens=2048):
    if 'gemini' in model_path:
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
        elif model_path == 'claude-opus':
            claude_version = 'claude-3-opus-20240229'
        else:
            raise ValueError('Invalid model. Choose from (claude-haiku, claude-sonnet, claude-opus).')
        message = model.messages.create(
                        model=claude_version,
                        max_tokens=max_new_tokens,
                        temperature=0.5,
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
    if 'gemini' in model_path:
        import google.generativeai as genai

        GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
        genai.configure(api_key=GOOGLE_API_KEY)

        if model_path == 'gemini-1':
            model = genai.GenerativeModel('gemini-pro')
        else:
            raise ValueError('Invalid model. Choose from (gemini-1,).')
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

def preprocess_text(text):
    """Preprocesses a single text"""
    text = text.lower()  # Lowercase
    text = re.sub(r'[^\w\s-]', ' ', text)  # Remove punctuation except hyphen
    # text = re.sub(r'\d+', '', text)  # Remove numbers
    text = ' '.join(text.split())  # Remove extra spaces
    preprocessed_text = text
    words = nltk.word_tokenize(text)  # Tokenize
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]  # Remove stopwords
    preprocessed_text = ' '.join(words)
    return preprocessed_text

def get_related_query(q):
    prompt = f"""Given a research topic, generate a list of 2 alternate ways to express the same research area or concept. The generated topics should be in 4-6 words, use familiar words/concepts/phrases in the field, and maintain the core meaning and intent of the original query while using different terminology or phrasing. You may consider to keep some original words and only change the rest. Don't use "AI systems" or similar phrases. Use "AI" instead of "artificial intelligence". Please provide the topics as a comma-separated list, without any additional explanations or descriptions.
Original research topic: {q}
Generated alternative expressions:
    """
    model_path = 'claude-sonnet'
    model, _ = load_model_tokenizer(model_path=model_path)

    response = generate_response_api(prompt=prompt, model=model, model_path=model_path)
    related_qs = response.split(',')
    related_qs = [q.strip() for q in related_qs if q]
    return related_qs

def detect_arxiv_id(text):
    import re
    arxiv_ids = re.findall(r'\d+\.\d+(?:v\d+)?', text)
    arxiv_ids = set(arxiv_ids)
    arxiv_ids = [item for item in arxiv_ids if len(item) > 5]
    return arxiv_ids

def get_survey_prompt(response, q):
    return f"""Instruction: Please add the following elements to the provided research survey:
1. Create a clear, concise title that accurately reflects the topic and purpose of the survey. You can further use the user query "{q}" to improve the introduction. Label this paragraph as "Title". For example: "Title: Survey on Machine Learning Algorithms".
2. Write a brief introduction that explains the survey's purpose, importance, and any necessary background information. The introduction should encourage participation and explain how the data will be used. Label this paragraph as "Introduction". For example: "Introduction: This survey...".
3. Write a short conclusion to summarize and propose future directions. Label this paragraph as "Conclusion". For example: "Conclusion: In conclusion, this survey...".

Please make these additions without altering the existing text of the research. The original text should be represented as "[original text]".

The survey: 
"{response}"

The survey with additions:
"""

def get_bigrams(text):
    """
    Extracts 2-grams (bigrams) from a given text.

    Args:
        text: The input text string.

    Returns:
        A list of bigrams, where each bigram is a tuple of two consecutive words.
    """

    tokens = nltk.word_tokenize(text)  # Tokenize the text into words
    bigrams = ngrams(tokens, 2)         # Generate bigrams using nltk's ngrams function
    return list(bigrams)                # Convert to a list for easier use 


