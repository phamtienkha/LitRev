import argparse
from litrev.llm import generate_response, load_model_tokenizer
from litrev.response import summarize_content

if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--k_search", type=int, default=30)
    args = parser.parse_args()

    model_gemini, _ = load_model_tokenizer(model_path='gemini')
    model, tokenizer = load_model_tokenizer(model_path=args.model_path)

    q = input("Enter your query: ")
    response = generate_response(q=q, 
                                model=model_gemini, 
                                model_path='gemini',
                                tokenizer=None, 
                                k_search=args.k_search)
    probs = summarize_content(response=response,
                            model=model,
                            model_path=args.model_path)
    print(probs)
    
