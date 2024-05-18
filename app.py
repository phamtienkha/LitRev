import asyncio
from litrev.const import HELLO_MSG, API_MODELS, INDEX_TYPE
from litrev.llm import generate_response
from litrev.utils import load_model_tokenizer
from litrev.response import summarize_content

# Create a new event loop
loop = asyncio.new_event_loop()

# Set the event loop as the current event loop
asyncio.set_event_loop(loop)

import streamlit as st

model_gemini_1, _ = load_model_tokenizer(model_path='gemini-1')

# App title
st.set_page_config(page_title="Literature Review Chatbot", page_icon=":robot:")

# Sidebar
with st.sidebar:
    st.title('Literature Review Chatbot')
    k_search = st.sidebar.slider('#papers', min_value=10, max_value=100, value=30, step=10)
    model_probs = st.sidebar.selectbox('Choose a model', API_MODELS, key='model_probs', index=3)
    index_type = st.sidebar.selectbox('Choose a indexing type', INDEX_TYPE, key='index', index=1)


# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": HELLO_MSG}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def new_conversation():
    st.session_state.messages = [{"role": "assistant", "content": HELLO_MSG}]
st.sidebar.button('New Conversation', on_click=new_conversation)

# Load mode and tokenizer for problem summarization
model, _ = load_model_tokenizer(model_path=model_probs)

# Function for generating response.
def get_response(prompt_input):
    string_dialogue = ""
    for dict_message in st.session_state.messages[-3:]:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"

    # prompt = build_prompt(prompt_input, context=string_dialogue)
    response = generate_response(q=prompt_input, 
                                 model=model_gemini_1, 
                                 model_path='gemini-1',
                                 tokenizer=None, 
                                 k_search=k_search,
                                 index_type=index_type)
    probs = summarize_content(response=response,
                              q=prompt_input, 
                              model=model,
                              model_path=model_probs, 
                              max_new_tokens=4096)
    return probs

# User-provided prompt
if user_input := st.chat_input(disabled=False, placeholder="Search for a research topic"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_response(user_input)
            placeholder = st.empty()
            placeholder.markdown(response)

    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
