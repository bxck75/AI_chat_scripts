import streamlit as st
import os
from rich import print as rp
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())  # Assumes you have a.env file in the same directory as your script
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
EMAIL = os.getenv('EMAIL')
PASSWD = os.getenv('PASSWD')
from streamlit_chat import message
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.chains import ConversationChain
from langchain_core.runnables.base import Runnable
from hugchat import hugchat
from hugchat.login import Login

class RunnableChatBot(Runnable):
    def __init__(self, chatbot):
        self.chatbot = chatbot
    
    def invoke(self, input, config=None, **kwargs):
        #rp(input.text)
        # Extract the 'adjective' from the input dictionary
        adjective = input.text
        
        # Make a call to the HugChat API via the wrapped chatbot instance
        return str(self.chatbot.chat(adjective))

    # Keep the run method for backwards compatibility
    def run(self, input_text):
        return self.chatbot.chat(input_text)

st.set_page_config(page_title="HugChat - An LLM-powered Streamlit app")

# Sidebar contents
with st.sidebar:
    st.title('🤗💬 HugChat App')
    
    st.header('Hugging Face Chat')
    hf_email = EMAIL#st.text_input('Enter E-mail:', type='password')
    hf_pass = PASSWD#st.text_input('Enter password:', type='password')
    
    st.markdown('''The LLM-powered chatbot!''')
    add_vertical_space(5)
    st.write('Made with ❤️ by Me')

# Generate empty lists for generated and past.
## generated stores AI generated responses
if 'generated' not in st.session_state:
    st.session_state['generated'] = ["I'm HugChat, How may I help you?"]
## past stores User's questions
if 'past' not in st.session_state:
    st.session_state['past'] = ['Hi!']

# Layout of input/response containers
input_container = st.container()
colored_header(label='', description='', color_name='blue-30')
response_container = st.container()

# User input
## Function for taking user provided prompt as input
def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text
## Applying the user input box
with input_container:
    user_input = get_text()

# Response output
## Function for taking user prompt as input followed by producing AI generated responses
def generate_response(query, email, passwd, system_prompt="You are a good assistant",model_id=0):
    # Hugging Face Login
    sign = Login(email, passwd)
    cookies = sign.login()
    sign.save_cookies()
    prompt_template = "Tell me a {adjective} joke"
    prompt = PromptTemplate(
        input_variables=["adjective"], template=prompt_template
    )
    # Create ChatBot                        
    chatbot = hugchat.ChatBot(cookies=cookies.get_dict(), system_prompt=system_prompt, default_llm=model_id)
    runnable_chatbot = RunnableChatBot(chatbot)
    #response = chatbot.chat(prompt)
    chain = prompt | runnable_chatbot | StrOutputParser()
    
    response = chain.invoke(input={'adjective':query})

    return response

## Conditional display of AI generated responses as a function of user provided prompts
with response_container:
    if user_input and hf_email and hf_pass:
        response = generate_response(user_input, hf_email, hf_pass)
        st.session_state.past.append(user_input)
        rp(response)
        st.session_state.generated.append(response)
        
        
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))