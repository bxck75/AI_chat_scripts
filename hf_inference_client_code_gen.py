# filename: hf_inference_client_code_gen.py

import os
from huggingface_hub import InferenceClient
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from components.hugging_chat_wrapper import HuggingChatWrapper
from langchain_core.runnables.base import Runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain_core.output_parsers.string import StrOutputParser
from components.hugging_chat_wrapper import HuggingChatWrapper
from typing import Any, List, Mapping, Optional
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import Field
from components.hugging_chat_wrapper import HuggingChatWrapper

import json
from rich import print as rp
# Set HuggingFace API key from environment variables
huggingface_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")  # Ensure this is set
wrapper=HuggingChatWrapper("hf_inference")

system_prompt= """You are a python project development guru and you task is to develop a full blown python project with the users input as guide. 
                    -respond in encapsulated artifacts of the following types:
                        ['python', 'yaml', 'image_description', 'text', 'mermaid', 'chat', 'json', 'bash']
                     example pythonand image descriptions artifact:
                        **main.py**
                        ```python
                            print('hello Word')
                        ```
                        **hello_world.jpg**
                        ```image_description
                        A logo of Hello World in fluffy furry pink
                        ```
                    -Responding with encapsulated artifacst triggers actions:
                                            -(over)write the files/folders in a TempDir.
                                            -generate images
                                            -render mermaid flowcharts
                                            -save text artifacts as user guide
                                            -append chat artifacts to the chat history
                    -All artifacts are stored in a vectorstore 
                    -Extra context is auto-augmented if available from the vectorstore
                    -Make sure to fully implement all code to their files and leave no placeholders
                    -Make sure that the final result is a stable application complete with clipart.
                                               
                    Stay away from dangerous python/bash commands and develop/overwrite in the Tempdir.
                    Finaly, 
                        When all is done!, 
                                    Make a permanent folder in the 'FinalProjects' folder and move the project to its new home!"""



  

class HuggingChatLLM(LLM):
    project_name: str = Field(default="AgenticDashboard")
    system_prompt: str = Field(default=system_prompt)
    model_id: int = Field(default=0)
    
    _chatbot: Optional[HuggingChatWrapper] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._chatbot = wrapper.set_bot(system_prompt=system_prompt,model_id=0)
       

    @property
    def _llm_type(self) -> str:
        return "huggingchat"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        response = self._chatbot.chat(prompt)
        if stop:
            for stop_sequence in stop:
                if stop_sequence in response:
                    response = response[:response.index(stop_sequence)]
        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "project_name": self.project_name,
            "system_prompt": self.system_prompt,
            "model_id": self.model_id,
        }

class RunnableChatBot(Runnable):
    def __init__(self, chatbot):
        self.chatbot = chatbot
    
    def invoke(self, input, config=None, **kwargs):
        
        task=input.get('input')
        rp(f"task:{task}")

        return str(self.chatbot.chat(task)) 


chatbot = RunnableChatBot(wrapper.set_bot(system_prompt=system_prompt,model_id=0))
                                                                                                                                       
# Prompt template for code generatio

# Data model for structured output
class Code(BaseModel):
    """Code output"""

    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")
    description: str = "Schema for code solutions to questions about LCEL."

# Function to send the request to HuggingFace Inference API
def generate_code(question: str):
    #response = client.text_generation(model=model_id, prompt=question, max_new_tokens=200)
    response = chatbot.invoke(input={'input':question})
    return response

# Testing the inference with a question
question = """ Make [[['TowerSim']]] a skyscraper tycoon game
        Real banking system, Inventory, cashflow for player so he can build new levels on the skyscraper  
        'rooms',shops','offices' for rent  
        'elevators' to transport people to their homes and work etc,
        random +/- events, fluctuate prices, 
        write user manual, and describe clipart and cover images
        the goals of the game is a mega high skyscraper and lots money """
response = generate_code(question)
rp(f"Response:{response}")

# Select LLM
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field




# Prompt
code_gen_prompt_claude = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a coding assistant. Ensure any code you provide can be executed with all required imports and variables \n
            defined. Structure your answer: 1) a prefix describing the code solution, 2) the imports, 3) the functioning code block.
            \n Here is the user question:{input}""",
        ),
        ("placeholder", "{messages}"),
    ]
)



with open('__name__', 'r') as f:
    code = f.read().strip()
# LLM
code_gen_chain = chatbot.invoke({'input': code})