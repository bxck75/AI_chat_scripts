import os,sys,re
from rich import print as rp
from langchain import hub
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
import os
import re
import tempfile
import pandas as pd
from typing import List, Dict, Any
from langchain_core.runnables.base import Runnable
from langchain.agents.utils import validate_tools_single_input
from langchain_community.docstore import Wikipedia
from langchain_core.prompts import PromptTemplate
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, QFileDialog, QLabel
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain_core.documents import Document
from components.hugging_chat_wrapper import HuggingChatWrapper
from PyQt6.QtWidgets import ( QApplication, QMainWindow, QWidget, QVBoxLayout, QTabWidget,  QTextEdit, QLineEdit, QPushButton)
from langchain.agents import Tool, AgentExecutorIterator,AgentType, initialize_agent,AgentExecutor
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import create_react_agent,create_self_ask_with_search_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain_experimental.llm_bash.bash import BashProcess
from langchain_community.tools import ShellTool
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain_core.output_parsers.string import StrOutputParser
from langchain.memory import ConversationBufferMemory,ConversationSummaryBufferMemory,CombinedMemory
from langchain_community.tools.file_management import (
        CopyFileTool,
        DeleteFileTool,
        FileSearchTool,
        ListDirectoryTool,
        MoveFileTool,
        ReadFileTool,
        WriteFileTool,
    )
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_community.tools.wikidata.tool import WikidataAPIWrapper, WikidataQueryRun
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


# Initialize the search tool
wikidata = WikidataQueryRun(api_wrapper=WikidataAPIWrapper())
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
ddg = DuckDuckGoSearchRun(api_wrapper=DuckDuckGoSearchAPIWrapper())


# Define tools
file_copy = CopyFileTool()
dir_list = ListDirectoryTool()
read_file = ReadFileTool()
write_file = WriteFileTool()
move_file = MoveFileTool()
file_search = FileSearchTool()
bash_shell = ShellTool()
python_repl = PythonAstREPLTool()

tools =[
    Tool(
        name = 'Copy File',
        func = file_copy.run,
        verbose=True,
        description = '''
        A tool to copy files from one location to another.
        Input should be a dictionary with 'source' and 'destination' keys.
        '''
    ),
    Tool(
        name = 'List Directory',
        func = file_copy.run,
        verbose=True,
        description = '''
        A tool to list files in a directory.
        Input should be a dictionary with 'directory' key.
        '''
    ),
    Tool(
        name = 'List Directory',
        func = file_copy.run,
        verbose=True,
        description = '''
        A tool to read a file and return its content.
        Input should be a dictionary with 'file' key.
        '''
    ),
    Tool(
        name = 'Write File',
        func = file_copy.run,
        verbose=True,
        description = '''
        A tool to write content to a file.
        Input should be a dictionary with 'content' and 'file' keys.
        '''
    ),
    Tool(
        name = 'Move File',
        func = move_file.run,
        verbose=True,
        description = '''
        A tool to move files from one location to another.
        Input should be a dictionary with 'source' and 'destination' keys.
        '''
    ),
    Tool(
        name = 'File Search',
        func = file_search.run,
        verbose=True,
        description = '''
        A tool to search for files.
        Input should be a dictionary with 'query' and 'directory' keys.
        '''
    ),
    Tool(
        name = 'Python REPL',
        func = python_repl.run,
        handle_tool_error=True,
        handle_validation_error=True,
        verbose=True,
        description = '''
        A Python shell. Use this to execute python commands. 
        Input should be a valid python command. 
        When using this tool, sometimes output is abbreviated - make sure 
        it does not look abbreviated before using it in your answer.
        '''
    ),
    Tool(
        name = 'DuckDuckGo Search',
        func = ddg.run,
        verbose=True,
        description = '''
        A wrapper around DuckDuckGo Search. 
        Useful for when you need to answer questions about current events. 
        Input should be a search query.
        '''
    ),
    Tool(
        name = 'Bash Shell',
        func = bash_shell.run,
        verbose=True,
        description = '''
        A tool to run bash commands.
        Use this to execute bash commands.
        Input should be a string with the bash command.
        Example: 'mkdir new_project && cd new_project && ls -l'
        '''
    )
]


wikitool = Tool( name="Intermediate Answer",func=wikipedia.run,description='''
                                            A wrapper around Wikipedia Search 
                                            Useful for when you need to answer questions about current events. 
                                            Input should be a search query'''
         )
ddg = Tool( name="Intermediate Answer",func=ddg.run, description='''
                                            A wrapper around DuckDuckGo Search. 
                                            Useful for when you need to answer questions about current events. 
                                            Input should be a search query.'''
         )


#self_ask_prompt = hub.pull("hwchase17/self-ask-with-search", api_key=os.getenv('HUGGINGFACEHUB_API_TOKEN'))

template = '''
Examples:
    Question                            : Are both the directors of Jaws and Casino Royale from the same country?
    Are follow up questions needed here : Yes.
    Follow up                           : Who is the director of Jaws?
    Intermediate answer                 : The director of Jaws is Steven Spielberg.
    Follow up                           : Where is Steven Spielberg from?
    Intermediate answer                 : The United States.
    Follow up                           : Who is the director of Casino Royale?
    Intermediate answer                 : The director of Casino Royale is Martin Campbell.
    Follow up                           : Where is Martin Campbell from?
    Intermediate answer                 : New Zealand.
    So the final answer is              : No

History: {chat_history}
Context: {context}
Question: {input}

Are followup questions needed here:{agent_scratchpad}'''

wrapper = HuggingChatWrapper(project_name='AgenticDashboard')
from typing import Any, List, Mapping, Optional
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import Field
from components.hugging_chat_wrapper import HuggingChatWrapper

class HuggingChatLLM(LLM):
    project_name: str = Field(default="AgenticDashboard")
    system_prompt: str = Field(default="You are a helpful AI assistant.")
    model_id: int = Field(default=0)
    
    _chatbot: Optional[HuggingChatWrapper] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._chatbot = HuggingChatWrapper(project_name=self.project_name)
        self._chatbot.set_bot(system_prompt=self.system_prompt, model_id=self.model_id)

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
        #rp(f"input:{input.text}")
        msgs=input.to_messages()
        #rp(f"msgs:{msgs}")
        adjective = msgs[0].content#['adjective']
        #rp(f"adjective:{adjective}")

        return str(self.chatbot.chat(adjective))

'''Runnable llm'''
runnable_llm = RunnableChatBot(wrapper.set_bot(system_prompt="""
                    You are a python project development guru and you task is to develop a full blown python project with the users input as guide. 
                    -respond in encapsulated artifacts of the following types:
                        ['python', 'yaml', 'image_description', 'text', 'mermaid', 'chat', 'json', 'bash']
                     example pythonand image descriptions artifact:
                        **main.py**
                        ```python
                            print('hello Word')
                        ```
                        **hello_world.jpg
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
                    -Extra context is auto-augmented if available
                    -Make sure to fully implement all code to their files and leave no placeholders
                    -Make sure that the final result is a stable application complete with clipart.
                                               
                    Stay away from dangerous python/bash commands and develop/overwrite in the Tempdir.
                    Finaly, 
                        When all is done!, 
                                    Make a permanent folder in the 'FinalProjects' folder and move the project to its new home!
                                        """,  model_id=0)
                                )


# Default llm
huggingchat_llm = HuggingChatLLM(
    project_name="MyProject",
    system_prompt="You are a Python expert. Provide concise and accurate answers.",
    model_id=1
)
conv_memory = ConversationBufferMemory(
    memory_key="chat_history_lines", input_key="input"
)
# Now you can use huggingchat_llm wherever LangChain expects an LLM
summary_memory = ConversationSummaryBufferMemory(llm=huggingchat_llm, input_key="input")

# Combined
memory = CombinedMemory(memories=[conv_memory, summary_memory])

prompt = PromptTemplate.from_template(
    template,
    partial_variables={
        "chat_history": "No previous conversation.",
        "agent_scratchpad": """
                                                          
                    Tip:Try limiting the maximum number of python artifacts in a response to 3"""
    }
)

self_ask_tool = [ddg] 
# Ask your question (replace this with your question)
npc_react = create_self_ask_with_search_agent(runnable_llm,  self_ask_tool, prompt)


executor = AgentExecutor(
    name='Joop',
    memory=memory,
    agent=npc_react,
    tools = tools,
    verbose = False, 
    max_iterations = 10,
    tags=['game','pyqt6','AI'],
    return_intermediate_steps=True,
    handle_parsing_errors=True, # continue on error 
    
) 



from typing import Literal

from langchain_core.tools import tool


@tool
def get_weather(city: Literal["nyc", "sf"]):
    """Use this to get weather information."""
    if city == "nyc":
        return "It might be cloudy in nyc"
    elif city == "sf":
        return "It's always sunny in sf"
    else:
        raise AssertionError("Unknown city")

# Define the graph

from langgraph.prebuilt import create_react_agent

graph = create_react_agent(llm = RunnableChatBot(wrapper.set_bot(system_prompt="")), tools=tools)

def parse_project_structure(content: str, base_dir: str):
    """
    Parses the project structure from the content and creates the directories and files
    inside the specified base directory.
    
    :param content: The string content containing the project structure.
    :param base_dir: The base directory where the structure will be created.
    """
    # Match each line representing a directory or file
    pattern = r'\|----\s*(.*)'  # Match each line starting with '|----'
    lines = re.findall(pattern, content)
    
    current_path = base_dir
    for line in lines:
        # Check if the line ends with a directory slash '/'
        if line.endswith('/'):
            dir_path = os.path.join(current_path, line.rstrip('/').strip())
            os.makedirs(dir_path, exist_ok=True)
        else:
            # This is a file
            file_path = os.path.join(current_path, line.strip())
            os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure directories exist
            with open(file_path, 'w') as f:
                f.write('')  # Create an empty file

question ="""
        Make [[['TowerSim']]] a skyscraper tycoon game
        Real banking system, Inventory, cashflow for player so he can build new levels on the skyscraper  
        'rooms',shops','offices' for rent  
        'elevators' to transport people to their homes and work etc,
        random +/- events, fluctuate prices, 
        write user manual, and describe clipart and cover images
        the goals of the game is a mega high skyscraper and lots money 
        """
#question=""""""
output = executor.invoke({'input': question, 'context': wrapper.knowledge_retriever.retrieve(query=question, nr_of_docs=1)})

artifact_detector = wrapper.artifact_detector
#raw_artifacts = artifact_detector.detect_artifacts(output)
artifacts = artifact_detector._extract_filenames_and_artifacts(output) 
for artifact in artifacts: 
    print(f"Filename: {artifact[0]}") 
    print(f"Code Block: {artifact[1]}") 
    print("="*40)
rp(f"Final output: {dir(output)}")