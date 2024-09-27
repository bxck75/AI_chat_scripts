""" project_structure.txt
textCopylangraph_chatbot/
├── main.py
├── requirements.txt
├── chatbot/
│   ├── __init__.py
│   ├── agent.py
│   ├── gui.py
│   ├── llm.py
│   ├── state.py
│   └── tools.py
└── README.md """
import sys
from typing import Annotated, Sequence, TypedDict
from PyQt6.QtWidgets import QApplication, QMainWindow, QTextEdit, QLineEdit, QPushButton, QVBoxLayout, QWidget
from PyQt6.QtCore import QThread, pyqtSignal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
import json
import os,sys,re
from rich import print as rp
from langchain import hub
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
import os
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

#from mediawikiapi import MediaWikiAPI
# Initialize the search tool
#wikidata = WikidataQueryRun(api_wrapper=WikidataAPIWrapper())
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
        rp(f"input:{input.text}")
        msgs=input
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
    system_prompt="""
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
                                        """,model_id=1
    )
conv_memory = ConversationBufferMemory(
    memory_key="chat_history_lines", input_key="input"
)
# Now you can use huggingchat_llm wherever LangChain expects an LLM
summary_memory = ConversationSummaryBufferMemory(llm=huggingchat_llm, input_key="input")

# Combined
memory = CombinedMemory(memories=[conv_memory, summary_memory])

import sys
from PyQt6.QtWidgets import QApplication
from chatbot.gui import ChatbotWindow
from chatbot.agent import create_graph
from typing import Any, List, Mapping, Optional
def main():
    app = QApplication(sys.argv)
    graph = create_graph()
    window = ChatbotWindow(graph)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
from langgraph.graph import StateGraph, END
from langchain.agents import AgentExecutor, AgentType, initialize_agent
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory, CombinedMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
#import AgentState
from components.hugging_chat_wrapper import HuggingChatLLM, RunnableChatBot


def create_graph():
    huggingchat_llm = HuggingChatLLM(
        project_name="MyProject",
        system_prompt="""
        You are a python project development guru and your task is to develop a full blown python project with the user's input as a guide. 
        Respond in encapsulated artifacts of the following types:
        ['python', 'yaml', 'image_description', 'text', 'mermaid', 'chat', 'json', 'bash']
        Make sure to fully implement all code to their files and leave no placeholders.
        Make sure that the final result is a stable application complete with clipart.
        Stay away from dangerous python/bash commands and develop/overwrite in the TempDir.
        When all is done, make a permanent folder in the 'FinalProjects' folder and move the project to its new home!
        """,
        model_id=1
    )

    conv_memory = ConversationBufferMemory(memory_key="chat_history_lines", input_key="input")
    summary_memory = ConversationSummaryBufferMemory(llm=huggingchat_llm, input_key="input")
    memory = CombinedMemory(memories=[conv_memory, summary_memory])

    def tool_node(state: AgentState):
        outputs = []
        for tool_call in state['messages'][-1].tool_calls:
            tool_result = next(tool for tool in tools if tool.name == tool_call["name"]).invoke(
                json.loads(tool_call["arguments"])
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}

    def call_model(state: AgentState):
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

        prompt = PromptTemplate.from_template(
            template,
            partial_variables={
                "chat_history": state['messages'],
                "agent_scratchpad": """
                Tip: Try limiting the maximum number of python artifacts in a response to 3
                """
            }
        )

        agent = initialize_agent(
            tools,
            huggingchat_llm,
            agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors="Check your output and make sure it conforms!",
        )

        executor = AgentExecutor(
            name='Joop',
            memory=memory,
            agent=agent,
            tools=tools,
            verbose=False,
            max_iterations=10,
            tags=['game', 'pyqt6', 'AI'],
            return_intermediate_steps=True,
            handle_parsing_errors=True,
        )

        system_prompt = SystemMessage(content="You are a helpful AI assistant. Please respond to the user's query to the best of your ability!")
        response = agent.invoke(input={'input': [state["messages"][-1]], 'context': system_prompt})
        return {"messages": [response]}

    def should_continue(state: AgentState):
        last_message = state["messages"][-1]
        return "continue" if last_message.tool_calls else "end"

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )
    workflow.add_edge("tools", "agent")
    return workflow.compile()
from PyQt6.QtWidgets import QMainWindow, QTextEdit, QLineEdit, QPushButton, QVBoxLayout, QWidget
from PyQt6.QtCore import QThread, pyqtSignal
from langchain_core.messages import HumanMessage, AIMessage

class AgentThread(QThread):
    update_signal = pyqtSignal(str)

    def __init__(self, graph, input_text):
        super().__init__()
        self.graph = graph
        self.input_text = input_text

    def run(self):
        inputs = {"messages": [HumanMessage(content=self.input_text)]}
        for output in self.graph.stream(inputs):
            message = output["messages"][-1]
            if isinstance(message, AIMessage):
                self.update_signal.emit(message.content)

class ChatbotWindow(QMainWindow):
    def __init__(self, graph):
        super().__init__()
        self.graph = graph
        self.setWindowTitle("LangGraph Agent Chatbot")
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        self.chat_history = QTextEdit(self)
        self.chat_history.setReadOnly(True)
        layout.addWidget(self.chat_history)

        self.input_field = QLineEdit(self)
        layout.addWidget(self.input_field)

        self.send_button = QPushButton("Send", self)
        self.send_button.clicked.connect(self.send_message)
        layout.addWidget(self.send_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def send_message(self):
        user_input = self.input_field.text()
        self.chat_history.append(f"You: {user_input}")
        self.input_field.clear()

        self.thread = AgentThread(self.graph, user_input)
        self.thread.update_signal.connect(self.update_chat)
        self.thread.start()

    def update_chat(self, message):
        self.chat_history.append(f"Agent: {message}")

from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import Field
from langchain_core.runnables.base import Runnable
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
        rp(f"input:{input}")
        # msgs = input
        adjective = input[0].content
        return str(self.chatbot.chat(adjective))

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

from langchain.tools import DuckDuckGoSearchRun
from langchain_community.tools.file_management import (
    CopyFileTool,
    DeleteFileTool,
    FileSearchTool,
    ListDirectoryTool,
    MoveFileTool,
    ReadFileTool,
    WriteFileTool,
)
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain_community.tools import ShellTool

def get_weather(location: str):
    """The current weather in a given location.
    Args:
        location (str): The city or town where you want the weather.
    Returns:
        str: A description of the weather in the specified location."""
    if any([city in location.lower() for city in ['sf','san francisco']]):
        return "It's sunny in San Francisco, but you better look out if you're a Gemini 😈."
    else:
        return f"I am not sure what the weather is in {location}"

tools = [
    Tool(name="Copy File", func=CopyFileTool().run, description="Copy files from one location to another."),
    Tool(name="List Directory", func=ListDirectoryTool().run, description="List files in a directory."),
    Tool(name="Read File", func=ReadFileTool().run, description="Read a file and return its content."),
    Tool(name="Write File", func=WriteFileTool().run, description="Write content to a file."),
    Tool(name="Move File", func=MoveFileTool().run, description="Move files from one location to another."),
    Tool(name="File Search", func=FileSearchTool().run, description="Search for files."),
    Tool(name="Python REPL", func=PythonAstREPLTool().run, description="Execute Python commands."),
    Tool(name="DuckDuckGo Search", func=DuckDuckGoSearchRun().run, description="Search the web using DuckDuckGo."),
    Tool(name="Bash Shell", func=ShellTool().run, description="Run bash commands."),
    Tool(name="Get Weather", func=get_weather, description="Get the current weather in a given location."),
]
""" requirements.txt
CopyPyQt6
langchain
langgraph
huggingface_hub
python-dotenv
rich
pandas
chromadb
README.md
markdownCopy# LangGraph Agent Chatbot """

#This project implements a chatbot using LangGraph and PyQt6. The chatbot uses a combination of tools and language models to provide intelligent responses to user queries.

## Installation

#1. Clone this repository
#2. Install the required packages:
#pip install -r requirements.txt

## Usage


#Enter your queries in the input field and click "Send" to interact with the chatbot.
""" 
## Project Structure

- `main.py`: Entry point of the application
- `chatbot/`: Contains the core components of the chatbot
  - `agent.py`: Defines the LangGraph agent and workflow
  - `gui.py`: Implements the PyQt6 GUI
  - `llm.py`: Contains the custom LLM implementations
  - `state.py`: Defines the agent state
  - `tools.py`: Defines the tools available to the agent

## Contributing

Feel free to submit issues or pull requests if you have any improvements or bug fixes.

## License

This project is licensed under the MIT License.
chatbot_logo.jpg
image_descriptionCopyA logo for the LangGraph Agent Chatbot. The image features a stylized chat bubble with a robotic face inside, symbolizing an AI-powered chatbot. The chat bubble is colored in shades of blue and green, representing intelligence and growth. The robotic face has glowing eyes and a friendly expression. Below the chat bubble, the text "LangGraph Agent Chatbot" is written in a modern, sans-serif font.
This project structure organizes the code into separate files for better maintainability and readability. The main.py file serves as the entry point, while the chatbot package contains the core components of the chatbot.
 """