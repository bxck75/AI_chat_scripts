from langchain_experimental.agents.agent_toolkits.python.base import create_python_agent
from rich import print as rp
from langchain_community.tools import ShellTool
from langchain.agents.types import AgentType
from langchain.chains.llm import LLMChain
from langchain_core.callbacks.base import BaseCallbackManager
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import SystemMessage
from langchain import hub
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
from langchain.agents import Tool, AgentExecutor, create_react_agent,AgentType, initialize_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.prompts import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.tools.python.tool import PythonAstREPLTool
import os,sys,re
from rich import print as rp
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QTabWidget, 
    QTextEdit, QLineEdit, QPushButton
)
from langchain_core.runnables.base import Runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain_core.output_parsers.string import StrOutputParser
from components.hugging_chat_wrapper import HuggingChatWrapper
from langchain.memory import ConversationBufferMemory
from langchain_community.tools.file_management import (
        CopyFileTool,
        DeleteFileTool,
        FileSearchTool,
        ListDirectoryTool,
        MoveFileTool,
        ReadFileTool,
        WriteFileTool,
    )
from langchain_experimental.llm_bash.bash import BashProcess
from langchain_community.tools import ShellTool
from langchain_experimental.agents.agent_toolkits.python.prompt import PREFIX
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_experimental.tools.python.tool import PythonREPL
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_core.runnables.base import Runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain_core.output_parsers.string import StrOutputParser
from components.hugging_chat_wrapper import HuggingChatWrapper
from langchain.memory import ConversationBufferMemory
from langchain_community.tools.file_management import (
        CopyFileTool,
        DeleteFileTool,
        FileSearchTool,
        ListDirectoryTool,
        MoveFileTool,
        ReadFileTool,
        WriteFileTool,
    )
from langchain_experimental.llm_bash import BashProcess



# Content of the prompt template
template = '''
Design a top level plan and create the needed folders and files, Populate the files with classes and methods, 
Implement all classes with code conform autopep8 using the following instructions:
Do not use a tool if not required. 
Question: {question}
'''



bash = BashProcess(
    strip_newlines = True,
    return_err_output = True,
    persistent = True
)
bash.run(['echo "hello world"'])


class RunnableChatBot(Runnable):
    def __init__(self, chatbot):
        self.chatbot = chatbot
    
    def invoke(self, input, config=None, **kwargs):
        #rp(f"input:{input.text}")
        msgs=input.to_messages()
        #rp(f"msgs:{msgs}")
        adjective = msgs[0].content#['adjective']
        rp(f"adjective:{adjective}")

        return str(self.chatbot.chat(adjective))


# Setup the HuggingChatWrapper
wrapper = HuggingChatWrapper(project_name='AgenticDashboard')
runnable_bot = RunnableChatBot(wrapper.set_bot(system_prompt="""You are a python expert and our task is to develop python code with the users input as guide. 
# Set up the LLM                                                   Only output the fullly implemented code.""", model_id=2))
llm= runnable_bot

bash = BashProcess(persistent=True)


file_copy = CopyFileTool()
file_copy_tool = Tool(
    name = 'Copy File',
    func = file_copy.run,
    verbose=True,
    description = '''
    A tool to copy files from one location to another.
    Input should be a dictionary with 'source' and 'destination' keys.
    '''
)
dir_list = ListDirectoryTool()
dir_list_tool = Tool(
    name = 'List Directory',
    func = file_copy.run,
    verbose=True,
    description = '''
    A tool to list files in a directory.
    Input should be a dictionary with 'directory' key.
    '''
)
read_file =ReadFileTool()
read_file_tool = Tool(
    name = 'List Directory',
    func = file_copy.run,
    verbose=True,
    description = '''
    A tool to read a file and return its content.
    Input should be a dictionary with 'file' key.
'''
)
write_file = WriteFileTool()
write_file_tool = Tool(
    name = 'Write File',
    func = file_copy.run,
    verbose=True,
    description = '''
    A tool to write content to a file.
    Input should be a dictionary with 'content' and 'file' keys.
    
'''
)


# Set up the Move File tool
move_file = MoveFileTool()
move_file_tool = Tool(
    name = 'Move File',
    func = move_file.run,
    verbose=True,
    description = '''
    A tool to move files from one location to another.
    Input should be a dictionary with 'source' and 'destination' keys.
    '''
)

# Set up the File Search tool
file_search = FileSearchTool()
file_search_tool = Tool(
    name = 'File Search',
    func = file_search.run,
    verbose=True,
    description = '''
    A tool to search for files.
    Input should be a dictionary with 'query' and 'directory' keys.
    '''
)

python_repl = PythonAstREPLTool()
python_repl_tool = Tool(
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
)
# Set up the DuckDuckGo Search tool
search = DuckDuckGoSearchRun()
duckduckgo_tool = Tool(
    name = 'DuckDuckGo Search',
    func = search.run,
    verbose=True,
    description = '''
    A wrapper around DuckDuckGo Search. 
    Useful for when you need to answer questions about current events. 
    Input should be a search query.
    '''
)


bash_shell = ShellTool()
bash_shell_tool = Tool(
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


file_management_tools = [file_copy_tool,
                         dir_list_tool,
                         read_file_tool,
                         write_file_tool,
                         move_file_tool,
                         file_search_tool
                         ]

kut=[duckduckgo_tool, python_repl_tool, bash_shell_tool] 
# Create an array that contains all the tools used by the agent
tools = file_management_tools
# Create the prompt template
prompt_template = PromptTemplate.from_template(template)
prompt = hub.pull('hwchase17/react', api_key=os.getenv('HUGGINGFACEHUB_API_TOKEN'))
#rp(prompt)
# Create a ReAct agent
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    mame='Joop',
    memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
    agent=agent, 
    tools = tools,
    verbose = False, # explain all reasoning steps
    return_intermediate_steps=True,
    handle_parsing_errors=True, # continue on error 
    max_iterations = 10 # try up to 10 times to find the best answer
)



agent_executor = create_python_agent(
    llm=llm,
    tool= ShellTool(name='terminal',description=f"Run shell commands on this {_get_platform()} machine."),
    verbose=True
)
"""You have access to the terminal through the bash variable. Open the browser, 
                   go to youtube and search for Lex Fridman. 
                   Then write the text contents from the page into a file called page.txt"""

agent_executor.run('Dont write a python script this is a test....Use your directory list tool')