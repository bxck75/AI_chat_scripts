import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QTextEdit, QLabel, QSpinBox
from PyQt6.QtCore import QTimer, Qt
from langchain_core.runnables.base import Runnable
from components.hugging_chat_wrapper import HuggingChatWrapper
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import StringPromptTemplate
from langchain.chains import LLMChain
from typing import List, Dict, Any
from langchain import hub
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
from langchain.agents import Tool, AgentExecutor, create_react_agent,AgentType, initialize_agent,tool
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
import os
from langchain.agents import AgentType, initialize_agent
from langchain_community.agent_toolkits.github.toolkit import GitHubToolkit
from langchain_community.utilities.github import GitHubAPIWrapper
from langchain_core.runnables.base import Runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain_core.output_parsers.string import StrOutputParser
from components.hugging_chat_wrapper import HuggingChatWrapper
from langchain.memory import ConversationBufferMemory
from tempfile import TemporaryDirectory
from langchain_community.agent_toolkits import FileManagementToolkit
### Import LangChain Components and OpenAI API Key
from langchain.chains import LLMMathChain
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.tools import Tool

import logging

import time
from typing import Literal
from langchain_core.tools import tool


from langchain_experimental.pydantic_v1 import BaseModel

logging.basicConfig(level=logging.ERROR)

import os
from uuid import uuid4
### Setup the LangSmith environment variables
unique_id = uuid4().hex[0:8]
from langchain_experimental.llm_bash.bash import BashProcess
from langchain_community.tools import ShellTool
from langchain_community.cache import InMemoryCache
from langchain.globals import set_llm_cache
set_llm_cache(InMemoryCache())
# We'll make a temporary directory to avoid clutter
working_directory = TemporaryDirectory()




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
llm = RunnableChatBot(wrapper.set_bot(system_prompt="""You are a python expert and our task is to develop python code with the users input as guide. 
# Set up the LLM                                                   Only output the fullly implemented code.""", model_id=2))


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

                    

###from langchain.agents.self_ask_with_search.base import create_self_ask_with_search_agent


# Create an array that contains all the tools used by the agent

# Create the prompt template
tools = FileManagementToolkit(
    root_dir=str(working_directory.name),
    selected_tools=["read_file", "write_file", "list_directory"],
).get_tools() + [duckduckgo_tool, python_repl_tool, bash_shell_tool]



# Content of the prompt template
template = '''
Design a top level plan and create the needed folders and files, 
Populate the files with classes and methods, 
Implement all classes with OOP code conform autopep8 using the following tools:
{tools}

Do not use a tool if not required.
Here are instrutions on how they work.
{tools_string}

Task instrutions: 
{question}
'''
from typing import Callable, List
from langchain.agents.agent import AgentOutputParser
# For backwards compatibility
from langchain.tools.render import render_text_description,ToolsRenderer
from langchain_core.tools import BaseTool

rtdaa=render_text_description(tools)
rp(f"tools  string: {rtdaa}")
prompt_template = PromptTemplate.from_template(template, partial_variables={'tools': tools, 'tools_string': rtdaa})

react_prompt = hub.pull('hwchase17/react', api_key=os.getenv('HUGGINGFACEHUB_API_TOKEN'))
self_ask_prompt = hub.pull("hwchase17/self-ask-with-search", api_key=os.getenv('HUGGINGFACEHUB_API_TOKEN'))



agent_executor = AgentExecutor(
    mame='Joop',
    memory=ConversationBufferMemory(memory_key="chat_history", return_messages=False),
    agent=create_react_agent(llm, tools, react_prompt, AgentOutputParser, ToolsRenderer),
    tools = tools,
    verbose = False, # explain all reasoning steps
    return_intermediate_steps=True,
    handle_parsing_errors=True, # continue on error 
    max_iterations = 1 # try up to 10 times to find the best answer
)

# Ask your question (replace this with your question)
question = """
    Make a druglords remake game in pyqt6. 
    Real banking system,
    multiple locations to travel to and trade,
    NPC merchants need to be react agent driven.
    Make sure that the final result is a multi file project.
    Always make a new folder in the current root and write files only into that folder
    Besure to write all code to their files."""

output = agent_executor.invoke({'input': prompt_template.format(question=question)})


class DebateAgent:
    def __init__(self, name: str, tools: List[Tool]):
        self.name = name
        self.tools = tools
        self.llm = llm
        self.memory = ConversationBufferMemory(memory_key="chat_history")

    def debate(self, topic: str, context: str) -> str:
        # Implement the debate logic here
        prompt = f"You are {self.name}. Debate the topic: {topic}\nContext: {context}\nYour argument:"
        return self.llm(prompt)

class JuryAgent:
    def __init__(self):
        self.llm = llm
    
    def evaluate(self, topic: str, arguments: Dict[str, List[str]]) -> str:
        # Implement the evaluation logic here
        prompt = f"You are a jury. Evaluate the following arguments on the topic: {topic}\n"
        for agent, args in arguments.items():
            prompt += f"{agent}'s arguments:\n"
            for arg in args:
                prompt += f"- {arg}\n"
        prompt += "\nWho provided the most insightful arguments and why?"
        return self.llm(prompt)

class DebateApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Debate Arena")
        self.setGeometry(100, 100, 1200, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.topic_input = QTextEdit()
        self.topic_input.setPlaceholderText("Enter debate topic here...")
        self.layout.addWidget(self.topic_input)

        self.rounds_layout = QHBoxLayout()
        self.rounds_label = QLabel("Number of rounds:")
        self.rounds_input = QSpinBox()
        self.rounds_input.setRange(1, 10)
        self.rounds_input.setValue(3)
        self.rounds_layout.addWidget(self.rounds_label)
        self.rounds_layout.addWidget(self.rounds_input)
        self.layout.addLayout(self.rounds_layout)

        self.time_layout = QHBoxLayout()
        self.time_label = QLabel("Time limit (seconds):")
        self.time_input = QSpinBox()
        self.time_input.setRange(60, 600)
        self.time_input.setValue(300)
        self.time_layout.addWidget(self.time_label)
        self.time_layout.addWidget(self.time_input)
        self.layout.addLayout(self.time_layout)

        self.start_button = QPushButton("Start Debate")
        self.start_button.clicked.connect(self.start_debate)
        self.layout.addWidget(self.start_button)

        self.debate_output = QTextEdit()
        self.debate_output.setReadOnly(True)
        self.layout.addWidget(self.debate_output)

        # Initialize agents and jury
        self.llm = agent_executor
        self.agents = [
            DebateAgent("Agent 1", tools),
            DebateAgent("Agent 2", tools)
        ]
        self.jury = JuryAgent()

        self.current_round = 0
        self.max_rounds = 0
        self.time_limit = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.debate_round)

    def start_debate(self):
        self.debate_output.clear()
        self.current_round = 0
        self.max_rounds = self.rounds_input.value()
        self.time_limit = self.time_input.value() * 1000  # Convert to milliseconds
        topic = self.topic_input.toPlainText()

        if not topic:
            self.debate_output.append("Please enter a debate topic.")
            return

        self.debate_output.append(f"Debate Topic: {topic}\n")
        self.debate_output.append(f"Number of Rounds: {self.max_rounds}")
        self.debate_output.append(f"Time Limit: {self.time_limit / 1000} seconds\n")

        self.timer.start(self.time_limit)
        self.debate_round()

    def debate_round(self):
        if self.current_round >= self.max_rounds:
            self.end_debate()
            return

        self.current_round += 1
        self.debate_output.append(f"\n--- Round {self.current_round} ---\n")

        topic = self.topic_input.toPlainText()
        context = self.debate_output.toPlainText()

        for agent in self.agents:
            argument = agent.debate(topic, context)
            self.debate_output.append(f"{agent.name}: {argument}\n")

        if self.current_round == self.max_rounds:
            self.end_debate()

    def end_debate(self):
        self.timer.stop()
        self.debate_output.append("\n--- Debate Concluded ---\n")

        topic = self.topic_input.toPlainText()
        arguments = {agent.name: agent.memory.chat_memory.messages for agent in self.agents}
        winner = self.jury.evaluate(topic, arguments)

        self.debate_output.append(f"Jury's Decision:\n{winner}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    debate_app = DebateApp()
    debate_app.show()
    sys.exit(app.exec())
