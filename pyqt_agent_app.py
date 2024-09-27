# filename: pyqt_agent_app.py
import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QTabWidget, 
    QTextEdit, QLineEdit, QPushButton, QHBoxLayout
)
from PyQt6.QtCore import Qt
import os,sys,re
from rich import print as rp
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QTabWidget, 
    QTextEdit, QLineEdit, QPushButton
)
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
from langchain_core.runnables.base import Runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain_core.output_parsers.string import StrOutputParser
from components.hugging_chat_wrapper import HuggingChatWrapper
wrapper = HuggingChatWrapper(project_name='AgenticDashboard')

from prompt_toolkit import prompt

if __name__ == '__main__':
    answer = prompt('Give me some input: ')
    print('You said: %s' % answer)


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
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain_community.tools import ShellTool

from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory


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

    llm=RunnableChatBot(wrapper.set_bot(system_prompt="""You are a python expert and our task is to develop code with the users input as guide. 
                                        Only output the fullly implemented code.""", model_id=1))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AgentApp()
    window.show()
    sys.exit(app.exec())