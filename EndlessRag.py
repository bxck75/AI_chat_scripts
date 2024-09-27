import sys
import os
from langchain.chains.llm import LLMChain
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, QLineEdit, QFileDialog
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.prompts import StringPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
from hugging_chat_wrapper import HuggingChatWrapper,HuggingFaceEmbeddings
class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += f"Action: {action.tool}\nAction Input: {action.tool_input}\nObservation: {observation}\nThought: I now know the result of the action. I should consider my next step carefully.\n"
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

class CustomOutputParser:
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        
        action_match = re.search(r"Action: (.*?)[\n]*Action Input:[\s]*(.*)", llm_output, re.DOTALL)
        if not action_match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = action_match.group(1).strip()
        action_input = action_match.group(2)
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

class ChatbotThread(QThread):
    update_chat = pyqtSignal(str)

    def __init__(self, agent_chain):
        super().__init__()
        self.agent_chain = agent_chain
        self.query = ""

    def run(self):
        response = self.agent_chain.run(self.query)
        self.update_chat.emit(response)


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

class RAGChatbot(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RAG Chatbot")
        self.setGeometry(100, 100, 800, 600)
        self.wrapper = HuggingChatWrapper('EndlessRag')
        self.bot = RunnableChatBot(self.wrapper.chatbot)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.layout.addWidget(self.chat_history)

        self.input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.returnPressed.connect(self.send_message)
        self.input_layout.addWidget(self.input_field)

        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        self.input_layout.addWidget(self.send_button)

        self.layout.addLayout(self.input_layout)

        self.load_button = QPushButton("Load Document")
        self.load_button.clicked.connect(self.load_document)
        self.layout.addWidget(self.load_button)

        self.vectorstore = None
        self.agent_chain = None

        self.chatbot_thread = None

    def load_document(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Document")
        if file_path:
            loader = TextLoader(file_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(documents)

            embeddings = HuggingFaceEmbeddings()
            self.vectorstore = Chroma.from_documents(texts, embeddings)

            retriever = self.vectorstore.as_retriever()
            qa_chain = RetrievalQA.from_chain_type(llm=self.bot, chain_type="stuff", retriever=retriever)

            tools = [
                Tool(
                    name="RAG QA System",
                    func=qa_chain.run,
                    description="Useful for answering questions about the loaded document.",
                )
            ]

            template = """Answer the following questions as best you can. You have access to the following tools:

            {tools}

            Use the following format:

            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question

            Begin!

            Question: {input}
            {agent_scratchpad}"""

            prompt = CustomPromptTemplate(
                template=template,
                tools=tools,
                input_variables=["input", "intermediate_steps"]
            )

            output_parser = CustomOutputParser()

            llm_chain = LLMChain(llm=self.bot, prompt=prompt)
            agent = LLMSingleActionAgent(
                llm_chain=llm_chain,
                output_parser=output_parser,
                stop=["\nObservation:"],
                allowed_tools=[tool.name for tool in tools]
            )

            memory = ConversationBufferWindowMemory(k=5)

            self.agent_chain = AgentExecutor.from_agent_and_tools(
                agent=agent, tools=tools, verbose=True, memory=memory
            )

            self.chat_history.append("Document loaded and processed. You can now start chatting!")

    def send_message(self):
        user_input = self.input_field.text()
        self.input_field.clear()

        if not user_input:
            return

        self.chat_history.append(f"You: {user_input}")

        if self.agent_chain is None:
            self.chat_history.append("Chatbot: Please load a document first.")
            return

        self.chatbot_thread = ChatbotThread(self.agent_chain)
        self.chatbot_thread.update_chat.connect(self.update_chat_history)
        self.chatbot_thread.query = user_input
        self.chatbot_thread.start()

    def update_chat_history(self, response):
        self.chat_history.append(f"Chatbot: {response}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    chatbot = RAGChatbot()
    chatbot.show()
    sys.exit(app.exec())