# filename: chat_tab_with_huggingface_and_huggingchat.py

import sys,os
from langchain.chains import ConversationChain
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QTabWidget, 
    QTextEdit, QLineEdit, QPushButton
)
from rich import print as rp
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,QGridLayout, 
                             QListWidget, QCheckBox, QTextEdit, QLineEdit, QPushButton, QTabWidget,QComboBox,QDialogButtonBox,
                             QSplitter, QListWidgetItem, QFileDialog, QLabel,QMessageBox,QDialog,QTableWidget, QTableWidgetItem,
                             QScrollArea,QDoubleSpinBox,QPlainTextEdit,QSpinBox,QDockWidget,QSizePolicy,QHeaderView,QInputDialog,QAbstractItemView)
from PyQt6 import QtWidgets
from PyQt6.QtGui import QIcon,QPalette, QColor, QPixmap, QTextCharFormat,QSyntaxHighlighter,QTextCursor,QFont
from PyQt6.QtCore import QSize, QDir, Qt, pyqtSlot, QObject, pyqtSignal, QRunnable, QThreadPool, QTimer, QUrl, pyqtSlot,QObject
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import QThreadPool, QObject, pyqtSignal, QRunnable, pyqtSlot
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.base import Runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from components.hugging_chat_wrapper import HuggingChatWrapper
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain_community.llms import HuggingFaceEndpoint  # Import for HuggingFace LLM
from components.hugging_chat_wrapper import HuggingChatWrapper
from hugchat.login import Login
from hugchat import hugchat

# Define Worker Signals
class WorkerSignals(QObject):
    result = pyqtSignal(dict)
    error = pyqtSignal(tuple)


# Define Worker for threading
class Worker(QRunnable):
    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        try:
            result = self.func(*self.args, **self.kwargs)
            self.signals.result.emit(result)
        except Exception as e:
            self.signals.error.emit((str(e),))

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



class ChatTab(QWidget):
    def __init__(self, thread_logger, hf_api_token):
        super().__init__()

        self.thread_logger = thread_logger
        self.thread_logger.thread_pool = QThreadPool()
        self.wrapper = HuggingChatWrapper(project_name="chainbot")
        self.hf_api_token = hf_api_token
        self.chain = self.generate_chain(self.wrapper.email,self.wrapper.password)
        # Setup Chat Layout
        self.layout = QVBoxLayout(self)

        # Chat Display Area (QTextEdit)
        self.chat_display = QTextEdit(self)
        self.chat_display.setReadOnly(True)
        self.layout.addWidget(self.chat_display)

        # Input Field (QLineEdit)
        self.chat_input = QLineEdit(self)
        self.layout.addWidget(self.chat_input)

        # Send Button
        self.send_button = QPushButton('Send', self)
        self.layout.addWidget(self.send_button)

        # Langchain setup
        #self.setup_langchain()
    
        # Button Click Event
        self.send_button.clicked.connect(self.send_message)

    def setup_langchain(self):
        # Change to whichever LLM you want to use (HuggingChatWrapper or HuggingFaceEndpoint)
        repo_id = "HuggingFaceH4/zephyr-7b-beta"
                # Setup Prompt Template
        self.prompt = ChatPromptTemplate.from_template(
            "Answer the user's question to the best of your ability. "
            'You must always output a JSON object with an "answer" key and a "followup_question" key. '
            "{question}"
        )
        self.wrapper.set_bot(system_prompt=self.prompt)
        self.bot = self.wrapper.chatbot
        chain = ConversationChain(llm=self.bot)
        self.llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            max_length=128,
            temperature=0.5,
            huggingfacehub_api_token=self.hf_api_token
        )


        # Chains
        self.bot_chain = self.prompt | chain 
        #self.llm_chain = self.prompt | self.llm | SimpleJsonOutputParser()

        # Use HuggingChatWrapper for testing by default
        self.chain = self.bot_chain  # You can switch to self.llm_chain if desired
# Function for generating LLM response
    
    def generate_chain(prompt, email, passwd):
        # Hugging Face Login
        sign = Login(email, passwd)
        cookies = sign.login()
        sign.save_cookies()
        # Create ChatBot                        
        chatbot = hugchat.ChatBot(cookies=cookies.get_dict())
        run_llm=RunnableChatBot(chatbot=chatbot)
        chain = ConversationChain(llm=run_llm)
        #response = chain.run(input=prompt)
        return chain
    
    def send_message(self):
        user_message = self.chat_input.text()
        if user_message.strip():
            self.chat_display.append(f"You: {user_message}")
            self.chat_input.clear()
            self.thread_logger.logger.info(f"User sent message: {user_message}")

            try:

                # Pass the plain user message as a string, not as a dictionary
                worker = Worker(self.chain.invoke, user_message)  # Input is a simple string
                worker.signals.result.connect(self.handle_response)
                worker.signals.error.connect(self.handle_error)
                self.thread_logger.logger.info("Worker for handling user message created, starting the worker...")
                self.thread_logger.thread_pool.start(worker)
                self.thread_logger.logger.info("Worker for user message started successfully")
            except Exception as e:
                self.thread_logger.logger.error(f"Failed to start worker for user message: {str(e)}")

    def handle_response(self, response):
        if response:
            answer = response.get('answer')
            followup_question = response.get('followup_question')

            if answer:
                self.chat_display.append(f"Assistant: {answer}")
            if followup_question:
                self.chat_display.append(f"Follow-up: {followup_question}")

        # Scroll to the bottom of the chat display
        self.chat_display.verticalScrollBar().setValue(
            self.chat_display.verticalScrollBar().maximum()
        )

    def handle_error(self, error):
        error_message = error[0]
        self.chat_display.append(f"Error: {error_message}")
        self.thread_logger.logger.error(f"Error in worker thread: {error_message}")


class MainWindow(QMainWindow):
    def __init__(self, thread_logger, hf_api_token):
        super().__init__()
        self.setWindowTitle('Chat with AI')

        # Setup tabs
        self.tabs = QTabWidget(self)
        self.setCentralWidget(self.tabs)

        # Add ChatTab with thread_logger and Hugging Face API token
        self.chat_tab = ChatTab(thread_logger, hf_api_token)
        self.tabs.addTab(self.chat_tab, "Chat")

class ThreadLogger(QObject):
    log_updated = pyqtSignal(str)
    thread_count_updated = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.thread_pool = QThreadPool.globalInstance()
        self.thread_pool.setMaxThreadCount(8)
        self.log = []
        self.setup_logging()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_thread_count)
        self.timer.start(1000)  # Update every second

    def setup_logging(self):
        # Set up the logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Create a file handler
        file_handler = logging.FileHandler('chatbots.log')
        file_handler.setLevel(logging.INFO)

        # Create a formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add the handler to the logger
        self.logger.addHandler(file_handler)

        # Create a custom handler to emit signals
        custom_handler = logging.Handler()
        custom_handler.emit = self.log_handler
        custom_handler.setFormatter(formatter)
        self.logger.addHandler(custom_handler)

    def log_handler(self, record):
        log_entry = self.logger.handlers[0].formatter.format(record)  # Format using the first handler
        self.log.append(log_entry)
        self.log_updated.emit(log_entry)

    def update_thread_count(self):
        active_thread_count = self.thread_pool.activeThreadCount()
        self.thread_count_updated.emit(active_thread_count)


# Run the application
if __name__ == '__main__':
    import os
    from dotenv import load_dotenv, find_dotenv
    import logging,warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    logging.basicConfig(filename='chatbots.log', level=logging.DEBUG)
    warnings.filterwarnings("ignore", category=DeprecationWarning, message="sipPyTypeDict")
    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--disable-gpu"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    load_dotenv(find_dotenv())  # Assumes you have a.env file in the same directory as your script
    
    app = QApplication(sys.argv)


    # Initialize the main window with a logger and the API token
    
    main_window = MainWindow(ThreadLogger(), os.getenv("HUGGINGFACEHUB_API_TOKEN"))
    main_window.show()

    sys.exit(app.exec())