# filename: chat_tab_with_huggingface_endpoint.py

import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QTabWidget, 
    QTextEdit, QLineEdit, QPushButton
)
from PyQt6.QtCore import QThreadPool, QObject, pyqtSignal, QRunnable, pyqtSlot
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain_huggingface import HuggingFaceEndpoint
from components.hugging_chat_wrapper import HuggingChatWrapper
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


class ChatTab(QWidget):
    def __init__(self, thread_logger, hf_api_token):
        super().__init__()

        self.thread_logger = thread_logger
        self.thread_logger.thread_pool = QThreadPool()
        self.wrapper = HuggingChatWrapper(project_name="chainbot")
        self.hf_api_token = hf_api_token

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
        self.setup_langchain()

        # Button Click Event
        self.send_button.clicked.connect(self.send_message)

    def setup_langchain(self):
        repo_id = "HuggingFaceH4/zephyr-7b-beta"#"mistralai/Mistral-7B-Instruct-v0.2"
        self.bot = self.wrapper.chatbot.chat
        # Setup the HuggingFaceEndpoint LLM with the repo_id
        self.llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            max_length=128,
            temperature=0.5,
            huggingfacehub_api_token=self.hf_api_token  # Pass the token
        )

        # Setup Prompt Template
        self.prompt = ChatPromptTemplate.from_template(
            "Answer the user's question to the best of your ability. "
            'You must always output a JSON object with an "answer" key and a "followup_question" key. '
            "{question}"
        )

        # Setup Chain
        self.bot_chain = self.prompt | self.bot | SimpleJsonOutputParser()
        self.llm_chain = self.prompt | self.llm | SimpleJsonOutputParser()
        self.chain = self.bot_chain # self.llm_chain
        
    def send_message(self):
        user_message = self.chat_input.text()
        if user_message.strip():
            self.chat_display.append(f"You: {user_message}")
            self.chat_input.clear()
            self.thread_logger.logger.info(f"User sent message: {user_message}")

            try:
                # Pass the chain invocation logic to the worker
                worker = Worker(self.chain.invoke, {"question": user_message})
                worker.signals.result.connect(self.handle_response)
                worker.signals.error.connect(self.handle_error)
                self.thread_logger.logger.info("Worker for handling user message created, starting the worker...")
                self.thread_logger.thread_pool.start(worker)
                self.thread_logger.logger.info("Worker for user message started successfully")
            except Exception as e:
                self.thread_logger.logger.error(f"Failed to start worker for user message: {str(e)}")

    def handle_response(self, response):
        # The response is expected to be a dict with 'answer' and 'followup_question'
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


# Mock logger for demonstration
class MockLogger:
    def __init__(self):
        self.thread_pool = None
        self.logger = self

    def info(self, message):
        print(f"[INFO] {message}")

    def error(self, message):
        print(f"[ERROR] {message}")


# Run the application
if __name__ == '__main__':
    import os
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())  # Assumes you have a.env file in the same directory as your script
    app = QApplication(sys.argv)

    # Set your Hugging Face API token
    HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    # Initialize the main window with a mock logger and the API token
    thread_logger = MockLogger()
    main_window = MainWindow(thread_logger, HUGGINGFACEHUB_API_TOKEN)
    main_window.show()

    sys.exit(app.exec())