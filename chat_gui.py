import sys
import os
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())  # Assumes you have a.env file in the same directory as your script
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
EMAIL = os.getenv('EMAIL')
PASSWD = os.getenv('PASSWD')
from PyQt6.QtWidgets import QApplication, QMainWindow, QTextEdit, QPushButton, QVBoxLayout, QWidget, QTabWidget
from PyQt6.QtCore import Qt
from hugchat import hugchat
from hugchat.login import Login
from langchain.chains import ConversationChain,SimpleSequentialChain
from langchain_core.prompts import ChatPromptTemplate



class ChatTab(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.layout.addWidget(self.chat_display)
        
        self.chat_input = QTextEdit()
        self.chat_input.setPlaceholderText("Type your message here...")
        self.layout.addWidget(self.chat_input)
        
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        self.layout.addWidget(self.send_button)

    def send_message(self):
        prompt = self.chat_input.toPlainText()
        if prompt.strip():
            self.chat_display.append(f"User: {prompt}")
            response = self.generate_response(prompt)
            self.chat_display.append(f"Assistant: {response}")
            self.chat_input.clear()

    def generate_response(self, question):
        email = EMAIL # Replace with actual email
        passwd = PASSWD # Replace with actual password
        prompt = ChatPromptTemplate.from_template(
            "Answer the user's question to the best of your ability. "
            'You must always output a JSON object with an "answer" key and a "followup_question" key. '
            "{question}"
        )
        sign = Login(email, passwd)
        cookies = sign.login()
        sign.save_cookies()
        chatbot = hugchat.ChatBot(cookies=cookies.get_dict())
        chain =  ConversationChain(llm=chatbot,prompt=prompt)
        #chain =  SimpleSequentialChain([chain])
        response = chain.invoke({input=question})
        print(response)
        return response

class ChatApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("HugChat - PyQt6 App")

        self.tab_widget = QTabWidget()
        self.chat_tab = ChatTab()

        self.tab_widget.addTab(self.chat_tab, "Chat")
        self.setCentralWidget(self.tab_widget)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChatApp()
    window.show()
    sys.exit(app.exec())