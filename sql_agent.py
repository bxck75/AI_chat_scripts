import sys
import sqlite3
from PyQt6.QtWidgets import QApplication, QMainWindow, QTextEdit, QLineEdit, QPushButton, QVBoxLayout, QWidget
from langchain import SQLDatabase
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from hugging_chat_wrapper import HuggingFaceEmbeddings,HuggingChatWrapper
class ChatbotWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SQL Agent Chatbot")
        self.setGeometry(100, 100, 500, 500)

        # Create widgets
        self.chat_area = QTextEdit(self)
        self.chat_area.setReadOnly(True)

        self.input_field = QLineEdit(self)
        self.send_button = QPushButton("Send", self)

        # Create layout
        layout = QVBoxLayout()
        layout.addWidget(self.chat_area)
        layout.addWidget(self.input_field)
        layout.addWidget(self.send_button)

        # Create central widget and set layout
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Connect button click to send_message method
        self.send_button.clicked.connect(self.send_message)

        # Initialize SQL Agent
        self.initialize_sql_agent()

    def initialize_sql_agent(self):
        # Initialize your SQL Database
        self.db = SQLDatabase.from_uri("sqlite:///your_database.db")

        tools = [
            Tool(
                name="SQL",
                func=self.db.run,
                description="useful for when you need to answer questions about data stored in a SQL database. You should ask targeted questions",
            ),
        ]

        # Initialize the SQL Agent with error handling
        self.sql_agent = initialize_agent(
            tools,
            ChatOpenAI(temperature=0),
            agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=self.handle_parsing_error,
        )

    def handle_parsing_error(self, error):
        return f"Error: {str(error)[:100]}... Please try rephrasing your question."

    def send_message(self):
        user_input = self.input_field.text()
        self.chat_area.append(f"You: {user_input}")
        self.input_field.clear()

        # Get response from SQL Agent
        try:
            response = self.sql_agent.run(user_input)
            self.chat_area.append(f"Bot: {response}")
        except Exception as e:
            self.chat_area.append(f"Bot: An error occurred: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChatbotWindow()
    window.show()
    sys.exit(app.exec())