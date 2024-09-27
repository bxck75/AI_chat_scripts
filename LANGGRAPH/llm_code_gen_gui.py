import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, QLineEdit
from PyQt6.QtGui import QTextCursor
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from langchain_core.messages import HumanMessage, AIMessage
from rich import print as rp

# Import your existing code
from llm_code_gen_huggingface_zephyr import LLMCodeGen  # Ensure this import works

class ChatbotWorker(QThread):
    output_ready = pyqtSignal(str)
    
    def __init__(self, code_gen):
        super().__init__()
        self.code_gen = code_gen
        self.question = ""

    def run(self):
        result = self.code_gen.run_question(self.question)
        if result and 'generation' in result:
            final_code = result['generation']
            output = f"Imports:\n{final_code.imports}\n\nCode:\n{final_code.code}"
            self.output_ready.emit(output)

class ChatbotGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LLM Code Generator Chatbot")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.layout.addWidget(self.chat_display)

        self.input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.returnPressed.connect(self.send_message)
        self.input_layout.addWidget(self.input_field)

        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        self.input_layout.addWidget(self.send_button)

        self.layout.addLayout(self.input_layout)

        self.code_gen = LLMCodeGen()
        self.worker = ChatbotWorker(self.code_gen)
        self.worker.output_ready.connect(self.display_response)

    def send_message(self):
        user_input = self.input_field.text()
        self.input_field.clear()
        self.display_message(f"User: {user_input}", "blue")
        
        self.worker.question = user_input
        self.worker.start()

    def display_message(self, message, color):
        # Use the actual enum value for color
        if color == "blue":
            self.chat_display.setTextColor(Qt.GlobalColor.blue)
        elif color == "green":
            self.chat_display.setTextColor(Qt.GlobalColor.green)
        elif color == "black":
            self.chat_display.setTextColor(Qt.GlobalColor.black)
        else:
            self.chat_display.setTextColor(Qt.GlobalColor.black)  # Default to black if color is unknown
        
        self.chat_display.append(message)
        self.chat_display.setTextColor(Qt.GlobalColor.black)  # Reset to black after appending
        self.chat_display.moveCursor(QTextCursor.MoveOperation.End)

    def display_response(self, response):
        self.display_message("Assistant: Here's the generated code:", "green")
        self.display_message(response, "black")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChatbotGUI()
    window.show()
    sys.exit(app.exec())