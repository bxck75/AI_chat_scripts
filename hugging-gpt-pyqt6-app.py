import sys
from typing import List
from PyQt6.QtWidgets import (QApplication, QMainWindow, QTextEdit, QPushButton, 
                             QVBoxLayout, QWidget, QHBoxLayout, QLabel, QLineEdit)
from PyQt6.QtCore import QThread, pyqtSignal

from langchain.base_language import BaseLanguageModel
from langchain_core.tools import BaseTool
from langchain_experimental.autonomous_agents.hugginggpt.repsonse_generator import load_response_generator
from langchain_experimental.autonomous_agents.hugginggpt.task_executor import TaskExecutor
from langchain_experimental.autonomous_agents.hugginggpt.task_planner import load_chat_planner
from langchain.llms import HuggingFaceEndpoint

class HuggingGPT:
    def __init__(self, llm: BaseLanguageModel, tools: List[BaseTool]):
        self.llm = llm
        self.tools = tools
        self.chat_planner = load_chat_planner(llm)
        self.response_generator = load_response_generator(llm)
        self.task_executor: TaskExecutor

    def run(self, input: str) -> dict:
        plan = self.chat_planner.plan(inputs={"input": input, "hf_tools": self.tools})
        self.task_executor = TaskExecutor(plan)
        execution_result = self.task_executor.run()
        response = self.response_generator.generate({"task_execution": self.task_executor})
        return {
            "plan": plan,
            "execution_result": execution_result,
            "response": response
        }

class WorkerThread(QThread):
    finished = pyqtSignal(dict)

    def __init__(self, hugging_gpt, input_text):
        super().__init__()
        self.hugging_gpt = hugging_gpt
        self.input_text = input_text

    def run(self):
        result = self.hugging_gpt.run(self.input_text)
        self.finished.emit(result)

class MainWindow(QMainWindow):
    def __init__(self, hugging_gpt):
        super().__init__()
        self.hugging_gpt = hugging_gpt
        self.setWindowTitle("HuggingGPT PyQt6 App")
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        # Input section
        input_layout = QHBoxLayout()
        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText("Enter your input here...")
        input_layout.addWidget(self.input_text)
        self.submit_button = QPushButton("Submit")
        self.submit_button.clicked.connect(self.on_submit)
        input_layout.addWidget(self.submit_button)
        layout.addLayout(input_layout)

        # Debugging section
        debug_layout = QHBoxLayout()
        self.plan_text = QTextEdit()
        self.plan_text.setReadOnly(True)
        self.plan_text.setPlaceholderText("Plan will be displayed here...")
        debug_layout.addWidget(self.plan_text)
        self.execution_text = QTextEdit()
        self.execution_text.setReadOnly(True)
        self.execution_text.setPlaceholderText("Execution results will be displayed here...")
        debug_layout.addWidget(self.execution_text)
        layout.addLayout(debug_layout)

        # Output section
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setPlaceholderText("Response will be displayed here...")
        layout.addWidget(self.output_text)

        # Tool management section
        tool_layout = QHBoxLayout()
        self.tool_input = QLineEdit()
        self.tool_input.setPlaceholderText("Enter tool name")
        tool_layout.addWidget(self.tool_input)
        self.add_tool_button = QPushButton("Add Tool")
        self.add_tool_button.clicked.connect(self.add_tool)
        tool_layout.addWidget(self.add_tool_button)
        layout.addLayout(tool_layout)

        # Prompt customization section
        prompt_layout = QHBoxLayout()
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText("Enter custom prompt here...")
        prompt_layout.addWidget(self.prompt_input)
        self.set_prompt_button = QPushButton("Set Custom Prompt")
        self.set_prompt_button.clicked.connect(self.set_custom_prompt)
        prompt_layout.addWidget(self.set_prompt_button)
        layout.addLayout(prompt_layout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def on_submit(self):
        input_text = self.input_text.toPlainText()
        self.output_text.setPlainText("Processing...")
        self.submit_button.setEnabled(False)

        self.worker_thread = WorkerThread(self.hugging_gpt, input_text)
        self.worker_thread.finished.connect(self.on_result_ready)
        self.worker_thread.start()

    def on_result_ready(self, result):
        self.plan_text.setPlainText(str(result['plan']))
        self.execution_text.setPlainText(str(result['execution_result']))
        self.output_text.setPlainText(result['response'])
        self.submit_button.setEnabled(True)

    def add_tool(self):
        tool_name = self.tool_input.text()
        # Here you would implement the logic to add a new tool
        # For demonstration, we'll just print the tool name
        print(f"Adding tool: {tool_name}")
        self.tool_input.clear()

    def set_custom_prompt(self):
        custom_prompt = self.prompt_input.toPlainText()
        # Here you would implement the logic to set a custom prompt
        # For demonstration, we'll just print the custom prompt
        print(f"Setting custom prompt: {custom_prompt}")
        self.prompt_input.clear()

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv,find_dotenv
    load_dotenv(find_dotenv())  # Assumes you have a.env file in the same directory as your script
    HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
    repo_id =  "HuggingFaceH4/zephyr-7b-beta"#"mistralai/Mistral-7B-Instruct-v0.2"

    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        max_length=128,
        temperature=0.5,
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    )

    tools = []  # Add your tools here

    hugging_gpt = HuggingGPT(llm, tools)

    app = QApplication(sys.argv)
    window = MainWindow(hugging_gpt)
    window.show()
    sys.exit(app.exec())
