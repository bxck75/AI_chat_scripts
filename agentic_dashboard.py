# filename: main.py
import sys
import json
import os
import re
from time import sleep
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from rich import print as rp
from gradio_client import Client

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,QGridLayout, 
                             QListWidget, QCheckBox, QTextEdit, QLineEdit, QPushButton, QTabWidget,QComboBox,QDialogButtonBox,
                             QSplitter, QListWidgetItem, QFileDialog, QLabel,QMessageBox,QDialog,QTableWidget, QTableWidgetItem,
                             QScrollArea,QDoubleSpinBox,QPlainTextEdit,QSpinBox,QDockWidget,QSizePolicy,QHeaderView,QInputDialog,QAbstractItemView)
from PyQt6 import QtWidgets
from PyQt6.QtGui import QIcon,QPalette, QColor, QPixmap, QTextCharFormat,QSyntaxHighlighter,QTextCursor,QFont
from PyQt6.QtCore import QSize, QDir, Qt, pyqtSlot, QObject, pyqtSignal, QRunnable, QThreadPool, QTimer, QUrl, pyqtSlot,QObject
from PyQt6.QtWebEngineWidgets import QWebEngineView
import warnings,logging
from components.hugging_chat_wrapper import HuggingChatWrapper,ConversationManager,PromptFactory
#from components.new_hug_wrapper import HugChatWrapper,ConversationManager
from components.syntax_highlighter import SyntaxHighlighter

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(filename='chatbots.log', level=logging.DEBUG)
warnings.filterwarnings("ignore", category=DeprecationWarning, message="sipPyTypeDict")
os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--disable-gpu"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception as e:
            self.signals.error.emit((str(e),))
        else:
            self.signals.result.emit(result)

    # TODO: prepair for more dynamic,generic handling of: 
    #     - diffuser/transformer API's from huggingface
    #     - other image processing APIs(vision,img2img,lipsync,etc)
    
class WorkerSignals(QObject):
    result = pyqtSignal(object)
    error = pyqtSignal(tuple)


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

class TopBar(QWidget):
    def __init__(self, thread_logger):
        super().__init__()
        self.thread_logger = thread_logger
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        # Top section (always visible)
        self.top_section = QHBoxLayout()
        self.thread_count_label = QLabel("Active Threads: 0")
        self.toggle_button = QPushButton("▼")
        self.toggle_button.setStyleSheet("""
            QPushButton {
                background-color: hsl(18, 50.4%, 47.5%);
                color: hsl(0, 0%, 100%);
                border: none;
                border-radius: 5px;
                padding: 5px 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: hsl(18, 56.8%, 43.5%);
            }
        """)
        self.toggle_button.clicked.connect(self.toggle_details)
        self.top_section.addWidget(self.thread_count_label)
        self.top_section.addStretch()
        self.top_section.addWidget(self.toggle_button)
        layout.addLayout(self.top_section)

        # Detailed section (collapsible)
        self.details_widget = QWidget()
        details_layout = QVBoxLayout()
        self.log_list = QListWidget()
        details_layout.addWidget(self.log_list)
        self.details_widget.setLayout(details_layout)
        self.details_widget.setVisible(False)
        layout.addWidget(self.details_widget)

        self.setLayout(layout)

        # Connect signals
        self.thread_logger.log_updated.connect(self.add_log_entry)
        self.thread_logger.thread_count_updated.connect(self.update_thread_count)

    def toggle_details(self):
        is_expanded = self.details_widget.isVisible()
        self.details_widget.setVisible(not is_expanded)
        self.toggle_button.setText("▲" if not is_expanded else "▼")
        
        # Calculate the height of the top section
        top_section_height = self.top_section.sizeHint().height() * 2
        QWIDGETSIZE_MAX = 16777215
        # Set a fixed height when expanded (4 lines of content)
        if is_expanded:
            # You can adjust the pixel value to fit 4 lines
            expanded_height = 2 * self.fontMetrics().height()
            self.setMaximumHeight(expanded_height + top_section_height)
        else:
            # When collapsing, remove the maximum height restriction
            self.setMaximumHeight(QWIDGETSIZE_MAX)
        

        #rp(f"TopBar height:{top_section_height}")
        #if is_expanded:
            # When collapsing, set the maximum height to include only the top section
        #    self.setMaximumHeight(top_section_height)
        #else:
            # When expanding, remove the maximum height restriction
        #    self.setMaximumHeight(QWIDGETSIZE_MAX)  # QWIDGETSIZE_MAX

    def add_log_entry(self, entry):
        self.log_list.addItem(QListWidgetItem(entry))
        self.log_list.scrollToBottom()

    def update_thread_count(self, count):
        self.thread_count_label.setText(f"Active Threads: {count}")

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

class ImageGenerator:
    def __init__(self, parent, thread_logger, wrapper):
        self.thread_logger = thread_logger
        self.parent = parent
        self.wrapper = wrapper
        self.client = None
        self.booster_prompt = "4k,awardwinning,mindblowing, ultra detailed..."
        self.default_prompt = "Decaying portrait of a skeleton..."
    
    def gen_two(self, prompt, is_negative=None, steps=None, cfg_scale=None, sampler=None, seed=None, strength=None, use_dev=False, new_img_path=None):
        if not self.client:
           self.thread_logger.logger.info(f"Client connected!")
           self.client = Client("K00B404/FLUX.1-Dev-Serverless-darn")

        worker = Worker(self.client.predict,
                prompt=f"{prompt} ",
                is_negative=is_negative or "(deformed, distorted, disfigured), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, misspellings, typos",
                steps=steps or 56,
                cfg_scale=cfg_scale or 7,
                sampler=sampler or "DPM++ 2M Karras",
                seed=seed or -1,
                strength=strength or 0.7,
                huggingface_api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
                use_dev=use_dev,
                api_name="/query"
        )
        
        # Log the parameters being used for image generation
        self.thread_logger.logger.info(f"Parameters - is_negative: {is_negative}, steps: {steps}, "
                                    f"cfg_scale: {cfg_scale}, sampler: {sampler}, seed: {seed}, "
                                    f"strength: {strength}")

        # worker = Worker(self.gen_to, prompt, is_negative, steps, cfg_scale, sampler, seed, strength)
        worker.signals.result.connect(self.handle_image_result, new_img_path=new_img_path)
        worker.signals.error.connect(self.handle_image_error)

        self.thread_logger.logger.info(f"Worker requesting image with for: \n\t{prompt}")
        self.thread_logger.thread_pool.start(worker)
        #rp(dir(worker))
        self.thread_logger.logger.info("Worker working...")
        #print(f'{img_path}:{seed}')

    def handle_image_result(self, result, new_img_path=None):
        self.thread_logger.logger.info(f"Worker Finished!")
        image_path, seed_used = result

        self.thread_logger.logger.info(f"Handling results. \nImage path: \n\t{image_path}, \nSeed used: \n\t{seed_used}")
        self.parent.display_generated_image(image_path, seed_used)

    def handle_image_error(self, error):
        error_message = error[0]
        self.thread_logger.logger.error(f"Error generating image: {error_message}")
        self.parent.chat_display.append(f"Error generating image: {error_message}")

    """ def generate_image(self, prompt, is_negative=None, steps=10, cfg_scale=3,
                   sampler="DPM++ 2M Karras", seed=-1, strength=0.7,
                   huggingface_api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN")):
        self.thread_logger.logger.info(f"Generating image with prompt: {prompt}")
        try:
            result = self.client.predict(
                prompt + self.booster_prompt,
                is_negative or "(deformed, distorted, disfigured), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, misspellings, typos",
                steps,
                cfg_scale,
                sampler,
                seed,
                strength,
                huggingface_api_key,
                "/query"
            )
            self.thread_logger.logger.info(f"Image generation completed successfully. Result: {result}")
            return result
        except Exception as e:
            self.thread_logger.logger.error(f"Error occurred during image generation: {str(e)}")
            raise """

class NewArtifactTab(QWidget):
    
    def __init__(self, parent, artifact, chatbot_instance):
        super().__init__()
        self.parent = parent
        self.filename = artifact.get('source')
        self.artifact_type = artifact.get('type')
        self.chatbot_instance = chatbot_instance.chat
        self.init_ui(artifact.get('content'))
    
    def init_values(self):
        self.max_class_length = 750 # lines per class before separation.
        self.iterations = self.iterations_input.value()
        self.code_content = self.code_edit.toPlainText()
        self.rules="""
            ALL response must be in encapsulated 'artifacts', 
            defined by the following file types:
                <type>          :       <encapsulation>
                "py"            :    "```python <content>```"
                "yaml"          :    "```yaml <content>```"
                "txt"           :    "```text <content>```"
                "png"           :    "```image_description <content>```"      
                "txt"           :    "```chat <content>```" 
                "mmd"           :    "```mermaid <content>```" 
            While answering think step-by-step and justify your answer.
            Always start the content of the artifact with # filename: <filename>.<type>
            """
        self.context="""No Context Found!"""
        self.iter_improver_prompt=f"""
            Provide these encapsulated artifacts:
                improved code 
                usage manual 
                mermaid flowchart

            Consider the following aspects:
                Code efficiency
                Enhancement
                Readability
                Best practices
                Potential bugs

            Here is some context to the query:
                {context}
            Now is your time to Shine! 
            Improve the following code:
                ```python
                {code_content}
                ```
            """
            # we use this to build the bot then repeat every x steps to force the rules 
        self.init_improver_prompt=f"""
            You are an expert Python developer and master code improver . 
            Your task is to improve scripts input by users.
            Response must always be in correctly encapsulated artifacts.
            Here are the rules:
                {rules}
            
            Provide these encapsulated artifacts:
                improved code 
                usage manual 
                mermaid flowchart

            Consider the following aspects:
                Code efficiency
                Enhancement
                Readability
                Best practices
                Potential bugs


            Here is some context to the query:
                {context}
            Now is your time to Shine! 
            Improve the following code:
                ```python
                {code_content}
                ```
            """ # we use this every step of the chat except when x

    def init_ui(self, content):
        layout = QVBoxLayout()
        # string content
        if isinstance(content, str):
            self.code_edit = QTextEdit()
            self.code_edit.setStyleSheet("""
                QPlainTextEdit {
                    background-color: hsl(60, 3.3%, 17.8%);
                    color: hsl(60, 6.7%, 97.1%);
                    border: 1px solid hsl(50, 5.9%, 40%);
                    border-radius: 5px;
                    padding: 5px;
                    font-family: 'Courier New', monospace;
                }
                """)
            if self.artifact_type == "mermaid":
                # Mermaid artifact handling (unchanged)
                self.web_view = QWebEngineView()
                mermaid_code = self.fix_mermaid_code(content)
                self.code_edit.setText(mermaid_code)
                clean_mermaid = self.load_mermaid_diagram(mermaid_code)
                self.web_view.setHtml(clean_mermaid)
                layout.addWidget(self.web_view)



            layout.addWidget(self.code_edit)

            button_layout = QHBoxLayout()
            self.save_button = QPushButton("Save")
            self.save_button.clicked.connect(self.save_artifact)
            self.query_button = QPushButton("Query")
            self.query_button.clicked.connect(self.query_artifact)
            self.close_button = QPushButton("X")
            self.close_button.clicked.connect(self.close_tab)
            button_layout.addWidget(self.close_button)
            button_layout.addWidget(self.save_button)
            button_layout.addWidget(self.query_button)
            # if python
            if self.artifact_type == "python":

                self.code_edit.setText(content)
                self.highlighter = SyntaxHighlighter(self.code_edit.document())

                self.improve_button = QPushButton("Improve")
                self.improve_button.clicked.connect(self.improve_code)
                self.iterations_input = QSpinBox()
                self.iterations_input.setMinimum(1)
                self.iterations_input.setMaximum(10)
                self.iterations_input.setValue(1)
                button_layout.addWidget(self.improve_button)
                button_layout.addWidget(self.iterations_input)

            layout.addLayout(button_layout)

            self.user_query_input = QTextEdit()
            self.user_query_input.setPlaceholderText("Enter your query about this code...")
            self.user_query_input.setStyleSheet("""
                QTextEdit {
                    background-color: hsl(60, 3.3%, 17.8%);
                    color: hsl(60, 6.7%, 97.1%);
                    border: 1px solid hsl(50, 5.9%, 40%);
                    border-radius: 5px;
                    padding: 5px;
                    font-family: 'Courier New', monospace;
                }
            """)
            layout.addWidget(self.user_query_input)
        
        # not str but QWidget
        elif isinstance(content, QLabel):
            # Image artifact (unchanged)
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setWidget(content)
            layout.addWidget(scroll_area)

            self.save_button = QPushButton("Save Image")
            self.save_button.clicked.connect(self.save_image)
            layout.addWidget(self.save_button)

        self.setLayout(layout)

    def close_tab(self):
        index = self.parent.artifact_tabs.indexOf(self)
        if index != -1:
            self.parent.artifact_tabs.removeTab(index)

    def improve_code(self):
        if self.artifact_type != "python" :
            return

        try: # request a bot from the manager
            """ rp(f"SELF:{dir(self)}")
            improver_chatbot = ConversationManager(
                os.getenv("EMAIL"),
                os.getenv("PASSWD"),
                self.parent.wrapper.cookie_folder,
                system_prompt=self.init_improver_prompt
            ).chatbot
            remote_conversations = improver_chatbot.get_remote_conversations(False) """
            for _ in range(self.iterations):
                #response = self.parent.send_message(code_content)
                self.code_edit.setText(self.extract_code_from_response(self.parent.send_message(self.code_edit.toPlainText())))
                #code_content = self.extract_code_from_response(response)
                # TODO pytest or other debugging
                #self.code_edit.setText(code_content)
            
            QMessageBox.information(self, "Code Improved", f"Code has been improved after {self.iterations} iterations.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to improve code: {str(e)}")

    def extract_code_from_response(self, response):
        # Extract code block from the chatbot's response
        code_block = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
        if code_block:
            return code_block.group(1)
        return response  # Return the full response if no code block is found

    def load_mermaid_diagram(self, mermaid_code):
        
        # Create HTML content with Mermaid diagram
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <script type="module">
                import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
                mermaid.initialize({ startOnLoad: true });
            </script>
        </head>
        <body>
            <div class="mermaid">
                ###CODE###
            </div>
        </body>
        </html>
        """
        html_string = html_content.replace("###CODE###", mermaid_code)
        return html_string

    def fix_mermaid_code(self, mermaid_code):
        no_label_pattern = re.compile(r'(\w+) -->\|(.+?)\|> (\w+)\[(.*?)\]')
        no_label_pattern_c = re.compile(r'(\w+)\[(.*?)\] -->\|(.+?)\|> (\w+)')
        no_label_pattern_b = re.compile(r'(\w+) -->\|(.+?)\|> (\w+)')
        no_label_pattern_a = re.compile(r'(\w+) -->\|(.+?)\| (\w+)\[(.*?)\]')
        label_pattern = re.compile(r'(\w+)\[(.*?)\] -->\|(.+?)\|> (\w+)\[(.*?)\]')
        label_pattern_a = re.compile(r'(\w+)\[(.*?)\] -->\|(.+?)\| (\w+)\[(.*?)\]')

        def no_label_replace_match(match):
            node1, action, node2, label2 = match.groups()
            intermediate_node = f"{node1}{action[0]}[{action}]"
            return f"{node1} --> {intermediate_node} --> {node2}[{label2}]"
        def replace_match_no_label(match):
            node1, label1, action, node2 = match.groups()
            intermediate_node = f"{node1}{action[0]}[{action}]"
            return f"{node1}[{label1}] --> {intermediate_node} --> {node2}"
        def no_label_replace_match_no_label(match):
            node1, action, node2 = match.groups()
            intermediate_node = f"{node1}{action[0]}[{action}]"
            return f"{node1} --> {intermediate_node} --> {node2}"
        def replace_match_label(match):
            node1, label1, action, node2, label2 = match.groups()
            intermediate_node = f"{node1}{action[0]}[{action}]"
            return f"{node1}[{label1}] --> {intermediate_node} --> {node2}[{label2}]"

        fixed_code = []

        for line in mermaid_code.split(".mmd\n").pop().split("\n"):
            line = line.replace("{","[").replace("}","]")
            if label_pattern.search(line):
                fixed_line = label_pattern.sub(replace_match_label, line)
                fixed_code.append(fixed_line)
            elif label_pattern_a.search(line):
                fixed_line = label_pattern_a.sub(replace_match_label, line)
                fixed_code.append(fixed_line)
            elif no_label_pattern_b.search(line):
                fixed_line = no_label_pattern_b.sub(no_label_replace_match_no_label, line)
                fixed_code.append(fixed_line)
            elif no_label_pattern.search(line):
                fixed_line = no_label_pattern.sub(no_label_replace_match, line)
                fixed_code.append(fixed_line)
            elif no_label_pattern_c.search(line):
                fixed_line = no_label_pattern_c.sub(replace_match_no_label, line)
                fixed_code.append(fixed_line)
            elif no_label_pattern_a.search(line):
                fixed_line = no_label_pattern_a.sub(no_label_replace_match, line)
                fixed_code.append(fixed_line)
            else:
                fixed_code.append(line)

        fixed_code_str = ";\n".join(fixed_code)
        
        return fixed_code_str+";"

    def save_artifact(self, ext='.py'):
        ext = self.artifact_type
        code_content = self.code_edit.toPlainText()
        first_line = code_content.split('\n')[0].strip()
        filename = self.filename

        if first_line.startswith('#'):
            # Extract filename from first line
            extracted_filename = first_line.replace('# filename: ', '').strip()
            if extracted_filename:
                filename = extracted_filename

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Artifact", filename, f"Files (*{ext})")
        if file_path:
            with open(file_path, 'w') as file:
                file.write(code_content)
            QMessageBox.information(self, "Save Successful", f"Artifact saved as {file_path}")
            
    def query_artifact(self):
        query = self.user_query_input.toPlainText()
        code_content = self.code_edit.toPlainText()

        prompt = f"{query}\n\n this:\n{code_content}\n\n"
        self.parent.wrapper.prompt_factory.history=""
        self.parent.wrapper.prompt_factory.context=""
        self.parent.wrapper.prompt_factory.task="Provide solution to the users query about the code under CONTEXT:"
        self.parent.wrapper.prompt_factory.rules="""
        ALL response must be in encapsulated 'artifacts', 
        defined by the following file types:
            <type>          :       <encapsulation>
            "###EXT###"           :    "```###LANGUAGE### <content>```"
            "yaml"          :    "```yaml <content>```"
            "txt"           :    "```text <content>```"
            "yaml"          :    "```image_description <content>```"
            "jpg"           :    "```image <content>```"
            "txt"           :    "```chat <content>```" 
            "mmd"           :    "```mermaid <content>```" 
        While answering think step-by-step and justify your answer.
        Always start the content of the artifact with # filename: <filename>.<type>
        """
        final_prompt = self.parent.wrapper.prompt_factory.update_chat_state(user_input=query, new_context=code_content)
        #rp(f"Final artifact query prompt:\n{final_prompt}")
        self.parent.send_message(input_text=final_prompt)
        #worker = Worker(self.chatbot_instance, prompt)
        #worker.signals.result.connect(self.parent.handle_response)
        #worker.signals.error.connect(self.parent.handle_error)
        #QThreadPool.globalInstance().start(worker)

    def save_image(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", self.filename, "Image Files (*.png *.jpg *.bmp)")
        if file_path:
            pixmap = self.findChild(QLabel).pixmap()
            if pixmap.save(file_path):
                QMessageBox.information(self, "Save Successful", f"Image saved as {file_path}")
            else:
                QMessageBox.warning(self, "Save Failed", "Failed to save the image.")

class BotProfile:
    def __init__(self, name, avatar_path, system_prompt, model, tools, avatar_image_description):
        self.name = name
        self.avatar_path = avatar_path
        self.system_prompt = system_prompt
        self.model = model
        self.tools = tools
        self.avatar_image_description = avatar_image_description

    def to_dict(self):
        return {
            'name': self.name,
            'avatar_path': self.avatar_path,
            'system_prompt': self.system_prompt,
            'model': self.model,
            'tools': self.tools,
            'avatar_image_description': self.avatar_image_description
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            data['name'], 
            data['avatar_path'], 
            data['system_prompt'], 
            data['model'], 
            data['tools'],
            data.get('avatar_image_description', '')  # Use get() to handle cases where this field might not exist in older data
        )
    
class BotForm(QWidget):
    def __init__(self, main_window, edit_mode=False, bot_to_edit=None):
        super().__init__()
        self.main_window = main_window
        self.edit_mode = edit_mode
        self.bot_to_edit = bot_to_edit
        self.avatar_path = bot_to_edit.avatar_path if edit_mode else ''
        self.llms = self.main_window.wrapper.chatbot.llms
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Edit Bot' if self.edit_mode else 'Add New Bot')

        layout = QVBoxLayout()

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText('Bot Name')
        layout.addWidget(QLabel('Bot Name:'))
        layout.addWidget(self.name_input)

        self.avatar_label = QLabel()
        self.avatar_label.setFixedHeight(100)
        self.avatar_label.setFixedWidth(100)
        layout.addWidget(self.avatar_label)

        self.upload_avatar_btn = QPushButton('Upload Avatar')
        self.upload_avatar_btn.clicked.connect(self.upload_avatar)
        layout.addWidget(self.upload_avatar_btn)

        self.avatar_description_input = QTextEdit()
        self.avatar_description_input.setPlaceholderText('Avatar Image Description')
        layout.addWidget(QLabel('Avatar Image Description:'))
        layout.addWidget(self.avatar_description_input)

        self.system_prompt_input = QTextEdit()
        self.system_prompt_input.setPlaceholderText('System Prompt')
        layout.addWidget(QLabel('System Prompt:'))
        layout.addWidget(self.system_prompt_input)

        models = [f"[{id}] {model.name}" for id, model in enumerate(self.llms)]
        self.model_selector = QComboBox()
        self.model_selector.addItems(models)
        layout.addWidget(QLabel('Chat Model:'))
        layout.addWidget(self.model_selector)

        self.tools = {}
        self.tools_layout = QVBoxLayout()
        for tool_name in ['Web Search', 'URL Scraper', 'Calculator']:
            checkbox = QCheckBox(tool_name)
            self.tools[tool_name] = checkbox
            self.tools_layout.addWidget(checkbox)
        layout.addWidget(QLabel('Select Tools:'))
        layout.addLayout(self.tools_layout)

        save_btn = QPushButton('Save Bot')
        save_btn.clicked.connect(self.save_bot)
        layout.addWidget(save_btn)

        self.setLayout(layout)

        if self.edit_mode:
            self.name_input.setText(self.bot_to_edit.name)
            self.system_prompt_input.setText(self.bot_to_edit.system_prompt)
            self.avatar_description_input.setText(self.bot_to_edit.avatar_image_description)
            self.model_selector.setCurrentText(self.bot_to_edit.model)
            for tool, checkbox in self.tools.items():
                checkbox.setChecked(tool in self.bot_to_edit.tools)
            if self.bot_to_edit.avatar_path:
                pixmap = QPixmap(self.bot_to_edit.avatar_path)
                self.avatar_label.setPixmap(pixmap.scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio))

    def upload_avatar(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Select Avatar Image', '', 'Images (*.png *.jpg *.jpeg)')
        if file_name:
            self.avatar_path = file_name
            pixmap = QPixmap(file_name)
            self.avatar_label.setPixmap(pixmap.scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio))

    def save_bot(self):
        bot_name = self.name_input.text()
        system_prompt = self.system_prompt_input.toPlainText()
        avatar_description = self.avatar_description_input.toPlainText()
        selected_model = self.model_selector.currentText()
        selected_tools = [tool for tool, checkbox in self.tools.items() if checkbox.isChecked()]

        if bot_name and system_prompt:
            bot_profile = BotProfile(bot_name, self.avatar_path, system_prompt, selected_model, selected_tools, avatar_description)
            if self.edit_mode:
                del self.main_window.bots[self.bot_to_edit.name]
            self.main_window.save_bot(bot_profile)
            self.close()
        else:
            QMessageBox.warning(self, 'Incomplete Form', 'Please complete all fields.')

class ChatWindow(QWidget):
    def __init__(self, bot_profile, parent):
        super().__init__()
        self.bot_profile = bot_profile
        self.parent = parent
        self.wrapper = self.parent.wrapper
        self.thread_logger = parent.thread_logger
        self.image_requester = ImageGenerator(wrapper=self.wrapper, thread_logger=self.thread_logger, parent=self)

        self.init_ui()
        self.setup_shortcuts()
        self.artifact_types = ['python', 'yaml', 'image_description', 'text', 'mermaid','chat']
    
    
    def init_ui(self):
        self.setWindowTitle(f'Chat with {self.bot_profile.name}')
        self.setGeometry(100, 100, 1200, 800)
        main_layout = QVBoxLayout()
        top_layout = QHBoxLayout()

        # Avatar display
        self.avatar_label = QLabel()
        if self.bot_profile.avatar_path:
            pixmap = QPixmap(self.bot_profile.avatar_path)
            self.avatar_label.setPixmap(pixmap.scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio))
        else:
            self.avatar_label.setText("No Avatar")
        self.avatar_label.setFixedSize(150, 150)
        top_layout.addWidget(self.avatar_label)

        # Bot info table
        self.info_table = QTableWidget(5, 2)
        self.info_table.setHorizontalHeaderLabels(['Property', 'Value'])
        self.info_table.verticalHeader().setVisible(False)
        self.info_table.setItem(0, 0, QTableWidgetItem('Name'))
        self.info_table.setItem(0, 1, QTableWidgetItem(self.bot_profile.name))
        self.info_table.setItem(1, 0, QTableWidgetItem('Model'))
        self.info_table.setItem(1, 1, QTableWidgetItem(self.bot_profile.model))
        model_index= int(self.bot_profile.model.split('[').pop().split(']').pop(0))
        self.info_table.setItem(2, 0, QTableWidgetItem('System Prompt'))
        self.info_table.setItem(2, 1, QTableWidgetItem(self.bot_profile.system_prompt))
        self.wrapper.chatbot=self.wrapper.set_bot(job='programmer',language='python', system_prompt=self.bot_profile.system_prompt)
        #self.wrapper.prompt_factory = PromptFactory( 
        #                    language='user defined',
        #                    task="Thinking step by step. Be as helpfull possible.",
        #                    rules=self.bot_profile.system_prompt,  
        #                    role=self.bot_profile.avatar_image_description
        #                )
        
        #self.wrapper.conversation_manager = ConversationManager(self.wrapper.email, self.wrapper.password, self.wrapper.cookie_folder,self.bot_profile.system_prompt, modelIndex = model_index)
        self.info_table.setItem(3, 0, QTableWidgetItem('Tools'))
        self.info_table.setItem(3, 1, QTableWidgetItem(', '.join(self.bot_profile.tools)))
        self.info_table.setItem(4, 0, QTableWidgetItem('Avatar Description'))
        self.info_table.setItem(4, 1, QTableWidgetItem(self.bot_profile.avatar_image_description))
        
        self.info_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.info_table.setFixedHeight(self.info_table.verticalHeader().length() + 
                                       self.info_table.horizontalHeader().height())
        top_layout.addWidget(self.info_table)

        main_layout.addLayout(top_layout)


        # Create the splitter to hold both chat and artifacts
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Chat display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        splitter.addWidget(self.chat_display)

        # Artifact tabs
        self.artifact_tabs = QTabWidget()  # Initialize artifact_tabs to prevent AttributeError
        splitter.addWidget(self.artifact_tabs)

        splitter.setSizes([700, 300])  # Initial sizes of the split areas
        main_layout.addWidget(splitter)

        # Input area
        input_layout = QHBoxLayout()
        self.chat_input = QLineEdit()
        input_layout.addWidget(self.chat_input)
        send_btn = QPushButton('Send')
        send_btn.clicked.connect(self.send_message)
        input_layout.addWidget(send_btn)
        main_layout.addLayout(input_layout)

        self.setLayout(main_layout)

    def setup_shortcuts(self):
        self.chat_input.returnPressed.connect(self.send_message)

    def send_message(self,input_text=None):
        if not input_text:
            user_message = self.chat_input.text()
        else:
            user_message = input_text
        #user_message = self.chat_input.text()
        if user_message.strip():
            self.chat_display.append(f"You: {user_message}")
            self.chat_input.clear()
            self.thread_logger.logger.info(f"User sent message: {user_message}")

            try:
                worker = Worker(self.wrapper.RAG_Augmented_Bot, user_message)
                worker.signals.result.connect(self.handle_response)
                worker.signals.error.connect(self.handle_error)
                self.thread_logger.logger.info("Worker for handling user message created, starting the worker...")
                self.thread_logger.thread_pool.start(worker)
                self.thread_logger.logger.info("Worker for user message started successfully")
            except Exception as e:
                self.thread_logger.logger.error(f"Failed to start worker for user message: {str(e)}")

    def handle_response(self, response):
        rp(f"response:{response}")
        for item in response:
            if item:
                item_type = item.get('type')
                content = item.get('content')
                if item_type == "chat":
                    self.chat_display.append(f"Assistant: {content}")
                else:
                    self.add_artifact_tab(item)
        
        self.chat_display.verticalScrollBar().setValue(
            self.chat_display.verticalScrollBar().maximum()
        )

    def handle_error(self, error):
        error_message = error[0]
        self.chat_display.append(f"Error: {error_message}")
        self.thread_logger.logger.error(f"Error in worker thread: {error_message}")

    def display_generated_image(self, image_path, seed_used):
        # Create a new tab for the generated image
        image_tab = QWidget()
        image_layout = QVBoxLayout()
        
        # Create a QLabel to display the image
        image_label = QLabel()
        pixmap = QPixmap(image_path)
        image_label.setPixmap(pixmap.scaled(512, 512, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        
        # Add image information
        info_label = QLabel(f"Seed used: {seed_used}")
        
        image_layout.addWidget(image_label)
        image_layout.addWidget(info_label)
        image_tab.setLayout(image_layout)
        # Get the current date and time, formatted as YYYYMMDD_HHMMSS
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Add the new tab
        image_name = f"Generated Image {self.artifact_tabs.count() + 1}"
        self.artifact_tabs.addTab(image_tab, image_name)
        filename=image_name.lower().replace(" ","_")
        rp(f"filename:{filename}")
        # Combine the filename with the date/time and the file extension

        self.artifact_tabs.setCurrentIndex(self.artifact_tabs.count() - 1)
        
        # Append a message to the chat display
        self.chat_display.append(f"Assistant: Image generated successfully. Saved under filename:{filename}")

    def append_styled_message(self, sender, message, color):
        cursor = self.chat_history.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        sender_format = QTextCharFormat()
        sender_format.setForeground(color)
        sender_format.setFontWeight(QFont.Weight.Bold)
        cursor.insertText(f"{sender}: ", sender_format)

        message_format = QTextCharFormat()
        message_format.setForeground(Qt.GlobalColor.white)
        cursor.insertText(f"{message}\n\n", message_format)

        self.chat_history.setTextCursor(cursor)
        self.chat_history.ensureCursorVisible()
        
    def add_artifact_tab(self, artifact: dict):
        artifact_type = artifact.get('type')
        artifact_content = artifact.get("content")
        artifact_filename = artifact.get("filename")

        if artifact_type == "image_description":
            '''Handle image generation task request for artifact'''
            
            self.image_requester.gen_two(f"{artifact_content}", use_dev=True)

        if artifact_type != "chat" and artifact_type in self.artifact_types:
            '''Handle non chat artifacts'''
            tab_title = artifact.get('source', '').lower() or artifact_type.capitalize()
            
            # Check if a tab with this title already exists
            existing_tab_index = -1
            for i in range(self.artifact_tabs.count()):
                if self.artifact_tabs.tabText(i) == tab_title:
                    existing_tab_index = i
                    break
            
            if existing_tab_index != -1:
                # If it exists, update its content
                existing_tab = self.artifact_tabs.widget(existing_tab_index)
                
                if isinstance(existing_tab, NewArtifactTab):
                    #if artifact_type == "mermaid":
                        #artifact_content += existing_tab.mermaid_to_jpg(artifact_content, existing_tab.filename.replace('mmd','jpg'))
                        
                    existing_tab.code_edit.setText(artifact_content)
                    existing_tab.chatbot_instance = self.wrapper.conversation_manager.chatbot
                    self.thread_logger.logger.info(f"{existing_tab.filename} Artifact  content updated!")
            else:
                # If it doesn't exist,
                # add a new tab
                new_tab = NewArtifactTab(self, artifact, self.wrapper.conversation_manager.chatbot )
                #if artifact_type == "mermaid":
                    #artifact_content +=new_tab.mermaid_to_jpg(artifact_content, new_tab.filename.replace('mmd','jpg'))
                    

                new_tab.code_edit.setText(artifact_content)
                self.artifact_tabs.addTab(new_tab, tab_title)
                self.thread_logger.logger.info(f"{tab_title} Artifact  content updated!")


            # Set the current tab to the one we just added or updated
            self.artifact_tabs.setCurrentIndex(existing_tab_index if existing_tab_index != -1 else self.artifact_tabs.count() - 1)
            self.thread_logger.logger.info(f"Focus on current Artifact tab")
 
    def apply_dark_theme(self):
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
        dark_palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)

        self.setPalette(dark_palette)
        self.setStyleSheet("""setText
            QToolTip { 
                color: #ffffff; 
                background-color: #2a82da; 
                border: 1px solid white; 
            }
            QSplitter::handle {
                background: #2a82da;
            }
            QSplitter::handle:horizontal {
                width: 4px;
            }
            QSplitter::handle:vertical {
                height: 4px;
            }
            QSplitter::handle:pressed {
                background: #2a5ada;
            }
        """)

        """ def send_message(self):
            user_message = self.chat_input.text()
            if user_message:
                self.chat_display.append(f'You: {user_message}')
                # Here you would implement chat logic with the selected bot
                bot_response = self.bot_profile.system_prompt  # For now, echo system_prompt as the response
                self.chat_display.append(f'{self.bot_profile.name}: {bot_response}')
                self.chat_input.clear() """

### Main ###
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.thread_logger = ThreadLogger()
        self.bots = self.load_bots()
        self.wrapper = HuggingChatWrapper(project_name='AgenticDashboard')
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Custom Chatbot Manager')
        self.setGeometry(100, 100, 1200, 800)
        # Layouts
        main_layout = QVBoxLayout()
        button_layout = QHBoxLayout()

        self.bot_list = QListWidget()
        self.load_bot_profiles()

        # Buttons
        self.add_bot_btn = QPushButton('Add New Bot')
        self.add_bot_btn.clicked.connect(self.show_bot_form)

        self.edit_bot_btn = QPushButton('Edit Bot')
        self.edit_bot_btn.clicked.connect(self.edit_bot)

        self.delete_bot_btn = QPushButton('Delete Bot')
        self.delete_bot_btn.clicked.connect(self.delete_bot)

        self.chat_btn = QPushButton('Chat with Selected Bot')
        self.chat_btn.clicked.connect(self.chat_with_bot)

        # Add to layout
        main_layout.addWidget(QLabel('Saved Bots:'))
        main_layout.addWidget(self.bot_list)
        button_layout.addWidget(self.add_bot_btn)
        button_layout.addWidget(self.edit_bot_btn)
        button_layout.addWidget(self.delete_bot_btn)
        button_layout.addWidget(self.chat_btn)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    def edit_bot(self):
        selected_item = self.bot_list.currentItem()
        if selected_item:
            selected_bot = self.bots[selected_item.text()]
            self.bot_form = BotForm(self, edit_mode=True, bot_to_edit=selected_bot)
            self.bot_form.show()
        else:
            QMessageBox.warning(self, 'No Bot Selected', 'Please select a bot to edit.')

    def delete_bot(self):
        selected_item = self.bot_list.currentItem()
        if selected_item:
            bot_name = selected_item.text()
            reply = QMessageBox.question(self, 'Delete Bot', f'Are you sure you want to delete {bot_name}?',
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                del self.bots[bot_name]
                self.save_bots()
                self.load_bot_profiles()
        else:
            QMessageBox.warning(self, 'No Bot Selected', 'Please select a bot to delete.')

    def show_bot_form(self):
        self.bot_form = BotForm(self)
        self.bot_form.show()

    def chat_with_bot(self):
        selected_item = self.bot_list.currentItem()
        if selected_item:
            selected_bot = self.bots[selected_item.text()]
            self.chat_window = ChatWindow(selected_bot, parent=self)
            self.chat_window.show()
        else:
            QMessageBox.warning(self, 'No Bot Selected', 'Please select a bot to chat with.')

    def load_bot_profiles(self):
        self.bot_list.clear()
        for bot_name in self.bots:
            self.bot_list.addItem(bot_name)

    def save_bot(self, bot_profile):
        self.bots[bot_profile.name] = bot_profile
        self.save_bots()
        self.load_bot_profiles()

    def load_bots(self):
        if os.path.exists('bots.json'):
            with open('bots.json', 'r') as file:
                return {bot['name']: BotProfile.from_dict(bot) for bot in json.load(file)}
        return {}

    def save_bots(self):
        with open('bots.json', 'w') as file:
            json.dump([bot.to_dict() for bot in self.bots.values()], file, indent=4)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
