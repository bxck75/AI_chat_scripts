import sys
import os
import json
import logging,warnings
import numpy as np
from datetime import datetime

from glob import glob
import faiss
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from rich import print as rp
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,QGridLayout, 
                             QListWidget, QCheckBox, QTextEdit, QLineEdit, QPushButton, QTabWidget,QComboBox,QDialogButtonBox,
                             QSplitter, QListWidgetItem, QFileDialog, QLabel,QMessageBox,QDialog,QTableWidget, QTableWidgetItem,
                             QScrollArea,QDoubleSpinBox,QPlainTextEdit,QSpinBox,QDockWidget,QSizePolicy,QHeaderView,QInputDialog,QAbstractItemView)
from PyQt6 import QtWidgets
from PyQt6.QtGui import QIcon,QPalette, QColor, QPixmap, QTextCharFormat,QSyntaxHighlighter,QTextCursor,QFont,QQuaternion,QImage,QVector3D
from PyQt6.QtCore import QSize, QDir, Qt, pyqtSlot, QObject, pyqtSignal, QRunnable, QThreadPool, QThread, QTimer, QUrl, pyqtSlot,QObject
from PyQt6.QtDataVisualization import (QScatter3DSeries, QScatterDataItem, Q3DScatter,
                                       QValue3DAxis, QScatterDataProxy)
from PyQt6.QtWebEngineWidgets import QWebEngineView
from hugchat import hugchat
from hugchat.login import Login
from langchain.chains.prompt_selector import ConditionalPromptSelector
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from components.hugging_chat_wrapper import HuggingChatWrapper
from components.PlatterPlotter import Plot
from chat_prompt_test import build_rag_prompt
from langchain.memory.prompt import SUMMARY_PROMPT,ENTITY_MEMORY_CONVERSATION_TEMPLATE,_DEFAULT_SUMMARIZER_TEMPLATE
import matplotlib.pyplot as plt
from langchain_core.runnables.base import Runnable
from dotenv import load_dotenv
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
load_dotenv()
import matplotlib

#rom PyQt5.QtWidgets import QWidget # type: ignore
from matplotlib.figure import Figure
matplotlib.use('qtagg')  # Ensure Matplotlib uses PyQt6
#FacePerceiverResampler._get_name()
warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(filename='chatbots.log', level=logging.DEBUG)
warnings.filterwarnings("ignore", category=DeprecationWarning, message="sipPyTypeDict")
os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--disable-gpu"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class EmbeddingVisualizer:
    def __init__(self, vector_storage = None, thread_logger=None, parent=None):
        """
        Initialize the class with a set of vectors (embeddings).
        :param vectors: numpy array of shape (n_samples, n_features), default None
        """
        self.parent = parent
        self.wrapper = parent.wrapper
        self.thread_logger = thread_logger
        self.persistence_path = self.parent.wrapper.storage_folder
        self.vector_storage = vector_storage
        
        #self.vectors = self.get_vectors_from_faiss(vectorstore=self.vector_storage.vector_store)
        self.vectors, self.datetimes = self.get_vectors_and_metadata_from_faiss(vectorstore=self.vector_storage.vector_store)
        
    def get_vectors_and_metadata_from_faiss(self, vectorstore):
        if hasattr(vectorstore.index, 'reconstruct'):
            # Number of vectors
            num_vectors = vectorstore.index.ntotal
            # Dimensionality
            d = vectorstore.index.d
            # Initialize array for vectors
            vectors = np.zeros((num_vectors, d), dtype=np.float32)
            # Initialize list for storage_datetime
            storage_datetimes = []
            
            # Retrieve vectors and metadata
            for i in range(num_vectors):
                vectors[i] = vectorstore.index.reconstruct(i)
                # Get metadata for the current vector
                #rp("index if:")
                #rp(vectorstore.index_to_docstore_id[i])
                metadata = vectorstore.docstore.search(vectorstore.index_to_docstore_id[i]).metadata
                #rp("MetaData:")
                #rp(metadata)
                storage_datetimes.append(metadata.get('storage_datetime'))
            
            self.parent.thread_logger.logger.info(f"Retrieved {len(vectors)} vectors and metadata for plot")
            return vectors, storage_datetimes
        else:
            raise NotImplementedError("Reconstruction is not supported for this index type.")
        
    def get_vectors_from_faiss(self, vectorstore):
        if hasattr(vectorstore.index, 'reconstruct'):
            # Number of vectors
            num_vectors = vectorstore.index.ntotal
            # Dimensionality
            d = vectorstore.index.d
            #rp(f'dims:{d}')
            # Initialize array
            vectors = np.zeros((num_vectors, d), dtype=np.float32)
            # Retrieve vectors
            for i in range(0,num_vectors):
                
                vectors[i] = vectorstore.index.reconstruct(i)
            self.parent.thread_logger.logger.info(f"Retrieved {len(vectors)} vectors for plot")
            return vectors
        else:
            raise NotImplementedError("Reconstruction is not supported for this index type.")

    def plot_3d_scatter(self):
        """
        Plot a 3D scatter plot using PCA for dimensionality reduction to 3D.
        """
        if self.vectors.shape[1] > 3:
            pca = PCA(n_components=3)
            reduced_vectors = pca.fit_transform(self.vectors)
        else:
            reduced_vectors = self.vectors

        fig = px.scatter_3d(x=reduced_vectors[:, 0], y=reduced_vectors[:, 1], z=reduced_vectors[:, 2],
                            color=reduced_vectors[:, 0], size_max=18)
        fig.update_traces(marker=dict(size=5))
        fig.show()

    def plot_tsne(self):
        """
        Plot a 2D t-SNE visualization of the vectors.
        """
        tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
        reduced_vectors = tsne.fit_transform(self.vectors)

        plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c='blue', cmap='Spectral')
        plt.title("t-SNE Visualization")
        plt.show()

    def plot_umap(self):
        """
        Plot a 2D UMAP visualization of the vectors.
        """
        reducer = umap.UMAP()
        reduced_vectors = reducer.fit_transform(self.vectors)

        plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c='green', cmap='Spectral')
        plt.title("UMAP Visualization")
        plt.show()

    def plot_force_directed_graph(self):
        """
        Plot a force-directed graph using the similarity matrix between vectors.
        """
        G = nx.random_geometric_graph(len(self.vectors), 0.125)  # Use a random geometric graph as an example
        pos = nx.spring_layout(G)

        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        node_x = []
        node_y = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

        fig = go.Figure(data=[go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(color='gray')),
                              go.Scatter(x=node_x, y=node_y, mode='markers', marker=dict(size=10, color='blue'))])
        fig.show()

    def plot_heatmap(self):
        """
        Plot a heatmap of the similarity matrix between vectors.
        """
        similarity_matrix = np.dot(self.vectors, self.vectors.T)  # Example: Cosine similarity
        sns.heatmap(similarity_matrix, cmap='coolwarm')
        plt.title("Similarity Matrix Heatmap")
        plt.show()

    def plot_chord_diagram(self):
        """
        Plot a simple chord diagram using a similarity matrix between vectors.
        """
        labels = ['A', 'B', 'C', 'D']
        matrix = np.random.randint(1, 5, size=(len(labels), len(labels)))  # Random example matrix

        fig = go.Figure(go.Heatmap(z=matrix, x=labels, y=labels))
        fig.show()

    def plot_parallel_coordinates(self):
        """
        Plot a parallel coordinates visualization.
        """
        fig = px.parallel_coordinates(self.vectors, color=self.vectors[:, 0], labels={
            str(i): f"Feature {i}" for i in range(self.vectors.shape[1])
        })
        fig.show()

    def plot_radial_chart(self):
        """
        Plot a radial bar chart for vector magnitudes or features.
        """
        labels = np.array([f'Feature {i}' for i in range(self.vectors.shape[1])])
        values = np.linalg.norm(self.vectors, axis=0)  # Example: norm of the vector

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        theta = np.linspace(0.0, 2 * np.pi, len(labels), endpoint=False)
        bars = ax.bar(theta, values, width=0.4)

        plt.title("Radial Bar Chart")
        plt.show()

class RunnableChatBot(Runnable):
    def __init__(self, chatbot):
        self.chatbot = chatbot
    
    def invoke(self, input, config=None, **kwargs):
        rp(f"input:{input.messages}")
        adjective = input.messages[0]#['adjective']
        return str(self.chatbot.chat(adjective.content))

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

class StorageManager(QWidget):
    def __init__(self, thread_logger, parent=None):
        super().__init__( parent)
        self.parent = parent
        self.thread_logger = thread_logger
        self.vector_storage = parent.wrapper.vector_storage
        self.visualizer = self.parent.visualizer
        
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        # File selection
        file_layout = QHBoxLayout()
        self.file_input = QLineEdit()
        self.file_input.setPlaceholderText("Select a file...")
        self.file_button = QPushButton("Browse File")
        file_layout.addWidget(self.file_input)
        file_layout.addWidget(self.file_button)

        # Folder selection
        folder_layout = QHBoxLayout()
        self.folder_input = QLineEdit()
        self.folder_input.setPlaceholderText("Select a folder...")
        self.folder_button = QPushButton("Browse Folder")
        folder_layout.addWidget(self.folder_input)
        folder_layout.addWidget(self.folder_button)

        # Add button
        self.add_button = QPushButton("Add and Persist")

        self.file_list = QListWidget()
        #self.load_button = QPushButton("Load File")
        #self.save_button = QPushButton("Save File")

        main_layout.addLayout(file_layout)
        main_layout.addLayout(folder_layout)
        main_layout.addWidget(self.add_button)
        main_layout.addWidget(self.file_list)
        #main_layout.addWidget(self.load_button)
        #main_layout.addWidget(self.save_button)

        self.file_button.clicked.connect(self.browse_file)
        self.folder_button.clicked.connect(self.browse_folder)
        self.add_button.clicked.connect(self.add_and_persist)
        #self.load_button.clicked.connect(self.load_file)
        #self.save_button.clicked.connect(self.save_file)

        self.setLayout(main_layout)
        self.apply_dark_theme()

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File", "", "All Files (*)")
        if file_path:
            self.file_input.setText(file_path)

    def browse_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.folder_input.setText(folder_path)

    def add_and_persist(self):
        document_paths = []
        extensions_to_load = ['.py', '.mmd', '.html', '.yaml', '.txt']

        if self.file_input.text():
            file_path = self.file_input.text()
            ext = os.path.splitext(file_path)[1]
            if ext in extensions_to_load:
                document_paths.append(self.file_input.text())
        # Check for folder input and use glob to recursively find files
        if self.folder_input.text():
            folder_path = self.folder_input.text()
            search_pattern = os.path.join(folder_path, '**', '*')  # This pattern finds all files recursively
            files = glob(search_pattern, recursive=True)
            for file_path in files:
                # Filter out directories and add only files
                ext = os.path.splitext(file_path)[1]
                if ext in extensions_to_load and os.path.isfile(file_path) and not "__" in file_path:
                    document_paths.append(file_path)
        
        # Process the document paths if any are found
        if len(document_paths) > 0:
            self.thread_logger.logger.info(f"Adding {len(document_paths)} documents")
            
            # Async worker that adds the documents to the vectorstore
            worker = Worker(self.vector_storage.add_and_persist, document_paths)
            worker.signals.result.connect(self.handle_storage_result)
            worker.signals.error.connect(self.handle_storage_error)
            self.thread_logger.thread_pool.start(worker)

            
    def handle_storage_result(self, result):
        # The result should be the list of added file paths
        added_file_paths = result
        self.thread_logger.logger.info(f"Done adding documents. {added_file_paths}")
        # Clear the QListWidget if you want to reset it (optional)
        self.file_list.clear()
        # Add each file path to the QListWidget
        for file_path in added_file_paths:
            self.file_list.addItem(file_path)
    
    def handle_storage_error(self, error):
        self.thread_logger.logger.info(error)    
   

    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File", "", "All Files (*)")
        if file_path:
            self.file_list.addItem(file_path)

    def save_file(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save File", "", "All Files (*)")
        if file_path:
            # Add saving logic here
            self.file_list.addItem(f"Saved: {file_path}")

    def apply_dark_theme(self):
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        self.setPalette(dark_palette)
        self.setStyleSheet("""
            QPushButton {
                background-color: #2a82da;
                color: white;
            }
            QLineEdit {
                background-color: #2a2a2a;
                color: white;
                border: 1px solid #3a3a3a;
            }
        """)

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


class ScatterPlotTab(QWidget):
    def __init__(self, wrapper, thread_logger, parent):
        super().__init__()
        self.colorbar = None  # Initialize colorbar reference
        self.parent = parent
        self.thread_logger = thread_logger
        self.wrapper = wrapper
        
        self.vector_store = self.wrapper.vector_storage.vector_store
        self.vectors, self.datetimes = None, None
        
        self.scatter = Q3DScatter()
        self.series = QScatter3DSeries()
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        container = QWidget.createWindowContainer(self.scatter)
        container.setStyleSheet("""
            QWidget {
            background-image:url("""+os.path.join(self.wrapper.gallery_folder,"backgrounds/transback_1.png")+""");
            background-repeat: no-repeat;
            background-position: center;
            }
            """)
        layout.addWidget(container)

        self.scatter.addSeries(self.series)

        # Maintain the axis ranges but hide axis labels and grid
        self.scatter.axisX().setLabelFormat('')  # Hide axis labels
        self.scatter.axisY().setLabelFormat('')  # Hide axis labels
        self.scatter.axisZ().setLabelFormat('')  # Hide axis labels

        self.scatter.axisX().setTitleVisible(False)  # Hide axis titles
        self.scatter.axisY().setTitleVisible(False)  # Hide axis titles
        self.scatter.axisZ().setTitleVisible(False)  # Hide axis titles

        self.scatter.axisX().setSegmentCount(1)  # Minimize grid segments (optional)
        self.scatter.axisY().setSegmentCount(1)  # Minimize grid segments (optional)
        self.scatter.axisZ().setSegmentCount(1)  # Minimize grid segments (optional)


        # Set a background image for the plot with transparency
        theme = self.scatter.activeTheme()
        #theme.setBackgroundEnabled(True)  # Enable background to show images
        
        # Load PNG image
        image = QImage(os.path.join(self.wrapper.gallery_folder,"backgrounds/transback_1.png"))  # Path to your PNG file
        
        # Set image as background with transparency
        pixmap = QPixmap.fromImage(image)
        #theme.setBackgroundPixmap(pixmap)
        theme.setBackgroundEnabled(False)
        
        # Optionally adjust background alpha level
        #theme.setBackgroundColor(Qt.ColorScheme.Dark)  # Makes background transparent

        self.get_vectors_and_metadata_from_faiss()
        self.update_data()

    def get_vectors_and_metadata_from_faiss(self):
        if hasattr(self.vector_store.index, 'reconstruct'):
            num_vectors = self.vector_store.index.ntotal
            d = self.vector_store.index.d
            vectors = np.zeros((num_vectors, d), dtype=np.float32)
            storage_datetimes = []
            
            for i in range(num_vectors):
                vectors[i] = self.vector_store.index.reconstruct(i)
                docstore_index = self.vector_store.index_to_docstore_id[i]
                metadata = self.vector_store.docstore.search(docstore_index).metadata
                datetime_str = metadata.get('storage_datetime')
                storage_datetimes.append(datetime.fromisoformat(datetime_str) if datetime_str else None)
            
            self.parent.thread_logger.logger.info(f"Retrieved {len(vectors)} vectors and metadata for plot")
            self.vectors = vectors
            self.datetimes = storage_datetimes
        else:
            raise NotImplementedError("Reconstruction is not supported for this index type.")

    def update_data(self):
        if self.vectors is None or self.datetimes is None:
            self.parent.thread_logger.logger.error("Vectors or datetimes not available for plotting")
            return

        data_proxy = QScatterDataProxy()
        self.series.setDataProxy(data_proxy)

        data_array = []

        min_date = min(dt for dt in self.datetimes if dt is not None)
        max_date = max(dt for dt in self.datetimes if dt is not None)
        date_range = (max_date - min_date).total_seconds()

        for vector, dt in zip(self.vectors, self.datetimes):
            item = QScatterDataItem()

            # Use first three dimensions of the vector for positioning
            item.setPosition(QVector3D(vector[0], vector[1], vector[2]))

            if dt:
                # Use rotation to represent datetime
                date_factor = (dt - min_date).total_seconds() / date_range
                rotation = QQuaternion.fromAxisAndAngle(QVector3D(0, 1, 0), date_factor * 360)
                item.setRotation(rotation)

            data_array.append(item)

        data_proxy.resetArray(data_array)

        # Correctly set up the axis ranges based on actual data
        self.scatter.axisX().setRange(self.vectors[:, 0].min(), self.vectors[:, 0].max())
        self.scatter.axisY().setRange(self.vectors[:, 1].min(), self.vectors[:, 1].max())
        self.scatter.axisZ().setRange(self.vectors[:, 2].min(), self.vectors[:, 2].max())

        self.scatter.seriesList()[0].setItemSize(0.1)  # Adjust point size as needed


class ScatterPlotUpdater(QObject):
    def __init__(self, scatter, vector_store):
        super().__init__()
        self.scatter = scatter
        self.vector_store = vector_store
        self.vectors = None
        self.datetimes = None
        self.series = QScatter3DSeries(QScatterDataProxy())
        self.scatter.addSeries(self.series)
        
        # Initial data retrieval and update
        self.get_vectors_and_metadata_from_faiss()
        self.update_data()

    def get_vectors_and_metadata_from_faiss(self):
        if hasattr(self.vector_store.index, 'reconstruct'):
            num_vectors = self.vector_store.index.ntotal
            d = self.vector_store.index.d
            vectors = np.zeros((num_vectors, d), dtype=np.float32)
            storage_datetimes = []
            
            for i in range(num_vectors):
                vectors[i] = self.vector_store.index.reconstruct(i)
                docstore_index = self.vector_store.index_to_docstore_id[i]
                metadata = self.vector_store.docstore.search(docstore_index).metadata
                datetime_str = metadata.get('storage_datetime')
                storage_datetimes.append(datetime.fromisoformat(datetime_str) if datetime_str else None)
            
            self.parent.thread_logger.logger.info(f"Retrieved {len(vectors)} vectors and metadata for plot")
            self.vectors = vectors
            self.datetimes = storage_datetimes
        else:
            raise NotImplementedError("Reconstruction is not supported for this index type.")

    def update_data(self):
        if self.vectors is None or self.datetimes is None:
            self.parent.thread_logger.logger.error("Vectors or datetimes not available for plotting")
            return

        data_proxy = QScatterDataProxy()
        self.series.setDataProxy(data_proxy)

        data_array = []

        min_date = min(dt for dt in self.datetimes if dt is not None)
        max_date = max(dt for dt in self.datetimes if dt is not None)
        date_range = (max_date - min_date).total_seconds()

        for vector, dt in zip(self.vectors, self.datetimes):
            item = QScatterDataItem()
            item.setPosition(QVector3D(vector[0], vector[1], vector[2]))

            if dt:
                date_factor = (dt - min_date).total_seconds() / date_range
                rotation = QQuaternion.fromAxisAndAngle(QVector3D(0, 1, 0), date_factor * 360)
                item.setRotation(rotation)

            data_array.append(item)

        data_proxy.resetArray(data_array)

        # Set axis ranges
        self.scatter.axisX().setRange(self.vectors[:, 0].min(), self.vectors[:, 0].max())
        self.scatter.axisY().setRange(self.vectors[:, 1].min(), self.vectors[:, 1].max())
        self.scatter.axisZ().setRange(self.vectors[:, 2].min(), self.vectors[:, 2].max())

        self.series.setItemSize(0.1)  # Adjust point size as needed



class ArtifactBot(QWidget):
    def __init__(self, role, wrapper, thread_logger, parent=None):
        super().__init__(parent)
        self.thread_logger = thread_logger
        self.parent = parent
        self.role = role
        self.thread_logger.logger
        self.logger = self.thread_logger.logger
        self.init_ui()
        self.wrapper = wrapper

        self.chatbot = wrapper.chatbot
        self.logger.info(f"ArtifactBot [{role}] waking up...")
        self.thread_logger = thread_logger
        self.gen_img    = self.parent.image_requester.gen_two
        self.logger.info(f"Image Generator init...")
        self.log_info   = self.logger.info
        self.log_err    = self.logger.error
        
        self.log_info(f"{__name__} init_ui, Done!")

    def init_ui(self):
        # Main widget and layout for Tab 1
        main_layout = QVBoxLayout(self)

        self.editor = QTextEdit()  # Assuming `SyntaxHighlighterEditor` is similar to QTextEdit
        self.editor.setPlainText("Editor content loaded here")
        
        self.setWindowTitle('Async Dark-themed HuggingChat App')

        # TopBar
        #self.top_bar = QWidget()  # Replace with TopBar instance if applicable
        #main_layout.addWidget(self.top_bar)

        content_layout = QHBoxLayout()

        # Splitter for chat and artifact display
        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Chat interface
        chat_widget = QWidget()
        chat_layout = QVBoxLayout()
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFixedHeight(600)
        self.user_input = QLineEdit()
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)

        chat_layout.addWidget(self.chat_display)
        chat_layout.addWidget(self.user_input)
        chat_layout.addWidget(self.send_button)
        chat_widget.setLayout(chat_layout)

        # Right: Multi-tabbed artifact display
        self.artifact_tabs = QTabWidget()

        self.splitter.addWidget(chat_widget)
        self.splitter.addWidget(self.artifact_tabs)
        content_layout.addWidget(self.splitter)

        main_layout.addLayout(content_layout)

        # Add additional features such as dark theme and shortcuts
        self.artifact_types = ["python", "yaml", "text", "image_description", "mermaid", "chat"]
        self.parent.apply_dark_theme()
        self.setup_shortcuts()

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
        filename=image_name.lower().replace(" ","_") + f"_{current_time}.jpg"
        full=os.path.join(self.parent.wrapper.gallery_folder, filename)
        # Combine the filename with the date/time and the file extension
        rp(f"Image file name:{full}")
        pixmap.save(full)
        self.artifact_tabs.setCurrentIndex(self.artifact_tabs.count() - 1)
    
        # Append a message to the chat display
        self.chat_display.append(f"Assistant: Image generated successfully. Saved under filename:{full}")

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
    
 
    def add_artifact_from_other_bot(self, artifact, chatbot_title):
        """Handle artifact addition from another bot (QuadChatbotWidget)."""
        artifact_type = artifact.get('type')
        artifact_filename = artifact.get("source")
        self.add_artifact_tab(artifact, title=f"[{chatbot_title}-{artifact_type}]")
        return f"Artifact created for {artifact_filename}"
    
    def add_artifact_tab(self, artifact: dict, title: str = None):
        artifact_type = artifact.get('type')
        artifact_content = artifact.get("content")
        artifact_filename = artifact.get("filename")

        if artifact_type == "image_description":
            '''Handle image generation task request for artifact'''
            # generate image
            #artifact_image_filename  = artifact_filename.replace('.txt', '.png')
            self.gen_img(prompt=f"{artifact_content}", steps=25, strength=0.9)

        if artifact_type != "chat" and artifact_type in self.artifact_types:
            '''Handle non chat artifacts'''
            if title:
                prefix_title = title+ "|"
            else:
                prefix_title = ""
            tab_title = prefix_title + artifact.get('source', '').lower() or prefix_title+" "+artifact_type.capitalize()
            
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
                    existing_tab.chatbot_instance = self.chatbot
                    self.thread_logger.logger.info(f"{existing_tab.filename} Artifact  content updated!")
            else:
                # If it doesn't exist,
                # add a new tab
                new_tab = NewArtifactTab(self, artifact, self.chatbot )
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
                color: #00CC33 ; 
                background-color:  #000000 ; 
                border: 1px solid white; 
            }
            QSplitter::handle {
                background: #330099;
            }
            QSplitter::handle:horizontal {
                width: 4px;
            }
            QSplitter::handle:vertical {
                height: 4px;
            }
            QSplitter::handle:pressed {
                background: #330099;
            }
        """)

    def setup_shortcuts(self):
        self.user_input.returnPressed.connect(self.send_message)

    def show_error_message(self, message):
        error_box = QMessageBox()
        error_box.setIcon(QMessageBox.Icon.Critical)
        error_box.setText("An error occurred")
        error_box.setInformativeText(message)
        error_box.setWindowTitle("Error")
        error_box.exec()

    @pyqtSlot()
    def send_message(self, input_text=None):
        if not input_text:
            user_message = self.user_input.text()
        else:
            user_message = input_text

        if not user_message.strip():
            return
        
        self.user_input.clear()
        self.chat_display.append(f"You: {user_message}")

        self.thread_logger.logger.info(f"User sent message: {user_message}")

        try:
            worker = Worker(self.wrapper.test_system, user_message)
            worker.signals.result.connect(self.handle_response)
            worker.signals.error.connect(self.handle_error)
            self.thread_logger.logger.info("Worker for handling user message created, starting the worker...")
            self.thread_logger.thread_pool.start(worker)
            self.thread_logger.logger.info("Worker for user message started successfully")
        except Exception as e:
            self.thread_logger.logger.error(f"Failed to start worker for user message: {str(e)}")

    def handle_response(self, response):
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

    def clear_artifact_tabs(self):
        self.artifact_tabs.clear()

    def load_conversation(self, conversation_id):
        self.wrapper.set_conversation(conversation_id)
        # Clear the chat display and load the conversation history
        self.chat_display.clear()
        conversation = self.wrapper.conversations.get(conversation_id)
        if conversation:
            for message in conversation.get('history', []):
                if message['role'] == 'user':
                    self.chat_display.append(f"You: {message['content']}")
                else:
                    self.chat_display.append(f"Assistant: {message['content']}")
    
    def toggle_thumbnail_gallery(self):
        if self.thumbnail_gallery.isVisible():
            self.thumbnail_gallery.hide()
            self.toggle_thumbnail_gallery_button.setText("Show Gallery")
        else:
            self.thumbnail_gallery.show()
            self.toggle_thumbnail_gallery_button.setText("Hide Gallery")
    
    def toggle_sidebar(self):
        if self.sidebar.isVisible():
            self.sidebar.hide()
            self.toggle_sidebar_button.setText("▶")
        else:
            self.sidebar.show()
            self.toggle_sidebar_button.setText("◀") 

    def update_conversation_list(self):
        self.conversation_list.clear()
        convs_list=self.wrapper.conversation_manager.list_conversations()
        rp(f"Conversations List:\n{convs_list}")
        for conversation in convs_list:
            
            item = QListWidgetItem(f"Conversation {conversation.id}|{conversation.title}")
            self.conversation_list.addItem(item)

    def switch_highlight_to(self,profile):
        self.editor.set_profile(profile)

    def highlight_text(self,text, profile='python'):
        self.switch_highlight_to(profile)
        self.editor.highlighter.highlightBlock(text.toPlainText())



class ChatWorker(QThread):
    response_ready = pyqtSignal(str)

    def __init__(self, generate_response_func, query, model_id, system_prompt):
        super().__init__()
        self.generate_response_func = generate_response_func
        self.query = query
        self.model_id = model_id
        self.system_prompt = system_prompt

    def run(self):
        response = self.generate_response_func(self.query, self.model_id, self.system_prompt)
        self.response_ready.emit(response)


class ChatTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)

        # Create a splitter for resizable sections
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Sidebar
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)

        # Avatar
        self.avatar_label = QLabel()
        self.avatar_label.setFixedSize(150, 150)
        self.avatar_label.setScaledContents(True)
        sidebar_layout.addWidget(self.avatar_label)

        # Bot info
        self.info_area = QTextEdit()
        self.info_area.setReadOnly(True)
        sidebar_layout.addWidget(self.info_area)

        sidebar.setLayout(sidebar_layout)
        splitter.addWidget(sidebar)

        # Chat area
        chat_widget = QWidget()
        chat_layout = QVBoxLayout(chat_widget)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        chat_layout.addWidget(self.chat_display)

        input_layout = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.returnPressed.connect(self.parent.send_message)
        send_button = QPushButton("Send")
        send_button.clicked.connect(self.parent.send_message)
        input_layout.addWidget(self.chat_input)
        input_layout.addWidget(send_button)

        chat_layout.addLayout(input_layout)
        splitter.addWidget(chat_widget)

        layout.addWidget(splitter)
        self.setLayout(layout)



    def update_profile(self, profile):
        if profile:
            # Update avatar
            avatar_path = profile.get('avatar_image', '')
            if avatar_path:
                pixmap = QPixmap(avatar_path)
                self.avatar_label.setPixmap(pixmap)
            else:
                self.avatar_label.clear()

            # Update bot info
            info = f"Name: {profile['name']}\n"
            info += f"Model: {self.parent.models[profile['model_id']]}\n"
            info += f"Role: {profile.get('role', 'N/A')}\n"
            info += f"\nRules:\n{profile.get('rules', 'N/A')}\n"
            info += f"\nVisual Description:\n{profile.get('avatar_visual_description', 'N/A')}"
            self.info_area.setPlainText(info)
        else:
            self.avatar_label.clear()
            self.info_area.clear()

    def display_message(self, message, is_user=True):
        sender = "You" if is_user else "Assistant"
        self.chat_display.append(f"{sender}: {message}")
        self.chat_display.moveCursor(QTextCursor.MoveOperation.End)

    def display_artifacts(self, artifacts):
        # Clear existing artifact tabs
        self.artifacts_tabs.clear()

        # Create new tabs for each artifact type
        for artifact_type, artifact_content in artifacts.items():
            if artifact_type != 'chat':
                artifact_display = QTextEdit()
                artifact_display.setPlainText(artifact_content)
                artifact_display.setReadOnly(True)
                self.artifacts_tabs.addTab(artifact_display, artifact_type.capitalize())

class ChatbotProfileManager(QMainWindow):
    def __init__(self):
        super().__init__()
        self.project_name = 'Chatbot_Profile_Manager'
        self.models=['meta-llama/Meta-Llama-3.1-70B-Instruct', 'CohereForAI/c4ai-command-r-plus-08-2024', 'mistralai/Mixtral-8x7B-Instruct-v0.1', 'NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO', 'mistralai/Mistral-7B-Instruct-v0.3', 'microsoft/Phi-3-mini-4k-instruct']
        self.wrapper = HuggingChatWrapper(project_name=self.project_name)
        self.thread_logger=ThreadLogger()
        self.visualizer = EmbeddingVisualizer(self.wrapper.vector_storage, self.thread_logger, self)
        self.init_ui()
        self.profiles = []
        self.load_profiles()
        self.current_profile = None


        self.update_profile_list()
    
    def init_ui(self):
        self.setWindowTitle(self.project_name.replace('_',' '))
        self.setGeometry(100, 100, 1100, 700)


        main_layout = QHBoxLayout()
      # Create TopBar
        #self.top_bar = TopBar(self.thread_logger)
        #main_layout.addWidget(self.top_bar)

        # Sidebar
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout()
        self.profile_list = QListWidget()
        self.profile_list.itemClicked.connect(self.load_profile)
        sidebar_layout.addWidget(QLabel("Saved Profiles:"))
        sidebar_layout.addWidget(self.profile_list)
        sidebar.setLayout(sidebar_layout)
        sidebar.setFixedWidth(200)

        # Main content
        content = QWidget()
        content.setStyleSheet("""
        """)
        content_layout = QVBoxLayout()
        

        # Define the icons for each tab
        quad_bot_icon           = QIcon(os.path.join(self.wrapper.gallery_folder, 'icons/quad_bot.png'))
        artifact_bot_icon       = QIcon(os.path.join(self.wrapper.gallery_folder, 'icons/chat.png'))
        image_gallery_icon      = QIcon(os.path.join(self.wrapper.gallery_folder, 'icons/gallery.png'))
        flux_capacitor_icon     = QIcon(os.path.join(self.wrapper.gallery_folder, 'icons/transcribe_bot.png'))
        profile_manager_icon    = QIcon(os.path.join(self.wrapper.gallery_folder, 'icons/mermaid_viewer.png'))
        dataset_manager_icon    = QIcon(os.path.join(self.wrapper.gallery_folder, 'icons/datasets.png'))
        store_manager_icon      = QIcon(os.path.join(self.wrapper.gallery_folder, 'icons/storage_manager.png'))
        transcribe_bot_icon     = QIcon(os.path.join(self.wrapper.gallery_folder, 'icons/transcribe_bot.png'))
        scatter_tab_icon        = QIcon(os.path.join(self.wrapper.gallery_folder, 'icons/scatter_plot.png'))
        plot_tab_icon           = QIcon(os.path.join(self.wrapper.gallery_folder, 'icons/scatter_plot.png'))
        class_analizer_tab_icon = QIcon(os.path.join(self.wrapper.gallery_folder, 'icons/class_analizer.png'))
        settings_icon           = QIcon(os.path.join(self.wrapper.gallery_folder, 'icons/settings.png'))
        evaluator_icon          = QIcon(os.path.join(self.wrapper.gallery_folder, 'icons/research.png'))
        log_viewer_icon         = QIcon(os.path.join(self.wrapper.gallery_folder, 'icons/log_viewer.png'))
  
        # Tabs
        self.tabs = QTabWidget()
        # Create QSplitter for resizable sections
        splitter = QSplitter(Qt.Orientation.Vertical)

        self.tabs.setStyleSheet("""
                QTabWidget::pane {
                    border: 1px;
                }
                QTabBar::tab {
                    background: black;
                    color: white;
                    padding: 10px;
                }
                QTabBar::tab:selected {
                    background: #000000;  /* Darker gray when selected */
                }
                QTabBar::tab:hover {
                    background: #000000;  /* Slightly lighter on hover */
                    border: 2px solid #06d800;  /* Add a gold border on hover */
                }
                QTabWidget::tab-bar {
                    alignment: center;
                }
                QWidget {
                    background-image: url();
                    background-repeat: no-repeat;
                    background-position: center;
                }
            """) #'/mnt/04ef09de-2d9f-4fc2-8b89-de7dc0155e26/new_code/HuggingChatWrapper/assets/backgrounds/transback_2.png'
        self.tabs.setIconSize(QSize(64, 64))  # Set the icon size to 32x32 pixels

        self.profile_tab = self.create_profile_tab()
        self.chat_A_tab = ChatTab(parent=self)
        self.chat_B_tab = ChatTab(parent=self)
        self.storage_tab = StorageManager(thread_logger=self.thread_logger, parent=self)
        self.scatter_tab = ScatterPlotTab(self.wrapper, self.thread_logger, self)
        self.plot_tab = Plot(self.wrapper, self.thread_logger, self)

        self.tabs.addTab(self.scatter_tab, scatter_tab_icon," ")
        self.tabs.setTabToolTip(7, "3D Scatter Plot")
        self.tabs.addTab(self.plot_tab, evaluator_icon," ")
        self.tabs.setTabToolTip(7, "3D Vector Plot")
        self.tabs.addTab(self.profile_tab, profile_manager_icon, " ")
        self.tabs.setTabToolTip(7, "Profile Manager")
        self.tabs.addTab(self.chat_A_tab, artifact_bot_icon, " ")
        self.tabs.setTabToolTip(7, f"Chat A")
        self.tabs.addTab(self.chat_B_tab, artifact_bot_icon, " ")
        self.tabs.setTabToolTip(7, f"Chat B")
        self.tabs.addTab(self.storage_tab, store_manager_icon, " ")
        self.tabs.setTabToolTip(7, "Storage Manager")

        content_layout.addWidget(self.tabs)
        content.setLayout(content_layout)

        main_layout.addWidget(sidebar)
        main_layout.addWidget(content)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
    

    


    def create_profile_tab(self):
        profile_tab = QWidget()
        layout = QVBoxLayout()

        self.name_input = QLineEdit()
        self.model_id_input = QComboBox()  # Changed to QComboBox
        self.model_id_input.addItems(self.models)  # Add all models to the dropdown
        self.system_prompt_input = QTextEdit()
        self.avatar_image_input = QLineEdit()
        self.avatar_visual_description_input = QTextEdit()
        self.role_input = QLineEdit()
        self.rules_input = QTextEdit()

        layout.addWidget(QLabel("Name:"))
        layout.addWidget(self.name_input)
        layout.addWidget(QLabel("Model ID:"))
        layout.addWidget(self.model_id_input)
        layout.addWidget(QLabel("System Prompt:"))
        layout.addWidget(self.system_prompt_input)
        layout.addWidget(QLabel("Avatar Image (path):"))
        layout.addWidget(self.avatar_image_input)
        layout.addWidget(QLabel("Avatar Visual Description:"))
        layout.addWidget(self.avatar_visual_description_input)
        layout.addWidget(QLabel("Role:"))
        layout.addWidget(self.role_input)
        layout.addWidget(QLabel("Rules:"))
        layout.addWidget(self.rules_input)

        button_layout = QHBoxLayout()
        save_button = QPushButton("Save")
        save_button.clicked.connect(self.save_profile)
        delete_button = QPushButton("Delete")
        delete_button.clicked.connect(self.delete_profile)
        button_layout.addWidget(save_button)
        button_layout.addWidget(delete_button)

        layout.addLayout(button_layout)
        profile_tab.setLayout(layout)
        return profile_tab

    def create_chat_tab(self):
        chat_tab = QWidget()
        layout = QVBoxLayout()

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        layout.addWidget(self.chat_display)

        input_layout = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.returnPressed.connect(self.send_message)
        send_button = QPushButton("Send")
        send_button.clicked.connect(self.send_message)
        input_layout.addWidget(self.chat_input)
        input_layout.addWidget(send_button)

        layout.addLayout(input_layout)
        chat_tab.setLayout(layout)
        return chat_tab

    def send_message(self):
        if not self.current_profile:
            QMessageBox.warning(self, "No Profile Selected", "Please select a profile before chatting.")
            return

        current_chat = self.tabs.currentWidget()
        user_message = current_chat.chat_input.text()
        if not user_message:
            return
        self.user_input = user_message
        current_chat.display_message(user_message, is_user=True)
        current_chat.chat_input.clear()

        self.chat_worker = ChatWorker(
            self.generate_response,
            user_message,
            self.current_profile['model_id'],
            self.current_profile['system_prompt']
        )
        self.chat_worker.response_ready.connect(self.display_response)
        self.chat_worker.start()
    
    def process_response(self, response):
        self.wrapper.artifacts = self.wrapper.artifact_detector.detect_artifacts(response, self.user_input)
        #self.user_input = None

    def display_response(self, response):
        self.process_response(response=response)
        artifacts = self.wrapper.artifacts
        
        current_chat = self.tabs.currentWidget()
        rp(artifacts)
        # Display chat artifacts in the main chat area
        #chat_artifacts = artifacts.get('chat', '')
        if artifacts:
            current_chat.display_message(artifacts, is_user=False)
        else:
            current_chat.display_message(response, is_user=False)
        
        # Display other artifacts in the tabbed area
        current_chat.display_artifacts(artifacts)

    def generate_response(self, query, model_id, system_prompt):
        email = os.getenv('EMAIL')
        passwd = os.getenv('PASSWD')
        sign = Login(email, passwd)
        cookies = sign.login(cookie_dir_path=self.wrapper.cookie_folder, save_cookies=True)
        sign.save_cookies(cookie_dir_path=self.wrapper.cookie_folder)
        
        self.wrapper.set_bot(system_prompt=system_prompt,model_id=model_id)
        #rp(f"model_id:{dir(model_id)}")
        #chatbot = hugchat.ChatBot(cookies=cookies.get_dict(), system_prompt=system_prompt, default_llm=self.models[int(model_id)])
        runnable_chatbot = RunnableChatBot(self.wrapper.chatbot)
        vector_storage = self.wrapper.vector_storage
        #self.wrapper.vector_storage.setup_contextual_retriever(llm=runnable_chatbot)
        chain, inputs = build_rag_prompt(runnable_llm=runnable_chatbot, vector_storage=vector_storage, question=query)
        #rp(f"runnable chatbot:{dir(runnable_chatbot.as_tool)}")
        #rp(f"runnable chatbot inputschema:{runnable_chatbot.get_input_schema().schema()}")
        #rp(f"runnable chatbot outputschema:{runnable_chatbot.get_output_schema().schema()}")
        #chain = prompt | runnable_chatbot | StrOutputParser()
        #rp(f"chain:{dir(chain.bind)}")
        response = chain.invoke(input=inputs)
        #response = chain.invoke(input={'input': query,'relevant_info':" ", 'history':" "})
        return response

    def load_profiles(self):
        try:
            with open("profiles.json", "r") as f:
                self.profiles = json.load(f)
        except FileNotFoundError:
            self.profiles = []

    def save_profiles(self):
        with open("profiles.json", "w") as f:
            json.dump(self.profiles, f)

    def update_profile_list(self):
        self.profile_list.clear()
        for profile in self.profiles:
            self.profile_list.addItem(profile["name"])

    def load_profile(self, item):
        profile = next((p for p in self.profiles if p["name"] == item.text()), None)
        if profile:
            self.current_profile = profile
            self.name_input.setText(profile["name"])
            self.model_id_input.setCurrentIndex(profile["model_id"])
            self.system_prompt_input.setPlainText(profile["system_prompt"])
            self.avatar_image_input.setText(profile.get("avatar_image", ""))
            self.avatar_visual_description_input.setPlainText(profile.get("avatar_visual_description", ""))
            self.role_input.setText(profile.get("role", ""))
            self.rules_input.setPlainText(profile.get("rules", ""))

    def save_profile(self):
        name = self.name_input.text()
        model_id = self.model_id_input.currentIndex() 
        system_prompt = self.system_prompt_input.toPlainText()
        avatar_image = self.avatar_image_input.text()
        avatar_visual_description = self.avatar_visual_description_input.toPlainText()
        role = self.role_input.text()
        rules = self.rules_input.toPlainText()

        if not name or not model_id:
            QMessageBox.warning(self, "Invalid Input", "Please fill in all required fields.")
            return

        new_profile = {
            "name": name,
            "model_id": model_id,
            "system_prompt": system_prompt,
            "avatar_image": avatar_image,
            "avatar_visual_description": avatar_visual_description,
            "role": role,
            "rules": rules
        }

        existing_profile = next((p for p in self.profiles if p["name"] == name), None)
        if existing_profile:
            existing_profile.update(new_profile)
        else:
            self.profiles.append(new_profile)

        self.current_profile = new_profile
        self.save_profiles()
        self.update_profile_list()
        QMessageBox.information(self, "Success", f"Profile for {self.current_profile['name']} saved successfully.")

    def delete_profile(self):
        name = self.name_input.text()
        self.profiles = [p for p in self.profiles if p["name"] != name]
        self.save_profiles()
        self.update_profile_list()
        self.name_input.clear()
        self.model_id_input.setCurrentIndex(0)
        self.system_prompt_input.clear()
        self.avatar_image_input.clear()
        self.avatar_visual_description_input.clear()
        self.role_input.clear()
        self.rules_input.clear()
        self.current_profile = None
        QMessageBox.information(self, "Success", "Profile deleted successfully.")
 
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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChatbotProfileManager()
    window.show()
    sys.exit(app.exec())