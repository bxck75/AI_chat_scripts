import os,re
import faiss
import numpy as np
import pickle
import logging
from uuid import uuid4
from rich import print as rp
from dotenv import load_dotenv,find_dotenv

load_dotenv(find_dotenv())
from hugchat import hugchat
from hugchat.login import Login
from hugchat.types.tool import Tool
from hugchat.types.assistant import Assistant
from hugchat.types.message import MessageNode as Message
from hugchat.types.file import File
from langchain_core.retrievers import BaseRetriever
from hugchat.hugchat import Conversation, Model, ChatBot
from typing import List, Dict, Any,Tuple,Optional
from datetime import datetime
import uuid
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from faiss_vector_store_plot  import VectorStorePlotter
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.chains import RetrievalQA
import warnings
#logging.basicConfig(filename='chatbots.log', level=logging.INFO)
warnings.filterwarnings("ignore", category=FutureWarning, message="clean_up_tokenization_spaces")
warnings.filterwarnings("ignore", category=DeprecationWarning, message="sipPyTypeDict")
warnings.filterwarnings("ignore", category=DeprecationWarning, message="langchain")
warnings.filterwarnings("ignore", message="clean_up_tokenization_spaces was not set. It will be set to True by default. This behavior will be deprecated in transformers v4.45, and will be then set to False by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884")
	
class VectorStorage:
    def __init__(self, dim: int = None, persistence_path: Optional[str] = None):
        self.dim = dim
        self.vector_store = None
        self.index = None
        self.docstore = None
        self.persistence_path = persistence_path
        self.embeddings = self.get_embeddings()
        self.setup_vector_store()
        self.time_weighted_retriever = self.setup_timed_retriever()

    def setup_timed_retriever(self,decay=0.05, k=4):
        self.time_weighted_retriever = TimeWeightedVectorStoreRetriever(
            vectorstore=self.vector_store,
            decay_rate=decay,
            k=k
        )

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

    def get_embeddings(self):
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_folder="./embeddings_cache",
           #show_progress=True,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

    def setup_vector_store(self):
        if self.persistence_path and os.path.exists(self.persistence_path):
            #print(f"Loading existing vector store from {self.persistence_path}")
            self.vector_store = self.load_vector_store()
        else:
            #print("Creating new vector store")
            self.create_new_vector_store()

    def create_new_vector_store(self, init_msg="hello world"):
        self.dim = len(self.embeddings.embed_query(init_msg))
        self.index = faiss.IndexFlatL2(self.dim)
        self.docstore = InMemoryDocstore()
        self.vector_store = FAISS(
            self.embeddings,
            index=self.index,
            docstore=self.docstore,
            index_to_docstore_id={}
            
        )
        
    def load_vector_store(self):
        return FAISS.load_local(self.persistence_path, self.embeddings, allow_dangerous_deserialization=True)

    def save_vector_store(self):
        if self.persistence_path:
            self.vector_store.save_local(self.persistence_path)
            print(f"Vector store saved to {self.persistence_path}")
        else:
            print("No persistence path specified. Vector store not saved.")
    
    def add_and_persist(self,file_paths: List[str]):
        docs, added_files = self.fetch_documents(file_paths)
        split_docs = self.split_documents(docs)
        self.add_vectors(split_docs)
        #self.save_vector_store()
        return added_files
    
    def fetch_documents(self, file_paths: List[str]):
        documents = []
        extensions_to_load = ['.py', '.mmd', '.html', '.yaml', '.txt']
        added_files = []
        # load documents from file_paths list
        for file_path in file_paths:
            # Check if the file extension is in the allowed list
            ext = os.path.splitext(file_path)[1]
            if ext not in extensions_to_load:
                continue
            
            try:
                # Attempt to open and read the file as UTF-8
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                doc = Document(page_content=content, metadata={'source': file_path})
                documents.append(doc)
                added_files.append(file_path)
            
            except UnicodeDecodeError as e:
                print(f"Error reading {file_path}: {e}")
                # Optionally, log the error or handle it as needed

        return documents,added_files

    def add_vectors(self, documents: List[Document], ids: Optional[List[str]] = None):
        split_docs = self.split_documents(documents)
        
        # Add datetime to metadata
        current_time = datetime.now().isoformat()
        for doc in split_docs:
            doc.metadata['storage_datetime'] = current_time

        if ids is None:
            self.vector_store.add_documents(split_docs)
        else:
            if len(ids) != len(split_docs):
                raise ValueError("The number of ids must match the number of documents after splitting.")
            self.vector_store.add_documents(documents=split_docs, ids=ids)
        self.save_vector_store()

    def add_vectors_old(self, documents: List[Document], ids: Optional[List[str]] = None):
        split_docs = self.split_documents(documents)
        if ids is None:
            self.vector_store.add_documents(split_docs)
        else:
            if len(ids) != len(split_docs):
                raise ValueError("The number of ids must match the number of documents after splitting.")
            self.vector_store.add_documents(documents=split_docs, ids=ids)
        self.save_vector_store()

    def search_vectors(self, query: str, k: int):
        return self.vector_store.similarity_search_with_score(query, k)
    

    def split_documents(self, documents: List[Document], chunk_s=1024, chunk_o=0):
        split_docs = []
        for doc in documents:
            ext = os.path.splitext(getattr(doc, 'metadata', {}).get('source', '') or 
                                   getattr(doc, 'metadata', {}).get('filename', ''))[1].lower()
            if ext == '.py':
                splitter = RecursiveCharacterTextSplitter.from_language(language='python', chunk_size=chunk_s, chunk_overlap=chunk_o)
            elif ext in ['.md', '.markdown']:
                splitter = RecursiveCharacterTextSplitter.from_language(language='markdown', chunk_size=chunk_s, chunk_overlap=chunk_o)
            elif ext in ['.html', '.htm']:
                splitter = RecursiveCharacterTextSplitter.from_language(language='html', chunk_size=chunk_s, chunk_overlap=chunk_o)
            else:
                splitter = CharacterTextSplitter(chunk_size=chunk_s, chunk_overlap=chunk_o, add_start_index=True)
            
            split_docs.extend(splitter.split_documents([doc]))
        return split_docs

    def delete_vectors(self, ids: List[str]):
        self.vector_store.delete(ids)
        self.save_vector_store()

    def get_document(self, id: str) -> Optional[Document]:
        return self.vector_store.docstore.search(id)

    def save_vectorstore_local(self, folder_path: str="vectorstore", index_name: str = "faiss_index"):
        documents = self.vector_store.docstore.values()
        
        docstore: Dict[str, Document] = {}
        index_to_docstore_id: Dict[int, str] = {}
        
        for i, doc in enumerate(documents):
            doc_id = str(uuid4())
            docstore[doc_id] = doc
            index_to_docstore_id[i] = doc_id
        
        self.vector_store.save_local(folder_path, index_name)
        
        with open(os.path.join(folder_path, f"{index_name}_docstore.pkl"), "wb") as f:
            pickle.dump(docstore, f)
        
        with open(os.path.join(folder_path, f"{index_name}_index_to_docstore_id.pkl"), "wb") as f:
            pickle.dump(index_to_docstore_id, f)
        
        print(f"Vectorstore saved successfully to {folder_path}")
        return folder_path

    @classmethod
    def load_vectorstore_local(cls, folder_path: str, index_name: str = "faiss_index", embeddings=None):
        allow_dangerous_deserialization = True
        
        with open(os.path.join(folder_path, f"{index_name}_docstore.pkl"), "rb") as f:
            docstore = pickle.load(f)
        
        with open(os.path.join(folder_path, f"{index_name}_index_to_docstore_id.pkl"), "rb") as f:
            index_to_docstore_id = pickle.load(f)
        
        vectorstore = FAISS.load_local(
            folder_path,
            embeddings or HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
            index_name,
            allow_dangerous_deserialization=allow_dangerous_deserialization
        )
        vectorstore.docstore = docstore
        vectorstore.index_to_docstore_id = index_to_docstore_id
        
        instance = cls()
        instance.vector_store = vectorstore
        return instance

class Artifact:
    def __init__(self, content: Any, type: str, metadata: Dict[str, Any] = None):
        self.content = content
        self.type = type
        self.metadata = metadata or {}

class KnowledgeRetriever():
    def __init__(self, parent):
        self.parent = parent
        self.vector_storage = self.parent.vector_storage

    def _get_relevant_documents(self, query: str) -> List[Document]:
        # This method is required by BaseRetriever
        return [doc for doc, _ in self.retrieve(query, k=5)]

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        # Async version if needed
        return self._get_relevant_documents(query)

    def retrieve(self, query: str, k: int) -> List[Tuple[Document, float]]:
        return self.vector_storage.search_vectors(query, k)
    
    def retrieve_summarized(self, query: str, nr_of_docs: int = 5) -> Document:
        base_context = """This knowledge concentrate will provide extra context."""
        base_retriever = self.vector_storage.vector_store.as_retriever(query=query, k=nr_of_docs)
        merger_retriever = MergerRetriever(base_retriever, base_context, verbose=True)
        results = merger_retriever._get_relevant_documents(query)
        return results[0]

class MergerRetriever():
    def __init__(self, base_retriever, base_context: str, verbose: bool = False):
        self.base_retriever = base_retriever
        self.base_context = base_context
        self.verbose = verbose

    def _get_relevant_documents(self, query: str) -> List[Document]:
        # Get documents from the base retriever
        base_docs = self.base_retriever.invoke(query)
        
        if self.verbose:
            print(f"Retrieved {len(base_docs)} documents from base retriever")

        # Merge the base context with the retrieved documents
        merged_content = self.base_context + "\n\n" + "\n\n".join([doc.page_content for doc in base_docs])
        
        # Create a new document with the merged content
        merged_doc = Document(page_content=merged_content, metadata={"source": "merged_context"})
        
        if self.verbose:
            print(f"Created merged document with {len(merged_content)} characters")

        return [merged_doc]

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        # Implement the async version if needed
        return self._get_relevant_documents(query)

class ArtifactCollector:
    def __init__(self, vector_storage: VectorStorage):
        self.artifacts: Dict[str, List[Artifact]] = {}

    def add_artifact(self, content: Any, type: str, metadata: Dict[str, Any] = None):
        if type not in self.artifacts:
            self.artifacts[type] = []
        self.artifacts[type].append(Artifact(content, type, metadata))

    def get_artifacts(self, type: str) -> List[Artifact]:
        return self.artifacts.get(type, [])



class ArtifactDetector:
    def __init__(self, vectorstorage: VectorStorage):
        self.vectorstorage = vectorstorage
        self.artifact_types = ['python', 'yaml', 'image_description', 'text', 'mermaid', 'chat', 'json', 'bash']
        self.artifacts = []

    def detect_artifacts(self, text: str, user_input: str) -> List[Dict[str, Any]]:
        """
        Detect artifacts within the provided text and return a list of dictionaries containing artifact details.

        :param text: The text containing potential artifacts.
        :param user_input: The original user input associated with the text.
        :return: A list of dictionaries containing artifact details.
        """
        artifacts = []
        for artifact_type in self.artifact_types:
            pattern = rf"```{artifact_type}\s*(.*?)```"
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                content = match.strip()
                source = "ChatHistoryText" if artifact_type == 'chat' else self._extract_filename(content)
                
                artifact = {
                    "type": artifact_type,
                    "source": source,
                    "user_input": user_input,
                    "content": content,
                    "storage_time_date": datetime.now().isoformat(),
                    "retrieval_counter": 0,
                    "unique_id": str(uuid.uuid4())
                }
                artifacts.append(artifact)
        
        documented_artifacts = self._to_documents(artifacts)
        self.vectorstorage.add_vectors(documented_artifacts)
        self.vectorstorage.save_vector_store()
        print(f"Number of documented artifacts: {len(documented_artifacts)}")
        return artifacts

    def _extract_filename(self, content: str) -> str:
        """
        Extracts the filename from the content based on specific patterns.

        :param content: The content of the artifact where the filename might be specified.
        :return: The extracted filename or a default if not found.
        """
        filename_patterns = [
            r'filename:\s*(\S+)',
            r'\*\*\s*(\S+\.\w+)\s*\*\*'
        ]
        
        for pattern in filename_patterns:
            match = re.search(pattern, content)
            if match:
                return match.group(1)
        
        return "No_filename_found.txt"

    def _to_documents(self, data: List[Dict[str, Any]]) -> List[Document]:
        """
        Convert the detected artifact data into Document objects.

        :param data: The list of artifact data dictionaries.
        :return: List of Document objects.
        """
        documents = []
        for d in data:
            document = Document(
                page_content=d['content'],
                metadata={
                    'source': d['source'],
                    'type': d['type'],
                    'user_input': d['user_input'],
                    'storage_time_date': d['storage_time_date'],
                    'retrieval_counter': d['retrieval_counter'],
                    'unique_id': d['unique_id']
                }
            )
            documents.append(document)
        return documents

    def increment_retrieval_counter(self, unique_id: str):
        """
        Increment the retrieval counter for a specific artifact.

        :param unique_id: The unique ID of the artifact.
        """
        for artifact in self.artifacts:
            if artifact['unique_id'] == unique_id:
                artifact['retrieval_counter'] += 1
                break
          
class ConversationManager:
    
    """Manages the add,deleting andf persistance of conversations amd sessions
        --- Because every session wil create a online conversation but no garbage collection is done by hugchat
            (   LIMIT 25    ) after that no more conversations wil be added, 
            hence new sessions will timeout with EMPTY response"""

    """
    Manages the adding, deleting, and persistence of conversations and sessions.
    Handles the limit of 25 conversations imposed by HuggingChat.
    """
    def __init__(self,email,passwd,cookie_folder, system_prompt, modelIndex=0):
        
        self.email = email
        self.passwd = passwd
        self.cookie_folder= cookie_folder,
        self.system_prompt = system_prompt  # this is the prompt we will use to interact with the chatbot.
        self.conversations: Dict[str, Dict] = {}
        self.current_conversation_id=None
        self.default_model_id = modelIndex # llama3.1 70B :1 = command-r Cohere
        self.chatbot =  self._login_and_create_chatbot()
        self.load_conversations()
       
  
    def _login_and_create_chatbot(self) -> hugchat.ChatBot:
        sign = Login(self.email, self.passwd)
        #rp(self.cookie_folder)
        cookies = sign.login(self.cookie_folder)
        return ChatBot(cookies=cookies.get_dict(),  system_prompt=self.system_prompt,default_llm=int(self.default_model_id))
    
    def load_conversations(self):
        conversations_list = self.chatbot.get_conversation_list()
        #rp(f"The online conversations:\n{conversations_list}")
        self.conversations = conversations_list

    def add_conversation(self, conversation_id: str = None) -> str:
        """
        Add a new conversation. If no ID is provided, create a new one.
        If the limit is reached, remove the oldest conversation.
        """
        if len(self.conversations) >= 25:
            oldest_id = min(self.conversations, key=lambda k: self.conversations[k]['last_used'])
            self.delete_conversation(oldest_id)

        if conversation_id is None:
            # before we make a new conversation we need to determine :
            # modelIndex 0~7 (default to 0), 
            # desired system_prompt_template , 
            # artifact collection, 
            # storage, 
            # retriever,
            #   
            # then we can :
            # ingest user inputs
            # retrieve potential artifacts on the users input
            # Inject this context in the system_prompt_template to prep it for creation
            # create the conversation(model_id, system_prompt, switch_to=True)
            # [Chat for the user can start]
            conversation_id = self.chatbot.new_conversation(model_id, system_prompt, switch_to=True)
            # the artifact retriever:
            #       retrieves context with the users intput
            #       concats this before the user input 
    #                   ONLY to the request towards the chatbot!
            #           NOT in the output of the chat! 
            #           NOT entering the the artifact collector!
        
            

        self.conversations[conversation_id] = {
            'id': conversation_id,
            'created_at': datetime.now().isoformat(),
            'last_used': datetime.now().isoformat()
        }
        self.save_conversations()
        self.current_conversation_id = conversation_id
        return conversation_id

    def delete_conversation(self, conversation_id: str):
        """Delete a conversation both locally and on HuggingChat."""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            self.chatbot.delete_conversation(conversation_id)
            self.save_conversations()

    def get_conversation(self, conversation_id: str) -> Dict:
        """Retrieve a conversation by its ID."""
        return self.conversations.get(conversation_id)

    def list_conversations(self) -> List[Dict]:
        """List all conversations."""
        self.load_conversations()
        return self.conversations

    def use_conversation(self, conversation_id: str):
        """Mark a conversation as used, updating its last_used timestamp."""
        if conversation_id in self.conversations:
            self.conversations[conversation_id]['last_used'] = datetime.now().isoformat()
            self.save_conversations()

    def clean_old_conversations(self, days: int = 30):
        """Remove conversations older than the specified number of days."""
        now = datetime.now()
        to_delete = []
        for conv in self.conversations:
            created_at = datetime.fromisoformat(conv.created_at)
            if (now - created_at).days > days:
                to_delete.append(str(conv))
                self.delete_conversation(str(conv))



        #for str(conv) in to_delete:
            #self.delete_conversation(str(conv))

    def get_or_create_conversation(self) -> str:
        """Get an existing conversation or create a new one if none exist."""
        if not self.conversations:
            return self.add_conversation()
        return next(iter(self.conversations))

    def chat(self, message: str, conversation_id: str = None, web_search: bool=False) -> str:
        """Send a message to a specific conversation or create a new one."""
        if conversation_id is None or conversation_id not in self.conversations:
            conversation_id = self.get_or_create_conversation()

        self.chatbot.change_conversation(conversation_id)
        response = self.chatbot.chat(text=message, web_search=web_search)
        self.use_conversation(conversation_id)
        return response

class PromptFactory:
    def __init__(self, 
                 language="english",
                 extention="txt",
                 task="Provide weather information", 
                 rules="Be concise and accurate", 
                 role="AI Assistant"
                 ):
        self.template = """
            You ACT in the ROLE of {role}
            Your TASK is to assist {task}

            Your chat with the user will be automatically augmented so you can respond even better:
                - recent chat 'HISTORY:'
                - retrieved 'CONTEXT:' from external sources.
                - 'RULES:' to follow

            Here's how you should respond:
            {rules}

            HISTORY:
            {history}
            **Final Notes:**
            Remember 'You Rock!' think step by step and don't break ACT nor ROLE nor TASK.
            CONTEXT:
            {context}
            User Question:
            {input}
            """
        self.language = language
        self.extention = extention
        self.task = task
        self.rules = rules
        self.role = role
        self.history = "The start of a new chat."
        self.context = "No context provided."

    def create_prompt(self, user_input):
        """
        Create a prompt using the current state of the PromptFactory.
        
        :param user_input: The user's input or question
        :return: The formatted prompt string
        """
        replacements = {
            "role": self.role.replace("{language}", self.language),
            "task": self.task,
            "rules": self.rules.replace("###EXT###",self.extention).replace("###LANGUAGE###",self.language),
            "history": self.history,
            "context": self.context,
            "input": user_input
        }
        
        return self.template.format(**replacements)

    def update_chat_state(self, user_input, new_history=None, new_context=None):
        """
        Update the chat state with new history, context, and user input.
        
        :param user_input: The new user input
        :param new_history: The updated chat history
        :param new_context: The updated context
        :return: The updated prompt string
        """
        # Append the new history if provided
        #if new_history:
        #    self.history += f"\n{new_history}"
        
        # Update the context if provided
        #if new_context:
        #    self.context = new_context
         # Add chat history
        if new_history:
            history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in new_history])
            self.history += f"\n\nCHAT HISTORY:\n{history_str}"

        # Add retrieved knowledge in the new format
        if new_context:
            context_str = ""
            for item in new_context:
                # Check if item has both document and score
                if isinstance(item, tuple) and len(item) == 2:
                    doc, score = item
                else:
                    # If only document is present, set a default score
                    doc = item
                    score = "N/A"
                if isinstance(doc, Document):
                    context_str += f"Score: {score}" 
                    context_str += f"Source: {doc.metadata['source']}\n"
                    context_str += f"Type: {doc.metadata['type']}\n"
                    context_str += f"Content:\n```\n{doc.page_content}\n```\n\n"
                else:
                    context_str += f"Score: {score}\n"
                    context_str += f"Content: {doc}\n\n"
            self.context = f"\n\nCONTEXT:\n{context_str}"

        # Update the chat history with the user's new input
        #self.history += f"\nUser: {user_input}"

        # Generate and return the updated prompt
        return self.create_prompt(user_input)

class HuggingChatWrapper:
    def __init__(self, project_name: str, cookie_folder: str = "cookies", gallery_folder: str="gallery", storage_folder: str ="storage", datasets_folder: str = "datasets"):        

        self.email = os.getenv("EMAIL")
        self.password = os.getenv("PASSWD")
        self.project_name = project_name
        
        self.cookie_folder = project_name+"/"+cookie_folder+'/'
        self.storage_folder = os.path.join(project_name,storage_folder)
        self.datasets_folder = os.path.join(project_name,datasets_folder)
        self.gallery_folder = os.path.join(project_name,gallery_folder)
        os.makedirs(self.project_name,exist_ok=True)
        os.makedirs(self.cookie_folder,exist_ok=True)
        os.makedirs(self.datasets_folder,exist_ok=True)
        os.makedirs(self.gallery_folder,exist_ok=True)

        self.history = ""
        self.artifacts = []
        job='programmer'
        language="""python"""
        #self.set_bot(job,language)
        self.vector_storage         = VectorStorage(persistence_path=self.storage_folder)
        self.vectorstore_plotter    = VectorStorePlotter(self.vector_storage.vector_store)
        #self.visualizer             = Visualizer(self) # ERROR! qapp before qwidget ERROR!
        self.knowledge_retriever    = KnowledgeRetriever(parent=self)
        self.artifact_detector      = ArtifactDetector(self.vector_storage)
        self.artifact_collector     = ArtifactCollector(self.vector_storage)
        #self.chat()
    
    def set_bot(self, job='programmer', language='python', system_prompt=None,model_id=0):
        """
        Set up the chatbot with the specified job, language, and system prompt.
        """
        role,task,rules = self.get_prompt_parts(role=f'{job}:{language}')
        
        self.prompt_factory=PromptFactory( 
                            language=language,
                            task=task,
                            rules=rules,  
                            role=role
                        )
        if not system_prompt:
            system_prompt = self.prompt_factory.create_prompt(user_input="")
        rp(f"system_prompt:{system_prompt}")
        #rp(self.email, self.password,self.cookie_folder,self.prompt_factory.create_prompt(user_input=""))
        self.conversation_manager   = ConversationManager(self.email, self.password,self.cookie_folder,system_prompt, modelIndex=model_id)
        self.chatbot = self.conversation_manager.chatbot
        
        return self.chatbot
    
    def get_prompt_parts(self, role='programmer:python',artifacts=True):
        base_role="Highly intelligent RAG augmented "
        base_rules="While answering think step-by-step and justify your answer."
        base_task="Assist users by "
        if 'programmer:' in role:
            lang=role.split(':').pop()
            ext='py'
            base_role +=f"""{lang} Coder"""
            base_task +=f"""generating {lang} code that is:
                OOP
                fully implemented
                procedural
                generic
                complete
                conform autopep8 format
                contains NO placeholders
            """
        elif 'code_analist:' in role:
            lang=role.split(':').pop()
            ext='py'
            base_role +=f"""{lang} Analyst"""
            base_task +=f"""generating a {lang} code analysis report that:
                is efficient
                is well-documented
                is clear
                is concise
                is complete
                contains a speculated list of 10 future features.
            """
        else:
            lang=role.split(':').pop()
            ext='txt'
            base_role +=f"""{lang} chat agent"""
            base_task +=f"""generating a {lang} text response that:
                is concise
                is well-structured
                is clear
                is complete
                is well-formatted
                contains no placeholders
            """
        if artifacts:
            base_rules+=f"""
            ALL response must be in encapsulated 'artifacts', 
            defined by the following file types:
                <type>          :       <encapsulation>
                "{ext}"         :    "```{lang} <content>```"
                "yaml"          :    "```yaml <content>```"
                "txt"           :    "```text <content>```"
                "yaml"          :    "```image_description <content>```"
                "jpg"           :    "```image <content>```"
                "txt"           :    "```chat <content>```" 
                "mmd"           :    "```mermaid <content>```" 
            Always start the content of the artifact with # filename: <filename>.<type>
            """
        return base_role,base_task, base_rules 
  
    def chat(self):
        # continues chat until context window is full of growing history
        while True:
            user_input = input("User:")
            self.test_system(user_input)
            rp(self.history)
    
    def test_system(self, user_input):
        knowledge_retrieved = self.knowledge_retriever.retrieve(query=user_input,k=1)
        updated_prompt      = self.prompt_factory.update_chat_state(user_input=user_input,new_history=self.history,new_context=knowledge_retrieved)
        raw_response        = self.chatbot.chat(text=updated_prompt)
        self.artifacts      = self.artifact_detector.detect_artifacts(text=raw_response, user_input=user_input)
        concat_content = ""
        for art in self.artifacts:
            rp(art)
            concat_content += str(art) + "\n"
        
        # TODO Implementation: Manage chat history size
        MAX_HISTORY_SIZE = 500  # Define the maximum allowed size for the chat history in characters

        # Combine the new interaction (user input + chatbot response) with the existing history
        new_interaction = f"User: {user_input}\nAssistant: {raw_response}\n"
        new_history_size = len(self.history) + len(new_interaction)

        # Check if the new history size exceeds the maximum allowed size
        if new_history_size > MAX_HISTORY_SIZE:
            # Determine how many characters need to be removed
            excess_characters = new_history_size - MAX_HISTORY_SIZE
            
            # Trim the oldest part of the history by removing excess characters
            self.history = self.history[excess_characters:]

        # Step 5: Update the chat history with the new interaction
        self.history += new_interaction

        return self.artifacts
    
    def RAG_Augmented_Bot(self, user_input):
        knowledge_retrieved = self.knowledge_retriever.retrieve(query=user_input,k=2)
        updated_prompt      = self.prompt_factory.update_chat_state(user_input=user_input,new_history=self.history,new_context=knowledge_retrieved)
        #rp(f"Updated prompt-->[{updated_prompt}]<--")
        raw_response        = self.chatbot.chat(text=updated_prompt)
        self.artifacts      = self.artifact_detector.detect_artifacts(text=raw_response, user_input=user_input)
        
        concat_content = ""
        
        concat_content = '\n'.join([str(artifact) for artifact in self.artifacts])
        
        #rp(dir(self.artifacts))
        # TODO Implementation: Manage chat history size
        MAX_HISTORY_SIZE = 500  # Define the maximum allowed size for the chat history in characters
        # Combine the new interaction (user input + chatbot response) with the existing history
        new_interaction = f"User: {user_input}\nAssistant: {raw_response}\n"
        new_history_size = len(self.history) + len(new_interaction)
        # Check if the new history size exceeds the maximum allowed size
        if new_history_size > MAX_HISTORY_SIZE:
            # Determine how many characters need to be removed
            excess_characters = new_history_size - MAX_HISTORY_SIZE
            # Trim the oldest part of the history by removing excess characters
            self.history = self.history[excess_characters:]
        # Step 5: Update the chat history with the new interaction
        self.history += new_interaction
        return self.artifacts
            

    def _chat(self, message: str) -> Message:
        relevant_artifacts = self.knowledge_retriever.retrieve(message, k=3)
        context = self._format_context(relevant_artifacts)
        
        full_message = f"{context}\n\nUser: {message}"
        
        response = self.chatbot.chat(full_message)
        self._collect_artifacts(response)
        return response

    def _collect_artifacts(self, response: Message):
        text = response.get_final_text()
        detected_artifacts = self.artifact_detector.detect_artifacts(text)
        
        for artifact in detected_artifacts:
            self.artifact_collector.add_artifact(artifact["content"], artifact["type"])
        
        self.artifact_collector.add_artifact(text, "text")

    def _format_context(self, relevant_artifacts: List[Tuple[Document, float]]) -> str:
        context = "Relevant information:\n"
        for doc, score in relevant_artifacts:
            context += f"- {doc.metadata.get('type', 'text')}: {doc.page_content[:100]}... (relevance: {score:.2f})\n"
        return context

    def retrieve_knowledge(self, query: str, k: int):
        return self.knowledge_retriever.retrieve(query, k)

