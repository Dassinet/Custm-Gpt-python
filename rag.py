import os
import shutil
import asyncio
import time
import json
import base64
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator, Union
from urllib.parse import urlparse
import uuid
import httpx

from dotenv import load_dotenv

# --- Qdrant ---
from qdrant_client import QdrantClient, models as qdrant_models
from langchain_qdrant import QdrantVectorStore

# --- Langchain & OpenAI Core Components ---
from openai import AsyncOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# --- Hybrid Search Components ---
from langchain.retrievers import EnsembleRetriever

# --- Langsmith ---
from langsmith import traceable
langsmith_api_key = os.getenv("LANGCHAIN_API_KEY")
langsmith_project = os.getenv("LANGCHAIN_PROJECT")
langsmith_tracing = os.getenv("LANGCHAIN_TRACING")
langsmith_endpoint = os.getenv("LANGCHAIN_ENDPOINT")

# First attempt standard imports
HYBRID_SEARCH_AVAILABLE = True
try:
    from langchain_community.retrievers import BM25Retriever
    from rank_bm25 import OkapiBM25
    HYBRID_SEARCH_AVAILABLE = True
    print("✅ BM25 package imported successfully")
except ImportError:
    # Implement our own simplified BM25 functionality
    print("⚠️ Standard rank_bm25 import failed. Implementing custom BM25 solution...")
    
    # Custom BM25 implementation
    import numpy as np
    from langchain_core.retrievers import BaseRetriever
    from typing import List, Dict, Any, Optional, Iterable, Callable
    from pydantic import Field, ConfigDict
    
    def default_preprocessing_func(text: str) -> List[str]:
        """Default preprocessing function that splits text on whitespace."""
        return text.lower().split()
    
    class BM25Okapi:
        """Simplified implementation of BM25Okapi when the rank_bm25 package is not available."""
        
        def __init__(self, corpus, k1=1.5, b=0.75, epsilon=0.25):
            self.corpus = corpus
            self.k1 = k1
            self.b = b
            self.epsilon = epsilon
            
            self.doc_freqs = []
            self.idf = {}
            self.doc_len = []
            self.avgdl = 0
            self.N = 0
            
            if not self.corpus:
                return
                
            self.N = len(corpus)
            self.avgdl = sum(len(doc) for doc in corpus) / self.N
            
            # Calculate document frequencies
            for document in corpus:
                self.doc_len.append(len(document))
                freq = {}
                for word in document:
                    if word not in freq:
                        freq[word] = 0
                    freq[word] += 1
                self.doc_freqs.append(freq)
                
                # Update inverse document frequency
                for word, _ in freq.items():
                    if word not in self.idf:
                        self.idf[word] = 0
                    self.idf[word] += 1
            
            # Calculate inverse document frequency
            for word, freq in self.idf.items():
                self.idf[word] = np.log((self.N - freq + 0.5) / (freq + 0.5) + 1.0)
        
        def get_scores(self, query):
            scores = [0] * self.N
            
            for q in query:
                if q not in self.idf:
                    continue
                    
                q_idf = self.idf[q]
                
                for i, doc_freqs in enumerate(self.doc_freqs):
                    if q not in doc_freqs:
                        continue
                    
                    doc_freq = doc_freqs[q]
                    doc_len = self.doc_len[i]
                    
                    # BM25 scoring formula
                    numerator = q_idf * doc_freq * (self.k1 + 1)
                    denominator = doc_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                    scores[i] += numerator / denominator
                    
            return scores
        
        def get_top_n(self, query, documents, n=5):
            if not query or not documents or not self.N:
                return documents[:min(n, len(documents))]
                
            scores = self.get_scores(query)
            top_n = sorted(range(self.N), key=lambda i: scores[i], reverse=True)[:n]
            return [documents[i] for i in top_n]
    
    class SimpleBM25Retriever(BaseRetriever):
        """A simplified BM25 retriever implementation when rank_bm25 is not available."""

        vectorizer: Any = None
        docs: List[Document] = Field(default_factory=list, repr=False)
        k: int = 4
        preprocess_func: Callable[[str], List[str]] = default_preprocessing_func

        model_config = ConfigDict(
            arbitrary_types_allowed=True,
        )

        @classmethod
        def from_texts(
            cls,
            texts: Iterable[str],
            metadatas: Optional[Iterable[dict]] = None,
            ids: Optional[Iterable[str]] = None,
            bm25_params: Optional[Dict[str, Any]] = None,
            preprocess_func: Callable[[str], List[str]] = default_preprocessing_func,
            **kwargs: Any,
        ) -> "SimpleBM25Retriever":
            """
            Create a SimpleBM25Retriever from a list of texts.
            Args:
                texts: A list of texts to vectorize.
                metadatas: A list of metadata dicts to associate with each text.
                ids: A list of ids to associate with each text.
                bm25_params: Parameters to pass to the BM25 vectorizer.
                preprocess_func: A function to preprocess each text before vectorization.
                **kwargs: Any other arguments to pass to the retriever.

            Returns:
                A SimpleBM25Retriever instance.
            """
            texts_list = list(texts)  # Convert iterable to list if needed
            texts_processed = [preprocess_func(t) for t in texts_list]
            bm25_params = bm25_params or {}
            
            # Create custom BM25Okapi vectorizer
            vectorizer = BM25Okapi(texts_processed, **bm25_params)
            
            # Create documents with metadata and ids
            documents = []
            metadatas = metadatas or ({} for _ in texts_list)
            
            if ids:
                documents = [
                    Document(page_content=t, metadata=m, id=i)
                    for t, m, i in zip(texts_list, metadatas, ids)
                ]
            else:
                documents = [
                    Document(page_content=t, metadata=m) 
                    for t, m in zip(texts_list, metadatas)
                ]
                
            return cls(
                vectorizer=vectorizer, 
                docs=documents, 
                preprocess_func=preprocess_func, 
                **kwargs
            )

        @classmethod
        def from_documents(
            cls,
            documents: Iterable[Document],
            *,
            bm25_params: Optional[Dict[str, Any]] = None,
            preprocess_func: Callable[[str], List[str]] = default_preprocessing_func,
            **kwargs: Any,
        ) -> "SimpleBM25Retriever":
            """
            Create a SimpleBM25Retriever from a list of Documents.
            Args:
                documents: A list of Documents to vectorize.
                bm25_params: Parameters to pass to the BM25 vectorizer.
                preprocess_func: A function to preprocess each text before vectorization.
                **kwargs: Any other arguments to pass to the retriever.

            Returns:
                A SimpleBM25Retriever instance.
            """
            documents_list = list(documents)  # Convert iterable to list if needed
            
            # Extract texts, metadatas, and ids from documents
            texts = []
            metadatas = []
            ids = []
            
            for doc in documents_list:
                texts.append(doc.page_content)
                metadatas.append(doc.metadata)
                if hasattr(doc, 'id') and doc.id is not None:
                    ids.append(doc.id)
                else:
                    ids.append(str(uuid.uuid4()))
                    
            return cls.from_texts(
                texts=texts,
                bm25_params=bm25_params,
                metadatas=metadatas,
                ids=ids,
                preprocess_func=preprocess_func,
                **kwargs,
            )
        
        def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
            """Get documents relevant to a query."""
            processed_query = self.preprocess_func(query)
            if self.vectorizer and processed_query:
                return self.vectorizer.get_top_n(processed_query, self.docs, n=self.k)
            return self.docs[:min(self.k, len(self.docs))]
                
        async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
            """Asynchronously get documents relevant to a query."""
            # Async implementation just calls the sync version for simplicity
            return self._get_relevant_documents(query, run_manager=run_manager)
    
    # Replace the standard BM25Retriever with our custom implementation
    BM25Retriever = SimpleBM25Retriever
    HYBRID_SEARCH_AVAILABLE = True
    print("✅ Custom BM25 implementation active - hybrid search enabled")

# Document Loaders & Transformers
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, BSHTMLLoader, TextLoader, UnstructuredURLLoader
)
from langchain_community.document_transformers import Html2TextTransformer

# Add import for image processing
try:
    from PIL import Image
    from io import BytesIO
    IMAGE_PROCESSING_AVAILABLE = True
except ImportError:
    IMAGE_PROCESSING_AVAILABLE = False
    print("PIL not found. Install with: pip install pillow")

# Web Search (Tavily)
try:
    from tavily import AsyncTavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    AsyncTavilyClient = None
    print("Tavily Python SDK not found. Web search will be disabled. Install with: pip install tavily-python")

# Custom local imports
from storage import CloudflareR2Storage

try:
    from langchain_community.chat_message_histories import ChatMessageHistory # Updated import
except ImportError:
    from langchain.memory import ChatMessageHistory # Fallback for older versions, though the target is community

# Add imports for other providers
try:
    import anthropic  # for Claude
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False
    print("Anthropic Python SDK not found. Claude models will be unavailable. Install with: pip install anthropic")

try:
    import google.generativeai as genai  # for Gemini
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Google GenerativeAI SDK not found. Gemini models will be unavailable. Install with: pip install google-generativeai")

try:
    from llama_cpp import Llama  # for Llama models
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    print("llama-cpp-python not found. Llama models will be unavailable. Install with: pip install llama-cpp-python")

try:
    from groq import AsyncGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("Groq Python SDK not found. Install with: pip install groq")

# OpenRouter (Optional)
try:
    # OpenRouter uses the same API format as OpenAI
    from openai import AsyncOpenAI as AsyncOpenRouter
    OPENROUTER_AVAILABLE = True
except ImportError:
    OPENROUTER_AVAILABLE = False
    print("OpenRouter will use OpenAI client for API calls.")

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Vector params for OpenAI's text-embedding-3-small
QDRANT_VECTOR_PARAMS = qdrant_models.VectorParams(size=1536, distance=qdrant_models.Distance.COSINE)
CONTENT_PAYLOAD_KEY = "page_content"
METADATA_PAYLOAD_KEY = "metadata"

if os.name == 'nt':  # Windows
    pass

def create_hybrid_retriever(vector_store: QdrantVectorStore, chunks: List[Document]) -> EnsembleRetriever:
    """Create hybrid retriever combining vector and keyword search."""
    try:
        logger.info("Setting up hybrid retriever...")
        
        # Get vector retriever from the vector store
        vector_retriever = vector_store.as_retriever(
            search_kwargs={"k": 5}  # Retrieve top 5 documents
        )
        
        if HYBRID_SEARCH_AVAILABLE and chunks:
            try:
                # Create BM25 keyword retriever
                logger.info("Creating BM25 keyword retriever...")
                keyword_retriever = BM25Retriever(chunks)
                if hasattr(keyword_retriever, 'k'):
                    keyword_retriever.k = 5  # Retrieve top 5 documents
                
                # Create ensemble retriever with balanced weights
                ensemble_retriever = EnsembleRetriever(
                    retrievers=[vector_retriever, keyword_retriever],
                    weights=[0.3, 0.7]  # Equal weight for vector and keyword search
                )
                
                logger.info("✅ Hybrid retriever created successfully")
                return ensemble_retriever
            except Exception as e_bm25:
                logger.error(f"Error creating BM25 retriever: {e_bm25}")
                logger.warning("⚠️ BM25 initialization failed, using vector search only")
                return vector_retriever
        else:
            logger.warning("⚠️ BM25 not available or no chunks provided, using vector search only")
            return vector_retriever
        
    except Exception as e:
        logger.error(f"Error creating hybrid retriever: {e}")
        logger.info("Falling back to vector search only")
        return vector_store.as_retriever(search_kwargs={"k": 5})

def format_docs(docs: List[Document]) -> str:
    """Format retrieved documents for context."""
    try:
        if not docs:
            return "No relevant documents found."
        
        formatted_docs = []
        for i, doc in enumerate(docs, 1):
            # Clean up the content and add metadata if available
            content = doc.page_content.strip()
            metadata_info = ""
            if hasattr(doc, 'metadata') and doc.metadata:
                # Add source information if available
                if 'source' in doc.metadata:
                    metadata_info = f" (Source: {doc.metadata['source']})"
                elif 'page' in doc.metadata:
                    metadata_info = f" (Page: {doc.metadata['page']})"
            
            formatted_docs.append(f"Document {i}{metadata_info}:\n{content}")
        
        return "\n\n".join(formatted_docs)
    except Exception as e:
        logger.error(f"Error formatting documents: {e}")
        return "Error processing documents."

class EnhancedRAG:
    def __init__(
        self,
        gpt_id: str,
        r2_storage_client: CloudflareR2Storage,
        openai_api_key: str,
        default_llm_model_name: str = "gpt-4o",
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        temp_processing_path: str = "local_rag_data/temp_downloads",
        tavily_api_key: Optional[str] = None,
        default_system_prompt: Optional[str] = None,
        default_temperature: float = 0.5,
        openrouter_api_key: Optional[str] = None,
    ):
        self.gpt_id = gpt_id
        self.r2_storage = r2_storage_client
        self.openai_api_key = openai_api_key
        self.tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
        self.openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        
        self.default_llm_model_name = default_llm_model_name
        self.default_system_prompt = default_system_prompt or (
            "You are a helpful and meticulous AI assistant. "
            "Provide comprehensive, detailed, and accurate answers based *solely* on the context provided. "
            "Structure your response clearly using Markdown. "
            "Use headings (#, ##, ###), subheadings, bullet points (* or -), and numbered lists (1., 2.) where appropriate to improve readability. "
            "For code examples, use Markdown code blocks with language specification (e.g., ```python ... ```). "
            "Feel free to use relevant emojis to make the content more engaging, but do so sparingly and appropriately. "
            "If the context is insufficient or does not contain the answer, clearly state that. "
            "Cite the source of your information if possible (e.g., 'According to document X...'). "
            "Do not make assumptions or use external knowledge beyond the provided context. "
            "Ensure your response is as lengthy and detailed as necessary to fully answer the query, up to the allowed token limit."
        )
        self.default_temperature = default_temperature
        self.max_tokens_llm = 32000  # Maximum for most models, will be overridden by API limits

        self.temp_processing_path = temp_processing_path
        os.makedirs(self.temp_processing_path, exist_ok=True)

        self.embeddings_model = OpenAIEmbeddings(
            api_key=self.openai_api_key,
            model="text-embedding-3-small"  # Explicitly set the model name
        )
        
        # Configure AsyncOpenAI client with custom timeouts
        timeout_config = httpx.Timeout(
            connect=15.0,  # Connection timeout
            read=180.0,    # Read timeout (important for waiting for stream chunks)
            write=15.0,    # Write timeout
            pool=15.0      # Pool timeout
        )
        self.async_openai_client = AsyncOpenAI(
            api_key=self.openai_api_key,
            timeout=timeout_config,
            max_retries=1 # Default is 2, reducing to 1 for faster failure if unrecoverable
        )
        
        # Initialize OpenRouter client
        self.openrouter_client = None
        if self.openrouter_api_key:
            try:
                if OPENROUTER_AVAILABLE:
                    self.openrouter_client = AsyncOpenRouter(
                        api_key=self.openrouter_api_key,
                        base_url="https://openrouter.ai/api/v1",
                        timeout=timeout_config,
                        max_retries=1
                    )
                    logger.info("✅ OpenRouter client initialized")
                else:
                    self.openrouter_client = AsyncOpenAI(
                        api_key=self.openrouter_api_key,
                        base_url="https://openrouter.ai/api/v1",
                        timeout=timeout_config,
                        max_retries=1
                    )
                    logger.info("✅ OpenRouter client initialized using OpenAI client")
            except Exception as e:
                logger.error(f"Error initializing OpenRouter client: {e}")
                logger.warning("⚠️ OpenRouter client initialization failed")

        # Initialize Tavily client if API key is available
        self.tavily_client = None
        if self.tavily_api_key and TAVILY_AVAILABLE:
            try:
                self.tavily_client = AsyncTavilyClient(api_key=self.tavily_api_key)
                logger.info("✅ Tavily web search client initialized")
            except Exception as e:
                logger.error(f"Error initializing Tavily client: {e}")
        
        # Initialize other clients for different LLM providers
        self.anthropic_client = None
        if CLAUDE_AVAILABLE:
            try:
                claude_api_key = os.getenv("ANTHROPIC_API_KEY")
                if claude_api_key:
                    self.anthropic_client = anthropic.AsyncAnthropic(api_key=claude_api_key)
                    logger.info("✅ Claude (Anthropic) client initialized")
            except Exception as e:
                logger.error(f"Error initializing Anthropic client: {e}")

        self.gemini_client = None
        if GEMINI_AVAILABLE:
            try:
                gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                if gemini_api_key:
                    genai.configure(api_key=gemini_api_key)
                    self.gemini_client = genai
                    logger.info("✅ Gemini client initialized")
            except Exception as e:
                logger.error(f"Error initializing Gemini client: {e}")

        self.groq_client = None
        if GROQ_AVAILABLE:
            try:
                groq_api_key = os.getenv("GROQ_API_KEY")
                if groq_api_key:
                    self.groq_client = AsyncGroq(api_key=groq_api_key)
                    logger.info("✅ Groq client initialized")
            except Exception as e:
                logger.error(f"Error initializing Groq client: {e}")

        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")

        if not self.qdrant_url:
            raise ValueError("Qdrant URL must be provided either as a parameter or via QDRANT_URL environment variable.")

        self.qdrant_client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key, timeout=20.0)
        print(f"Qdrant client initialized for URL: {self.qdrant_url}")

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )
        self.html_transformer = Html2TextTransformer()

        self.kb_collection_name = f"kb_{self.gpt_id}".replace("-", "_").lower()
        self.kb_retriever: Optional[BaseRetriever] = None
        self.kb_chunks: List[Document] = []  # Store chunks for hybrid retriever

        self.user_collection_retrievers: Dict[str, BaseRetriever] = {}
        self.user_chunks: Dict[str, List[Document]] = {}  # Store user chunks for hybrid retriever
        self.user_memories: Dict[str, ChatMessageHistory] = {}

    def _get_user_qdrant_collection_name(self, session_id: str) -> str:
        safe_session_id = "".join(c if c.isalnum() else '_' for c in session_id)
        return f"user_{safe_session_id}".replace("-", "_").lower()

    def _ensure_qdrant_collection_exists_sync(self, collection_name: str):
        try:
            self.qdrant_client.get_collection(collection_name=collection_name)
        except Exception as e:
            if "not found" in str(e).lower() or ("status_code=404" in str(e) if hasattr(e, "status_code") else False):
                print(f"Qdrant collection '{collection_name}' not found. Creating...")
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=QDRANT_VECTOR_PARAMS
                )
                print(f"Qdrant collection '{collection_name}' created.")
            else:
                print(f"Error checking/creating Qdrant collection '{collection_name}': {e} (Type: {type(e)})")
                raise

    def _get_qdrant_retriever_sync(self, collection_name: str, chunks: List[Document] = None) -> Optional[BaseRetriever]:
        """Create hybrid retriever with vector store and BM25."""
        try:
            self._ensure_qdrant_collection_exists_sync(collection_name)
            
            # Add this debug statement
            print(f"DEBUG: HYBRID_SEARCH_AVAILABLE={HYBRID_SEARCH_AVAILABLE}, chunks length: {len(chunks) if chunks else 0}")
            
            qdrant_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=collection_name,
                embedding=self.embeddings_model,
                content_payload_key=CONTENT_PAYLOAD_KEY,
                metadata_payload_key=METADATA_PAYLOAD_KEY
            )
            
            if chunks and len(chunks) > 0 and HYBRID_SEARCH_AVAILABLE:
                # Create hybrid retriever with both vector and BM25
                hybrid_retriever = create_hybrid_retriever(qdrant_store, chunks)
                print(f"✅ Hybrid retriever initialized for collection: {collection_name} with {len(chunks)} chunks")
                return hybrid_retriever
            else:
                # Better error reporting
                if not HYBRID_SEARCH_AVAILABLE:
                    print(f"⚠️ BM25 package not available. Using vector-only retriever.")
                elif not chunks or len(chunks) == 0:
                    print(f"⚠️ No chunks provided for BM25. Using vector-only retriever.")
                
                # Fallback to vector only if no chunks available
                vector_retriever = qdrant_store.as_retriever(search_kwargs={'k': 5})
                print(f"⚠️ Vector-only retriever initialized for collection: {collection_name}")
                return vector_retriever
        except Exception as e:
            print(f"Failed to create hybrid retriever for collection '{collection_name}': {e}")
            logger.error(f"Hybrid retriever creation failed: {e}")
            return None
            
    async def _get_user_retriever(self, session_id: str, search_k: int = 5) -> Optional[BaseRetriever]:
        """Get or create hybrid retriever for user session."""
        try:
            collection_name = self._get_user_qdrant_collection_name(session_id)
            
            if session_id not in self.user_collection_retrievers or self.user_collection_retrievers.get(session_id) is None:
                await asyncio.to_thread(self._ensure_qdrant_collection_exists_sync, collection_name)
                
                # Get user chunks for this session
                user_chunks = self.user_chunks.get(session_id, [])
                
                # Create hybrid retriever
                self.user_collection_retrievers[session_id] = self._get_qdrant_retriever_sync(collection_name, user_chunks)
                
                if self.user_collection_retrievers[session_id]:
                    logger.info(f"✅ User hybrid retriever for session '{session_id}' initialized.")
                else:
                    logger.error(f"❌ Failed to initialize user hybrid retriever for session '{session_id}'.")
            
            return self.user_collection_retrievers.get(session_id)
            
        except Exception as e:
            logger.error(f"Error getting user retriever for session '{session_id}': {e}")
            return None

    async def _get_user_memory(self, session_id: str) -> ChatMessageHistory:
        if session_id not in self.user_memories:
            self.user_memories[session_id] = ChatMessageHistory()
            print(f"Initialized new memory for session: {session_id}")
        return self.user_memories[session_id]

    async def _download_and_split_one_doc(self, r2_key_or_url: str) -> List[Document]:
        unique_suffix = uuid.uuid4().hex[:8]
        base_filename = os.path.basename(urlparse(r2_key_or_url).path) or f"doc_{hash(r2_key_or_url)}_{unique_suffix}"
        temp_file_path = os.path.join(self.temp_processing_path, f"{self.gpt_id}_{base_filename}")
        
        loaded_docs: List[Document] = []
        try:
            is_full_url = r2_key_or_url.startswith("http://") or r2_key_or_url.startswith("https://")
            r2_object_key_to_download = ""

            if is_full_url:
                parsed_url = urlparse(r2_key_or_url)
                is_our_r2_url = self.r2_storage.account_id and self.r2_storage.bucket_name and \
                                f"{self.r2_storage.bucket_name}.{self.r2_storage.account_id}.r2.cloudflarestorage.com" in parsed_url.netloc
                if is_our_r2_url:
                    r2_object_key_to_download = parsed_url.path.lstrip('/')
                else:
                    try:
                        loader = UnstructuredURLLoader(urls=[r2_key_or_url], mode="elements", strategy="fast", continue_on_failure=True, show_progress=False)
                        loaded_docs = await asyncio.to_thread(loader.load)
                        if loaded_docs and loaded_docs[0].page_content.startswith("Error fetching URL"): return []
                    except Exception as e_url: print(f"Error UnstructuredURLLoader {r2_key_or_url}: {e_url}"); return []
            else:
                r2_object_key_to_download = r2_key_or_url
            
            if not loaded_docs and r2_object_key_to_download:
                download_success = await asyncio.to_thread(
                    self.r2_storage.download_file, r2_object_key_to_download, temp_file_path
                )
                if not download_success: print(f"Failed R2 download: {r2_object_key_to_download}"); return []

                _, ext = os.path.splitext(temp_file_path); ext = ext.lower()
                
                print(f"Processing file with extension: {ext}")
                
                # Check if it's an image file by extension
                is_image = ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
                if is_image:
                    print(f"Detected image file: {temp_file_path}")
                    
                    # Try to process the image with multiple approaches
                    try:
                        # Read the image file as bytes
                        with open(temp_file_path, 'rb') as img_file:
                            image_data = img_file.read()
                        
                        # First attempt: Use the current model with vision capabilities
                        print(f"Processing image using {self.default_llm_model_name} vision capabilities...")
                        image_content = await self._process_image_with_vision(image_data)
                        
                        if image_content:
                            # Create a document from the description
                            doc = Document(
                                page_content=image_content,
                                metadata={
                                    "source": r2_key_or_url,
                                    "file_type": "image",
                                    "content_source": "vision_api"
                                }
                            )
                            loaded_docs = [doc]
                            print(f"Successfully processed image with vision API, extracted {len(image_content)} characters")
                        else:
                            # Create a default document if no content was extracted
                            doc = Document(
                                page_content="[This is an image file that couldn't be processed. Please ask specific questions about its content.]",
                                metadata={"source": r2_key_or_url, "file_type": "image"}
                            )
                            loaded_docs = [doc]
                    except Exception as e_img:
                        print(f"Image processing failed: {e_img}")
                        # Create a fallback document
                        doc = Document(
                            page_content="[This is an image file that could not be processed. Error: " + str(e_img) + "]",
                            metadata={"source": r2_key_or_url, "file_type": "image", "processing_error": str(e_img)}
                        )
                        loaded_docs = [doc]
                else:
                    # Handle regular document types
                    loader = None
                    try:
                        if ext == ".pdf": 
                            loader = PyPDFLoader(temp_file_path)
                        elif ext == ".docx": 
                            loader = Docx2txtLoader(temp_file_path)
                        elif ext in [".html", ".htm"]: 
                            loader = BSHTMLLoader(temp_file_path, open_encoding='utf-8')
                        else: 
                            loader = TextLoader(temp_file_path, autodetect_encoding=True)
                        
                        loaded_docs = await asyncio.to_thread(loader.load)
                        if ext in [".html", ".htm"] and loaded_docs:
                            loaded_docs = self.html_transformer.transform_documents(loaded_docs)
                    except Exception as e_load:
                        print(f"Error loading document: {e_load}")
                        return []
            
            if loaded_docs:
                for doc in loaded_docs:
                    doc.metadata["source"] = r2_key_or_url 
                return self.text_splitter.split_documents(loaded_docs)
            return []
        except Exception as e:
            print(f"Error processing source '{r2_key_or_url}': {e}")
            return []
        finally:
            if os.path.exists(temp_file_path):
                try: os.remove(temp_file_path)
                except Exception as e_del: print(f"Error deleting temp file {temp_file_path}: {e_del}")

    async def _process_image_with_vision(self, image_data: bytes) -> str:
        """Process an image using the user's chosen model with vision capabilities"""
        try:
            # Convert image to base64
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # Original model name selected by the user
            user_selected_model_name_lower = self.default_llm_model_name.lower()
            
            # 1. Gemini models
            if "gemini" in user_selected_model_name_lower and GEMINI_AVAILABLE and self.gemini_client:
                print(f"Using {self.default_llm_model_name} for image processing via Gemini")
                gemini_api_name = "gemini-1.5-pro" # Default vision model for Gemini
                try:
                    if "flash" in user_selected_model_name_lower:
                        gemini_api_name = "gemini-1.5-flash"
                    # (No other specific Gemini model name checks needed, defaults to 1.5-pro for vision)

                    image_parts = [{"mime_type": "image/jpeg", "data": base64_image}]
                    prompt_text = "Describe the content of this image in detail, including any visible text."
                    
                    api_model_to_call = self.gemini_client.GenerativeModel(gemini_api_name)
                    response = await api_model_to_call.generate_content_async(contents=[prompt_text] + image_parts)
                    
                    if hasattr(response, "text") and response.text:
                        return f"Image Content ({gemini_api_name} Analysis):\n{response.text}"
                    else:
                        error_message_from_response = "No text content in response"
                        if hasattr(response, 'prompt_feedback') and response.prompt_feedback and response.prompt_feedback.block_reason:
                            error_message_from_response = f"Blocked: {getattr(response.prompt_feedback, 'block_reason_message', '') or response.prompt_feedback.block_reason}"
                        elif hasattr(response, 'candidates') and response.candidates and response.candidates[0].finish_reason != 'STOP':
                            error_message_from_response = f"Finished with reason: {response.candidates[0].finish_reason}"
                        raise Exception(f"Gemini Vision ({gemini_api_name}) processing issue: {error_message_from_response}")

                except Exception as e_gemini:
                    resolved_gemini_api_name = gemini_api_name if 'gemini_api_name' in locals() else 'N/A'
                    print(f"Error with Gemini Vision (input: {self.default_llm_model_name} -> attempted: {resolved_gemini_api_name}): {e_gemini}")
                    raise Exception(f"Gemini Vision processing failed: {e_gemini}")
            
            # 2. OpenAI models (GPT-4o, GPT-4o-mini, GPT-4-vision)
            elif "gpt-" in user_selected_model_name_lower:
                openai_model_to_call = self.default_llm_model_name # Default to user selected
                if user_selected_model_name_lower == "gpt-4o-mini":
                    openai_model_to_call = "gpt-4o" # Use gpt-4o for gpt-4o-mini's vision tasks
                    print(f"Using gpt-4o for image processing (selected: {self.default_llm_model_name})")
                else:
                    print(f"Using {self.default_llm_model_name} for image processing")
                
                try:
                    response = await self.async_openai_client.chat.completions.create(
                        model=openai_model_to_call,
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Describe the content of this image in detail, including any visible text."},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                            ]
                        }]
                    )
                    return f"Image Content ({openai_model_to_call} Analysis):\n{response.choices[0].message.content}"
                except Exception as e_openai:
                    print(f"Error with OpenAI Vision ({openai_model_to_call}): {e_openai}")
                    raise Exception(f"OpenAI Vision processing failed: {e_openai}")
            
            # 3. Claude models
            elif "claude" in user_selected_model_name_lower and CLAUDE_AVAILABLE and self.anthropic_client:
                print(f"Using {self.default_llm_model_name} for image processing")
                try:
                    claude_model_to_call = "claude-3-5-sonnet-20240620" # Default to Claude 3.5 Sonnet
                    if "opus" in user_selected_model_name_lower:
                        claude_model_to_call = "claude-3-opus-20240229"
                    # No need to check for "3-5" in sonnet/haiku explicitly, direct model names are better
                    elif "claude-3-sonnet" in user_selected_model_name_lower: # Catches "claude-3-sonnet-20240229"
                         claude_model_to_call = "claude-3-sonnet-20240229"
                    elif "claude-3-haiku" in user_selected_model_name_lower: # Catches "claude-3-haiku-20240307"
                         claude_model_to_call = "claude-3-haiku-20240307"
                    # Specific checks for 3.5 models to ensure correct IDs
                    elif "claude-3.5-sonnet" in user_selected_model_name_lower:
                        claude_model_to_call = "claude-3.5-sonnet-20240620"
                    elif "claude-3.5-haiku" in user_selected_model_name_lower:
                         claude_model_to_call = "claude-3.5-haiku-20240307" # Assuming this is the correct ID from Anthropic docs

                    response = await self.anthropic_client.messages.create(
                        model=claude_model_to_call,
                        messages=[{
                            "role": "user", 
                            "content": [
                                {"type": "text", "text": "Describe the content of this image in detail, including any visible text."},
                                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64_image}}
                            ]
                        }]
                    )
                    return f"Image Content ({claude_model_to_call} Analysis):\n{response.content[0].text}"
                except Exception as e_claude:
                    print(f"Error with Claude Vision: {e_claude}")
                    raise Exception(f"Claude Vision processing failed: {e_claude}")
            
            # 4. Llama models (via Groq)
            elif "llama" in user_selected_model_name_lower and GROQ_AVAILABLE and self.groq_client:
                print(f"Processing Llama model {self.default_llm_model_name} for image via Groq")
                try:
                    groq_model_to_call = None
                    # More robust matching for Llama 4 Scout and Maverick
                    if "llama" in user_selected_model_name_lower and "4" in user_selected_model_name_lower and "scout" in user_selected_model_name_lower:
                        groq_model_to_call = "meta-llama/llama-4-scout-17b-16e-instruct"
                    elif "llama" in user_selected_model_name_lower and "4" in user_selected_model_name_lower and "maverick" in user_selected_model_name_lower:
                        groq_model_to_call = "meta-llama/llama-4-maverick-17b-128e-instruct"
                    elif "llava" in user_selected_model_name_lower: # For models like "llava-v1.5-7b"
                        groq_model_to_call = "llava-v1.5-7b-4096-preview"
                    elif "llama3" in user_selected_model_name_lower or "llama-3" in user_selected_model_name_lower:
                        # Llama 3 models on Groq do not support vision. This is an explicit failure.
                        raise Exception(f"The selected Llama 3 model ({self.default_llm_model_name}) does not support vision capabilities on Groq. Please choose a Llama 4 or LLaVA model for vision.")
                    else:
                        # Fallback for other Llama models not explicitly listed for vision
                        raise Exception(f"No configured vision-capable Llama model on Groq for '{self.default_llm_model_name}'. Supported for vision are Llama 4 Scout/Maverick and LLaVA.")

                    print(f"Attempting to use Groq vision model: {groq_model_to_call}")
                    
                    messages_for_groq = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Describe the content of this image in detail, including any visible text."},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                            ]
                        }
                    ]
                    if self.default_system_prompt:
                        messages_for_groq.insert(0, {"role": "system", "content": "You are an AI assistant that accurately describes images."})

                    response = await self.groq_client.chat.completions.create(
                        model=groq_model_to_call,
                        messages=messages_for_groq,
                        temperature=0.2,
                        stream=False
                    )
                    return f"Image Content ({groq_model_to_call} Analysis via Groq):\n{response.choices[0].message.content}"
                except Exception as e_llama_groq:
                    print(f"Error with Llama Vision through Groq (Model: {self.default_llm_model_name}): {e_llama_groq}")
                    raise Exception(f"Llama Vision processing failed: {e_llama_groq}")
            
            # If model doesn't match any of the known vision-capable types
            raise Exception(f"Model {self.default_llm_model_name} doesn't have a configured vision capability handler or required SDKs are not available.")
        except Exception as e:
            print(f"Error using Vision API: {e}")
            # Basic image properties fallback
            try:
                img = Image.open(BytesIO(image_data))
                width, height = img.size
                format_type = img.format
                mode = img.mode
                return f"[Image file: {width}x{height} {format_type} in {mode} mode. Vision processing failed with error: {str(e)}]"
            except Exception as e_img:
                return "[Image file detected but couldn't be processed. Vision API error: " + str(e) + "]"

    async def _index_documents_to_qdrant_batch(self, docs_to_index: List[Document], collection_name: str):
        """Index documents to Qdrant and store chunks for hybrid retriever."""
        if not docs_to_index: 
            return

        try:
            await asyncio.to_thread(self._ensure_qdrant_collection_exists_sync, collection_name)
            qdrant_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=collection_name,
                embedding=self.embeddings_model,
                content_payload_key=CONTENT_PAYLOAD_KEY,
                metadata_payload_key=METADATA_PAYLOAD_KEY
            )
            
            logger.info(f"Adding {len(docs_to_index)} document splits to Qdrant collection '{collection_name}'...")
            await asyncio.to_thread(
                qdrant_store.add_documents,
                documents=docs_to_index,
                batch_size=100
            )
            
            logger.info(f"✅ Successfully added/updated {len(docs_to_index)} splits in Qdrant collection '{collection_name}'.")
            
            # Store chunks for hybrid retriever
            if collection_name == self.kb_collection_name:
                self.kb_chunks.extend(docs_to_index)
                logger.info(f"Stored {len(docs_to_index)} chunks for KB hybrid retriever")
                
        except Exception as e:
            logger.error(f"Error adding documents to Qdrant collection '{collection_name}': {e}")
            raise

    async def update_knowledge_base_from_r2(self, r2_keys_or_urls: List[str]):
        """Update knowledge base with hybrid search capability."""
        try:
            logger.info(f"Updating KB for gpt_id '{self.gpt_id}' with {len(r2_keys_or_urls)} R2 documents...")
            
            processing_tasks = [self._download_and_split_one_doc(key_or_url) for key_or_url in r2_keys_or_urls]
            results_list_of_splits = await asyncio.gather(*processing_tasks)
            all_splits: List[Document] = [split for sublist in results_list_of_splits for split in sublist]

            if not all_splits:
                logger.warning(f"No content extracted from R2 sources for KB collection {self.kb_collection_name}.")
                return

            await self._index_documents_to_qdrant_batch(all_splits, self.kb_collection_name)
            
            # Create new hybrid retriever with updated chunks
            self.kb_retriever = self._get_qdrant_retriever_sync(self.kb_collection_name, self.kb_chunks)
            
            logger.info(f"✅ Knowledge Base for gpt_id '{self.gpt_id}' update process finished with hybrid search.")
            
        except Exception as e:
            logger.error(f"Error updating knowledge base: {e}")
            raise

    async def update_user_documents_from_r2(self, session_id: str, r2_keys_or_urls: List[str]):
        """Update user documents with hybrid search capability."""
        try:
            # Clear existing documents and retriever for this user session first
            logger.info(f"Clearing existing user-specific context for session '{session_id}' before update...")
            await self.clear_user_session_context(session_id)

            user_collection_name = self._get_user_qdrant_collection_name(session_id)
            logger.info(f"Updating user documents for session '{session_id}' with {len(r2_keys_or_urls)} R2 docs...")
            
            processing_tasks = [self._download_and_split_one_doc(key_or_url) for key_or_url in r2_keys_or_urls]
            results_list_of_splits = await asyncio.gather(*processing_tasks)
            all_splits: List[Document] = [split for sublist in results_list_of_splits for split in sublist]

            if not all_splits:
                logger.warning(f"No content extracted from R2 sources for user collection {user_collection_name}.")
                return

            await self._index_documents_to_qdrant_batch(all_splits, user_collection_name)
            
            # Store user chunks for hybrid retriever
            self.user_chunks[session_id] = all_splits
            
            # Create new hybrid retriever for the session
            self.user_collection_retrievers[session_id] = self._get_qdrant_retriever_sync(user_collection_name, all_splits)
            
            logger.info(f"✅ User documents for session '{session_id}' update process finished with hybrid search.")
            
        except Exception as e:
            logger.error(f"Error updating user documents for session '{session_id}': {e}")
            raise

    async def _get_retrieved_documents(
        self, 
        retriever: Optional[BaseRetriever], 
        query: str, 
        k_val: int = 5
    ) -> List[Document]:
        """Simplified document retrieval using hybrid retriever."""
        try:
            if not retriever:
                logger.warning("No retriever available for document retrieval")
                return []
            
            # Update search parameters if supported
            if hasattr(retriever, 'search_kwargs'):
                retriever.search_kwargs['k'] = k_val
            elif hasattr(retriever, 'retrievers'):
                # For EnsembleRetriever, update both sub-retrievers
                for sub_retriever in retriever.retrievers:
                    if hasattr(sub_retriever, 'search_kwargs'):
                        sub_retriever.search_kwargs['k'] = k_val
                    elif hasattr(sub_retriever, 'k'):
                        sub_retriever.k = k_val
            
            # Retrieve documents using hybrid search
            if hasattr(retriever, 'ainvoke'):
                docs = await retriever.ainvoke(query)
            else:
                docs = await asyncio.to_thread(retriever.invoke, query)
            
            logger.info(f"Retrieved {len(docs)} documents using hybrid search")
            return docs[:k_val]  # Ensure we don't exceed requested count
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []

    def _format_docs_for_llm_context(self, documents: List[Document], source_name: str) -> str:
        """Enhanced document formatting for LLM context."""
        if not documents: 
            return ""
        
        try:
            formatted_sections = []
            web_docs = []
            other_docs = []
            
            for doc in documents:
                source_type = doc.metadata.get("source_type", "")
                if source_type == "web_search" or "Web Search" in doc.metadata.get("source", ""):
                    web_docs.append(doc)
                else:
                    other_docs.append(doc)
            
            # Process web search documents first
            if web_docs:
                formatted_sections.append("## 🌐 WEB SEARCH RESULTS")
                for doc in web_docs:
                    source = doc.metadata.get('source', source_name)
                    title = doc.metadata.get('title', '')
                    url = doc.metadata.get('url', '')
                    
                    header = f"📰 **WEB SOURCE: {title}**"
                    if url: 
                        header += f"\n🔗 **URL: {url}**"
                    
                    formatted_sections.append(f"{header}\n\n{doc.page_content}")
            
            # Process other documents
            if other_docs:
                if web_docs:  
                    formatted_sections.append("## 📚 KNOWLEDGE BASE & USER DOCUMENTS")
                
                for doc in other_docs:
                    source = doc.metadata.get('source', source_name)
                    score = f"Score: {doc.metadata.get('score', 'N/A'):.2f}" if 'score' in doc.metadata else ""
                    title = doc.metadata.get('title', '')
                    
                    if "user" in source.lower():
                        header = f"📄 **USER DOCUMENT: {source}**"
                    else:
                        header = f"📚 **KNOWLEDGE BASE: {source}**"
                        
                    if title: 
                        header += f" - **{title}**"
                    if score: 
                        header += f" - {score}"
                    
                    formatted_sections.append(f"{header}\n\n{doc.page_content}")
            
            return "\n\n---\n\n".join(formatted_sections)
            
        except Exception as e:
            logger.error(f"Error formatting documents: {e}")
            return "Error processing documents for context."

    async def _get_web_search_docs(self, query: str, enable_web_search: bool, num_results: int = 3) -> List[Document]:
        if not enable_web_search or not self.tavily_client: 
            print(f"🌐 Web search is DISABLED for this query.")
            return []
        
        print(f"🌐 Web search is ENABLED. Searching web for: '{query}'")
        try:
            search_response = await self.tavily_client.search(
                query=query, 
                search_depth="advanced", # Changed from "basic" to "advanced" for more comprehensive search
                max_results=num_results,
                include_raw_content=True,
                include_domains=[]  # Can be customized to limit to specific domains
            )
            results = search_response.get("results", [])
            web_docs = []
            if results:
                print(f"🌐 Web search returned {len(results)} results")
                for i, res in enumerate(results):
                    content_text = res.get("raw_content") or res.get("content", "")
                    title = res.get("title", "N/A")
                    url = res.get("url", "N/A")
                    
                    if content_text:
                        print(f"🌐 Web result #{i+1}: '{title}' - {url[:60]}...")
                        web_docs.append(Document(
                            page_content=content_text[:4000],
                            metadata={
                                "source": f"Web Search: {title}",
                                "source_type": "web_search", 
                                "title": title, 
                                "url": url
                            }
                        ))
            return web_docs
        except Exception as e: 
            print(f"❌ Error during web search: {e}")
            return []
            
    async def _generate_llm_response(
        self, session_id: str, query: str, all_context_docs: List[Document],
        chat_history_messages: List[Dict[str, str]], llm_model_name_override: Optional[str],
        system_prompt_override: Optional[str], stream: bool = False
    ) -> Union[AsyncGenerator[str, None], str]:
        current_llm_model = llm_model_name_override or self.default_llm_model_name
        
        # Normalize model names for consistent matching
        normalized_model = current_llm_model.lower().strip()
        
        # Convert variations to canonical model names
        if "llama 4" in normalized_model or "llama-4" in normalized_model:
            current_llm_model = "meta-llama/llama-4-scout-17b-16e-instruct"
        elif "llama" in normalized_model and "3" in normalized_model:
            current_llm_model = "llama3-8b-8192"
        elif "gemini" in normalized_model and "flash" in normalized_model:
            current_llm_model = "gemini-flash-2.5"
        elif "gemini" in normalized_model and "pro" in normalized_model:
            current_llm_model = "gemini-pro-2.5"
        elif "claude" in normalized_model:
            current_llm_model = "claude-3.5-haiku-20240307"  # Use exact model ID with version
        elif normalized_model == "gpt-4o" or normalized_model == "gpt-4o-mini":
            current_llm_model = normalized_model  # Keep as is for OpenAI models
        
        current_system_prompt = system_prompt_override or self.default_system_prompt
        
        # Format context and query
        context_str = self._format_docs_for_llm_context(all_context_docs, "Retrieved Context")
        if not context_str.strip():
            context_str = "No relevant context could be found from any available source for this query. Please ensure documents are uploaded and relevant to your question."

        # Prepare user message
        user_query_message_content = (
            f"📚 **CONTEXT:**\n{context_str}\n\n"
            f"Based on the above context and any relevant chat history, provide a detailed, well-structured response to this query:\n\n"
            f"**QUERY:** {query}\n\n"
            f"Requirements for your response:\n"
            f"1. 🎯 Start with a relevant emoji and descriptive headline\n"
            f"2. 📋 Organize with clear headings and subheadings\n"
            f"3. 📊 Include bullet points or numbered lists where appropriate\n"
            f"4. 💡 Highlight key insights or important information\n"
            f"5. 📝 Reference specific information from the provided documents\n"
            f"6. 🔍 Use appropriate emojis (about 1-2 per section) to make content engaging\n"
            f"7. 📚 Make your response comprehensive, detailed and precise\n"
        )

        messages = [{"role": "system", "content": current_system_prompt}]
        messages.extend(chat_history_messages)
        messages.append({"role": "user", "content": user_query_message_content})

        user_memory = await self._get_user_memory(session_id)
        
        # Check if it's an OpenRouter model (various model names supported by OpenRouter)
        use_openrouter = (self.openrouter_client is not None and 
                         (normalized_model.startswith("openai/") or 
                          normalized_model.startswith("anthropic/") or
                          normalized_model.startswith("meta-llama/") or
                          normalized_model.startswith("google/") or
                          normalized_model.startswith("mistral/") or
                          "openrouter" in normalized_model))

        # Special case: Handle router-engine and OpenRouter routing models
        if normalized_model == "router-engine" or normalized_model.startswith("openrouter/"):
            if normalized_model == "router-engine":
                print(f"Converting 'router-engine' to 'openrouter/auto' for OpenRouter routing")
                current_llm_model = "openrouter/auto"  # Use OpenRouter's auto-routing
            # If it already starts with "openrouter/", keep it as is
            use_openrouter = True

        if use_openrouter:
            # Implementation for OpenRouter models (stream and non-stream)
            if stream:
                async def openrouter_stream_generator():
                    full_response_content = ""
                    try:
                        response_stream = await self.openrouter_client.chat.completions.create(
                            model=current_llm_model, 
                            messages=messages, 
                            temperature=self.default_temperature,
                            stream=True
                        )
                        
                        async for chunk in response_stream:
                            content_piece = chunk.choices[0].delta.content
                            if content_piece:
                                full_response_content += content_piece
                                yield content_piece
                    except Exception as e_stream:
                        print(f"OpenRouter streaming error: {e_stream}")
                        yield f"I apologize, but I couldn't process your request successfully with OpenRouter. Please try asking in a different way."
                    finally:
                        await asyncio.to_thread(user_memory.add_user_message, query)
                        await asyncio.to_thread(user_memory.add_ai_message, full_response_content)
                return openrouter_stream_generator()
            else:
                response_content = ""
                try:
                    completion = await self.openrouter_client.chat.completions.create(
                        model=current_llm_model, 
                        messages=messages, 
                        temperature=self.default_temperature,
                        stream=False
                    )
                    response_content = completion.choices[0].message.content or ""
                except Exception as e_nostream:
                    print(f"OpenRouter non-streaming error: {e_nostream}")
                    response_content = f"Error with OpenRouter: {str(e_nostream)}"
                
                await asyncio.to_thread(user_memory.add_user_message, query)
                await asyncio.to_thread(user_memory.add_ai_message, response_content)
                return response_content
        
        # GPT-4o or GPT-4o-mini models (OpenAI)
        if current_llm_model.startswith("gpt-"):
            # Implementation for OpenAI models (stream and non-stream)
            if stream:
                async def stream_generator():
                    full_response_content = ""
                    try:
                        response_stream = await self.async_openai_client.chat.completions.create(
                            model=current_llm_model, 
                            messages=messages, 
                            temperature=self.default_temperature,
                            stream=True
                        )
                        
                        async for chunk in response_stream:
                            content_piece = chunk.choices[0].delta.content
                            if content_piece:
                                full_response_content += content_piece
                                yield content_piece
                    except Exception as e_stream:
                        print(f"Error during streaming: {e_stream}")
                        yield f"I apologize, but I couldn't process your request successfully. Please try asking in a different way."
                    finally:
                        await asyncio.to_thread(user_memory.add_user_message, query)
                        await asyncio.to_thread(user_memory.add_ai_message, full_response_content)
                return stream_generator()
            else:
                response_content = ""
                try:
                    completion = await self.async_openai_client.chat.completions.create(
                        model=current_llm_model, messages=messages, temperature=self.default_temperature,
                        stream=False
                    )
                    response_content = completion.choices[0].message.content or ""
                except Exception as e_nostream:
                    print(f"LLM non-streaming error: {e_nostream}")
                    response_content = f"Error: {str(e_nostream)}"
                
                await asyncio.to_thread(user_memory.add_user_message, query)
                await asyncio.to_thread(user_memory.add_ai_message, response_content)
                return response_content
        
        # Claude 3.5 Haiku
        elif current_llm_model.startswith("claude") and CLAUDE_AVAILABLE and self.anthropic_client:
            if stream:
                async def claude_stream_generator():
                    full_response_content = ""
                    try:
                        system_content = current_system_prompt
                        claude_messages = []
                        
                        for msg in chat_history_messages:
                            if msg["role"] != "system":
                                claude_messages.append(msg)
                        
                        claude_messages.append({"role": "user", "content": user_query_message_content})
                        
                        # Use the updated Claude model
                        response_stream = await self.anthropic_client.messages.create(
                            model="claude-3.5-haiku-20240307",  # Use the exact model ID including version
                            system=system_content,
                            messages=claude_messages,
                            stream=True,
                            max_tokens=4000
                        )
                        
                        async for chunk in response_stream:
                            if chunk.type == "content_block_delta" and chunk.delta.text:
                                content_piece = chunk.delta.text
                                full_response_content += content_piece
                                yield content_piece
                                
                    except Exception as e_stream:
                        print(f"Claude streaming error: {e_stream}")
                        yield "I apologize, but I couldn't process your request successfully. Please try again later."
                    finally:
                        await asyncio.to_thread(user_memory.add_user_message, query)
                        await asyncio.to_thread(user_memory.add_ai_message, full_response_content)
                return claude_stream_generator()
            else:
                # Non-streaming Claude implementation
                response_content = ""
                try:
                    system_content = current_system_prompt
                    claude_messages = []
                    
                    for msg in chat_history_messages:
                        if msg["role"] != "system":
                            claude_messages.append(msg)
                    
                    claude_messages.append({"role": "user", "content": user_query_message_content})
                    
                    # Use the updated Claude model
                    response = await self.anthropic_client.messages.create(
                        model="claude-3.5-haiku-20240307",  # Use the exact model ID including version
                        system=system_content,
                        messages=claude_messages,
                        max_tokens=4000
                    )
                    response_content = response.content[0].text
                except Exception as e_nostream:
                    print(f"Claude non-streaming error: {e_nostream}")
                    response_content = f"Error: {str(e_nostream)}"
                
                await asyncio.to_thread(user_memory.add_user_message, query)
                await asyncio.to_thread(user_memory.add_ai_message, response_content)
                return response_content
        
        # Gemini models (flash-2.5 and pro-2.5)
        elif current_llm_model.startswith("gemini") and GEMINI_AVAILABLE and self.gemini_client:
            if stream:
                async def gemini_stream_generator():
                    full_response_content = ""
                    try:
                        # Convert messages to Gemini format
                        gemini_messages = []
                        for msg in messages:
                            if msg["role"] == "system":
                                continue
                            elif msg["role"] == "user":
                                gemini_messages.append({"role": "user", "parts": [{"text": msg["content"]}]})
                            elif msg["role"] == "assistant":
                                gemini_messages.append({"role": "model", "parts": [{"text": msg["content"]}]})
                        
                        # Add system message to first user message if needed
                        if messages[0]["role"] == "system" and len(gemini_messages) > 0:
                            for i, msg in enumerate(gemini_messages):
                                if msg["role"] == "user" and (not msg["parts"] or not msg["parts"][0].get("text")):
                                    msg["parts"][0]["text"] = "Please provide information based on the context."
                                    break
                        
                        # Map to the specific Gemini model version with exact identifiers
                        gemini_model_name = current_llm_model
                        if current_llm_model == "gemini-flash-2.5":
                            gemini_model_name = "gemini-2.5-flash-preview-04-17"
                        elif current_llm_model == "gemini-pro-2.5":
                            gemini_model_name = "gemini-2.5-pro-preview-05-06"
                            
                        model = self.gemini_client.GenerativeModel(model_name=gemini_model_name)
                        
                        response_stream = await model.generate_content_async(
                            gemini_messages,
                            generation_config={"temperature": self.default_temperature},
                            stream=True
                        )
                        
                        async for chunk in response_stream:
                            if hasattr(chunk, "text"):
                                content_piece = chunk.text
                                if content_piece:
                                    full_response_content += content_piece
                                    yield content_piece
                        
                    except Exception as e_stream:
                        print(f"Gemini streaming error: {e_stream}")
                        if "429" in str(e_stream) and "quota" in str(e_stream).lower():
                            yield "I apologize, but the Gemini service is currently rate limited. The system will automatically fall back to GPT-4o."
                            # Fall back to GPT-4o silently
                            try:
                                response_stream = await self.async_openai_client.chat.completions.create(
                                    model="gpt-4o", 
                                    messages=messages, 
                                    temperature=self.default_temperature,
                                    stream=True
                                )
                                
                                async for chunk in response_stream:
                                    content_piece = chunk.choices[0].delta.content
                                    if content_piece:
                                        full_response_content += content_piece
                                        yield content_piece
                            except Exception as fallback_e:
                                print(f"Gemini fallback error: {fallback_e}")
                                yield "I apologize, but I couldn't process your request successfully. Please try again later."
                        else:
                            yield "I apologize, but I couldn't process your request successfully. Please try again later."
                    finally:
                        await asyncio.to_thread(user_memory.add_user_message, query)
                        await asyncio.to_thread(user_memory.add_ai_message, full_response_content)
                return gemini_stream_generator()
            else:
                # Non-streaming Gemini implementation
                response_content = ""
                try:
                    # Convert messages to Gemini format
                    gemini_messages = []
                    for msg in messages:
                        if msg["role"] == "system":
                            continue
                        elif msg["role"] == "user":
                            gemini_messages.append({"role": "user", "parts": [{"text": msg["content"]}]})
                        elif msg["role"] == "assistant":
                            gemini_messages.append({"role": "model", "parts": [{"text": msg["content"]}]})
                    
                    # Add system message to first user message if needed
                    if messages[0]["role"] == "system" and len(gemini_messages) > 0:
                        for i, msg in enumerate(gemini_messages):
                            if msg["role"] == "user" and (not msg["parts"] or not msg["parts"][0].get("text")):
                                msg["parts"][0]["text"] = "Please provide information based on the context."
                                break
                    
                    # Map to the specific Gemini model version with exact identifiers
                    gemini_model_name = current_llm_model
                    if current_llm_model == "gemini-flash-2.5":
                        gemini_model_name = "gemini-2.5-flash-preview-04-17"
                    elif current_llm_model == "gemini-pro-2.5":
                        gemini_model_name = "gemini-2.5-pro-preview-05-06"
                    
                    model = self.gemini_client.GenerativeModel(model_name=gemini_model_name)
                    response = await model.generate_content_async(
                        gemini_messages,
                        generation_config={"temperature": self.default_temperature}
                    )
                    
                    if hasattr(response, "text"):
                        response_content = response.text
                    else:
                        response_content = "Error: Could not generate response from Gemini."
                except Exception as e_nostream:
                    print(f"Gemini non-streaming error: {e_nostream}")
                    response_content = f"Error: {str(e_nostream)}"
                
                await asyncio.to_thread(user_memory.add_user_message, query)
                await asyncio.to_thread(user_memory.add_ai_message, response_content)
                return response_content
        
        # Llama models (Llama 3 and Llama 4 Scout via Groq)
        elif (current_llm_model.startswith("llama") or current_llm_model.startswith("meta-llama/")) and GROQ_AVAILABLE and self.groq_client:
            # Map to the correct Llama model with vision capabilities
            if "4" in current_llm_model.lower() or "llama-4" in current_llm_model.lower() or current_llm_model.startswith("meta-llama/llama-4"):
                # Use a model that actually exists in Groq as fallback
                groq_model = "llama3-70b-8192"  # Higher quality Llama model available on Groq
                print(f"Using Groq with llama3-70b-8192 model (as fallback for Llama 4 Scout)")
            else:
                groq_model = "llama3-8b-8192"  # Keep default for Llama 3
                print(f"Using Groq with llama3-8b-8192 model")
            
            if stream:
                async def groq_stream_generator():
                    full_response_content = ""
                    try:
                        groq_messages = [{"role": "system", "content": current_system_prompt}]
                        groq_messages.extend(chat_history_messages)
                        groq_messages.append({"role": "user", "content": user_query_message_content})
                        
                        response_stream = await self.groq_client.chat.completions.create(
                            model=groq_model,
                            messages=groq_messages,
                            temperature=self.default_temperature,
                            stream=True
                        )
                        
                        async for chunk in response_stream:
                            content_piece = chunk.choices[0].delta.content
                            if content_piece:
                                full_response_content += content_piece
                                yield content_piece
                
                    except Exception as e_stream:
                        print(f"Groq streaming error: {e_stream}")
                        yield "I apologize, but I couldn't process your request successfully. Please try again later."
                    finally:
                        await asyncio.to_thread(user_memory.add_user_message, query)
                        await asyncio.to_thread(user_memory.add_ai_message, full_response_content)
                return groq_stream_generator()
            else:
                # Non-streaming Groq implementation
                response_content = ""
                try:
                    groq_messages = [{"role": "system", "content": current_system_prompt}]
                    groq_messages.extend(chat_history_messages)
                    groq_messages.append({"role": "user", "content": user_query_message_content})
                    
                    completion = await self.groq_client.chat.completions.create(
                        model=groq_model,
                        messages=groq_messages,
                        temperature=self.default_temperature,
                        stream=False
                    )
                    
                    response_content = completion.choices[0].message.content or ""
                except Exception as e_nostream:
                    print(f"Groq non-streaming error: {e_nostream}")
                    response_content = f"Error: {str(e_nostream)}"
                
                await asyncio.to_thread(user_memory.add_user_message, query)
                await asyncio.to_thread(user_memory.add_ai_message, response_content)
                return response_content
        
        # Fallback to GPT-4o when model not recognized
        else:
            print(f"Model {current_llm_model} not recognized. Falling back to gpt-4o.")
            fallback_model = "gpt-4o"
            
            # If streaming is requested, we must return a generator
            if stream:
                async def fallback_stream_generator():
                    full_response_content = ""
                    try:
                        completion = await self.async_openai_client.chat.completions.create(
                            model=fallback_model, 
                            messages=messages, 
                            temperature=self.default_temperature,
                            stream=True  # Important: use streaming for streaming requests
                        )
                        
                        async for chunk in completion:
                            content_piece = chunk.choices[0].delta.content
                            if content_piece:
                                full_response_content += content_piece
                                yield content_piece
                    except Exception as e_stream:
                        print(f"Fallback model streaming error: {e_stream}")
                        yield f"I apologize, but I couldn't process your request successfully. Please try asking in a different way."
                    finally:
                        await asyncio.to_thread(user_memory.add_user_message, query)
                        await asyncio.to_thread(user_memory.add_ai_message, full_response_content)
                return fallback_stream_generator()
            else:
                # Non-streaming fallback implementation
                try:
                    completion = await self.async_openai_client.chat.completions.create(
                        model=fallback_model, 
                        messages=messages, 
                        temperature=self.default_temperature,
                        stream=False
                    )
                    response_content = completion.choices[0].message.content or ""
                except Exception as e_fallback:
                    print(f"Fallback model error: {e_fallback}")
                    response_content = "I apologize, but I couldn't process your request with the requested model. Please try again with a different model."
                
                await asyncio.to_thread(user_memory.add_user_message, query)
                await asyncio.to_thread(user_memory.add_ai_message, response_content)
                return response_content

    async def _get_formatted_chat_history(self, session_id: str) -> List[Dict[str,str]]:
        user_memory = await self._get_user_memory(session_id)
        history_messages = []
        for msg in user_memory.messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            history_messages.append({"role": role, "content": msg.content})
        return history_messages

    async def query_stream(
        self, session_id: str, query: str, chat_history: Optional[List[Dict[str, str]]] = None,
        user_r2_document_keys: Optional[List[str]] = None,
        llm_model_name: Optional[str] = None, system_prompt_override: Optional[str] = None,
        enable_web_search: Optional[bool] = False
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream query response using hybrid search with LCEL pipeline."""
        try:
            logger.info(f"Starting streaming query for session: {session_id}")
            start_time = time.time()
            
            logger.info(f"🔄 Hybrid search: ALWAYS ACTIVE (BM25 Available: {HYBRID_SEARCH_AVAILABLE})")
            logger.info(f"🌐 Web search: {'ENABLED' if enable_web_search else 'DISABLED'}")
            logger.info(f"🧠 Using model: {llm_model_name or self.default_llm_model_name}")
            
            formatted_chat_history = await self._get_formatted_chat_history(session_id)
            retrieval_query = query
            
            logger.info(f"📝 Processing query: '{retrieval_query}'")
            
            # Create retrieval chain using LCEL
            # First gather all documents from different sources
            all_retrieved_docs = await self._gather_documents_from_all_sources(
                session_id, retrieval_query, user_r2_document_keys, enable_web_search
            )
            
            # LCEL implementation for response generation with streaming
            llm_stream_generator = await self._generate_response_with_lcel(
                session_id, query, all_retrieved_docs, formatted_chat_history,
                llm_model_name, system_prompt_override, stream=True
            )
            
            async for content_chunk in llm_stream_generator:
                yield {"type": "content", "data": content_chunk}
            
            total_time = int((time.time() - start_time) * 1000)
            logger.info(f"✅ Stream complete. Total processing time: {total_time}ms")
            yield {"type": "done", "data": {"total_time_ms": total_time}}
            
        except Exception as e:
            logger.error(f"Error in query_stream: {e}")
            yield {"type": "error", "data": f"Error processing query: {str(e)}"}

    async def query(
        self, session_id: str, query: str, chat_history: Optional[List[Dict[str, str]]] = None,
        user_r2_document_keys: Optional[List[str]] = None,
        llm_model_name: Optional[str] = None, system_prompt_override: Optional[str] = None,
        enable_web_search: Optional[bool] = False
    ) -> Dict[str, Any]:
        """Non-streaming query response using hybrid search with LCEL pipeline."""
        try:
            start_time = time.time()
            
            logger.info(f"🔄 Hybrid search: ALWAYS ACTIVE (BM25 Available: {HYBRID_SEARCH_AVAILABLE})")
            
            formatted_chat_history = await self._get_formatted_chat_history(session_id)
            
            # Gather all documents using LCEL approach
            all_retrieved_docs = await self._gather_documents_from_all_sources(
                session_id, query, user_r2_document_keys, enable_web_search
            )
            
            # Extract source names for response
            source_names_used = list(set([doc.metadata.get("source", "Unknown") 
                                          for doc in all_retrieved_docs if doc.metadata]))
            if not source_names_used and all_retrieved_docs: 
                source_names_used.append("Processed Documents")
            elif not all_retrieved_docs: 
                source_names_used.append("No Context Found")

            # Generate answer using LCEL pipeline
            answer_content = await self._generate_response_with_lcel(
                session_id, query, all_retrieved_docs, formatted_chat_history,
                llm_model_name, system_prompt_override, stream=False
            )
            
            return {
                "answer": answer_content, 
                "sources": source_names_used,
                "retrieval_details": {"documents_retrieved_count": len(all_retrieved_docs)},
                "total_time_ms": int((time.time() - start_time) * 1000)
            }
            
        except Exception as e:
            logger.error(f"Error in LCEL query: {e}")
            return {
                "answer": f"Error processing query: {str(e)}", 
                "sources": [],
                "retrieval_details": {"documents_retrieved_count": 0},
                "total_time_ms": 0
            }

    async def clear_user_session_context(self, session_id: str):
        """Clear user session context including chunks."""
        try:
            user_collection_name = self._get_user_qdrant_collection_name(session_id)
            
            # Delete Qdrant collection
            if self.qdrant_client:
                try:
                    await asyncio.to_thread(self.qdrant_client.delete_collection, collection_name=user_collection_name)
                    logger.info(f"Qdrant collection '{user_collection_name}' deleted.")
                except Exception as e:
                    if "not found" in str(e).lower():
                        logger.info(f"Qdrant collection '{user_collection_name}' not found during clear.")
                    else:
                        logger.error(f"Error deleting Qdrant collection '{user_collection_name}': {e}")
            
            # Clear retrievers, chunks, and memory
            if session_id in self.user_collection_retrievers: 
                del self.user_collection_retrievers[session_id]
            if session_id in self.user_chunks: 
                del self.user_chunks[session_id]
            if session_id in self.user_memories: 
                del self.user_memories[session_id]
                
            logger.info(f"✅ User session context cleared for session_id: {session_id}")
            
        except Exception as e:
            logger.error(f"Error clearing user session context for '{session_id}': {e}")

    async def clear_knowledge_base(self):
        """Clear knowledge base including chunks."""
        try:
            logger.info(f"Clearing KB for gpt_id '{self.gpt_id}'...")
            
            if self.qdrant_client:
                try:
                    await asyncio.to_thread(self.qdrant_client.delete_collection, collection_name=self.kb_collection_name)
                except Exception as e:
                    if "not found" in str(e).lower():
                        logger.info(f"KB Qdrant collection '{self.kb_collection_name}' not found.")
                    else: 
                        logger.error(f"Error deleting KB Qdrant collection: {e}")
            
            # Clear retriever and chunks
            self.kb_retriever = None
            self.kb_chunks = []
            
            # Ensure empty collection exists
            await asyncio.to_thread(self._ensure_qdrant_collection_exists_sync, self.kb_collection_name)
            logger.info(f"✅ Knowledge Base for gpt_id '{self.gpt_id}' cleared.")
            
        except Exception as e:
            logger.error(f"Error clearing knowledge base: {e}")

    async def clear_all_context(self):
        await self.clear_knowledge_base()
        active_session_ids = list(self.user_collection_retrievers.keys())
        for session_id in active_session_ids:
            await self.clear_user_session_context(session_id)
        self.user_collection_retrievers.clear(); self.user_memories.clear()
        if os.path.exists(self.temp_processing_path):
            try:
                await asyncio.to_thread(shutil.rmtree, self.temp_processing_path)
                os.makedirs(self.temp_processing_path, exist_ok=True)
            except Exception as e: print(f"Error clearing temp path '{self.temp_processing_path}': {e}")
        print(f"All context (KB, all user sessions, temp files) cleared for gpt_id '{self.gpt_id}'.")

    async def _gather_documents_from_all_sources(
        self, session_id: str, retrieval_query: str, 
        user_r2_document_keys: Optional[List[str]] = None,
        enable_web_search: bool = False,
        k_val: int = 5
    ) -> List[Document]:
        """Gather documents from all sources using LCEL approach."""
        try:
            all_retrieved_docs: List[Document] = []
            
            # Get user document context using hybrid search
            user_session_retriever = await self._get_user_retriever(session_id)
            user_session_docs = await self._get_retrieved_documents(
                user_session_retriever, 
                retrieval_query, 
                k_val=k_val
            )
            if user_session_docs: 
                logger.info(f"📄 Retrieved {len(user_session_docs)} user-specific documents via hybrid search")
                all_retrieved_docs.extend(user_session_docs)
            
            # Get knowledge base context using hybrid search
            kb_docs = await self._get_retrieved_documents(
                self.kb_retriever, 
                retrieval_query, 
                k_val=k_val
            )
            if kb_docs: 
                logger.info(f"📚 Retrieved {len(kb_docs)} knowledge base documents via hybrid search")
                all_retrieved_docs.extend(kb_docs)
            
            # Add web search results if enabled
            if enable_web_search and self.tavily_client:
                web_docs = await self._get_web_search_docs(retrieval_query, True, num_results=4)
                if web_docs:
                    logger.info(f"🌐 Retrieved {len(web_docs)} web search documents")
                    all_retrieved_docs.extend(web_docs)
            
            # Process any adhoc document keys
            if user_r2_document_keys:
                logger.info(f"📎 Processing {len(user_r2_document_keys)} ad-hoc document keys")
                adhoc_load_tasks = [self._download_and_split_one_doc(r2_key) for r2_key in user_r2_document_keys]
                results_list_of_splits = await asyncio.gather(*adhoc_load_tasks)
                adhoc_docs_count = 0
                for splits_from_one_doc in results_list_of_splits:
                    adhoc_docs_count += len(splits_from_one_doc)
                    all_retrieved_docs.extend(splits_from_one_doc) 
                logger.info(f"📎 Added {adhoc_docs_count} splits from ad-hoc documents")
            
            # Deduplicate documents
            unique_docs_content = set()
            deduplicated_docs = []
            for doc in all_retrieved_docs:
                if doc.page_content not in unique_docs_content:
                    deduplicated_docs.append(doc)
                    unique_docs_content.add(doc.page_content)
            
            logger.info(f"🔍 Retrieved {len(deduplicated_docs)} total unique documents via hybrid search")
            return deduplicated_docs
        except Exception as e:
            logger.error(f"Error gathering documents: {e}")
            return []

    @traceable
    async def _generate_response_with_lcel(
        self, session_id: str, query: str, all_context_docs: List[Document],
        chat_history_messages: List[Dict[str, str]], llm_model_name_override: Optional[str],
        system_prompt_override: Optional[str], stream: bool = False
    ) -> Union[AsyncGenerator[str, None], str]:
        """Generate response using LCEL pipeline."""
        try:
            current_llm_model = llm_model_name_override or self.default_llm_model_name
            current_system_prompt = system_prompt_override or self.default_system_prompt
            
            # Format context 
            context_str = self._format_docs_for_llm_context(all_context_docs, "Retrieved Context")
            if not context_str.strip():
                context_str = "No relevant context could be found from any available source for this query. Please ensure documents are uploaded and relevant to your question."
            
            # Create a context-aware RAG prompt template
            rag_prompt_template = f"""
            System: {current_system_prompt}
            
            You are answering a question based on the provided context.
            
            Context:
            {{context}}
            

            Chat History:
            {{chat_history}}
            
            User Question: {{question}}
            
            Please provide a comprehensive, detailed, and accurate answer based solely on the context provided.
            Use structured markdown formatting to improve readability.
            If the context is insufficient to answer the question, clearly state that.
            """
            
            # Create the prompt template
            prompt = ChatPromptTemplate.from_template(rag_prompt_template)
            
            # Format chat history
            chat_history_str = ""
            if chat_history_messages:
                for msg in chat_history_messages:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role and content:
                        chat_history_str += f"{role.capitalize()}: {content}\n"
            
            # In LCEL, we handle streaming differently depending on the model type
            if current_llm_model.startswith("gpt-"):
                # OpenAI models
                chat_model = ChatOpenAI(
                    model=current_llm_model,
                    temperature=self.default_temperature,
                    streaming=stream,
                    api_key=self.openai_api_key
                )
                
                # Build the LCEL RAG chain
                rag_chain = (
                    {"context": lambda _: context_str, 
                     "question": lambda _: query,
                     "chat_history": lambda _: chat_history_str}
                    | prompt 
                    | chat_model
                    | StrOutputParser()
                )
                
                # Handle streaming
                if stream:
                    async def stream_response():
                        response_stream = await rag_chain.ainvoke({})
                        full_response = ""
                        async for chunk in response_stream:
                            full_response += chunk
                            yield chunk
                        
                        # Update chat history
                        user_memory = await self._get_user_memory(session_id)
                        await asyncio.to_thread(user_memory.add_user_message, query)
                        await asyncio.to_thread(user_memory.add_ai_message, full_response)
                    
                    return stream_response()
                else:
                    response = await rag_chain.ainvoke({})
                    
                    # Update chat history
                    user_memory = await self._get_user_memory(session_id)
                    await asyncio.to_thread(user_memory.add_user_message, query)
                    await asyncio.to_thread(user_memory.add_ai_message, response)
                    
                    return response
            else:
                # For other models, fall back to our existing implementation
                # since they may require different implementations
                return await self._generate_llm_response(
                    session_id, query, all_context_docs, chat_history_messages,
                    llm_model_name_override, system_prompt_override, stream
                )
                
        except Exception as e:
            logger.error(f"Error in LCEL response generation: {e}")
            error_msg = f"Error generating response: {str(e)}"
            if stream:
                async def error_stream():
                    yield error_msg
                return error_stream()
            return error_msg

async def main_test_rag_qdrant():
    print("Ensure QDRANT_URL and OPENAI_API_KEY are set in .env for this test.")
    if not (os.getenv("OPENAI_API_KEY") and os.getenv("QDRANT_URL")):
        print("Skipping test: OPENAI_API_KEY or QDRANT_URL not set.")
        return

    class DummyR2Storage:
        async def download_file(self, key: str, local_path: str) -> bool:
            with open(local_path, "w") as f:
                f.write("This is a test document for RAG.")
            return True

        async def upload_file(self, file_data, filename: str, is_user_doc: bool = False):
            return True, f"test/{filename}"

        async def download_file_from_url(self, url: str):
            return True, f"test/doc_from_url_{url[-10:]}"

    rag = EnhancedRAG(
        gpt_id="test_gpt",
        r2_storage_client=DummyR2Storage(),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        qdrant_url=os.getenv("QDRANT_URL"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY")
    )

    await rag.update_knowledge_base_from_r2(["test/doc1.txt"])
    session_id = "test_session"
    await rag.update_user_documents_from_r2(session_id, ["test/doc2.txt"])

    async for chunk in rag.query_stream(session_id, "What is in the test document?", enable_web_search=False):
        print(chunk)

if __name__ == "__main__":
    print(f"rag.py loaded. Qdrant URL: {os.getenv('QDRANT_URL')}. Tavily available: {TAVILY_AVAILABLE}. BM25 available: {HYBRID_SEARCH_AVAILABLE}")