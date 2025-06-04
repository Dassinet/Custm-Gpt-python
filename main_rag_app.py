from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Body, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
import json
import os
import asyncio
from typing import List, Dict, Any, Optional, Union
import shutil
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import time
import logging
from io import BytesIO
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from contextlib import asynccontextmanager

from storage import CloudflareR2Storage
from rag import EnhancedRAG, CLAUDE_AVAILABLE, HYBRID_SEARCH_AVAILABLE, GEMINI_AVAILABLE, GROQ_AVAILABLE
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

load_dotenv()

# # --- Langsmith ---
# langsmith_api_key = os.getenv("LANGCHAIN_API_KEY")
# langsmith_project = os.getenv("LANGCHAIN_PROJECT")
# langsmith_tracing = os.getenv("LANGCHAIN_TRACING")
# langsmith_endpoint = os.getenv("LANGCHAIN_ENDPOINT")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the cleanup function first
async def cleanup_r2_expired_files():
    """Periodic task to clean up expired R2 files"""
    logger.info("Running scheduled cleanup of expired R2 files...")
    try:
        # Initialize r2_storage first to avoid reference before assignment
        r2_storage = CloudflareR2Storage()
        await asyncio.to_thread(r2_storage.cleanup_expired_files)
        logger.info("✅ Scheduled R2 cleanup completed successfully")
    except Exception as e:
        logger.error(f"❌ Error during scheduled R2 cleanup: {e}")

# Define the lifespan manager before app initialization
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    scheduler = AsyncIOScheduler()
    scheduler.add_job(cleanup_r2_expired_files, 'interval', hours=6)
    scheduler.start()
    logger.info("✅ Scheduler started: R2 cleanup will run every 6 hours")
    logger.info(f"🔄 Hybrid search: ALWAYS ACTIVE (BM25 Available: {HYBRID_SEARCH_AVAILABLE})")
    
    yield  # This is where the app runs
    
    # Shutdown code
    scheduler.shutdown()
    logger.info("Scheduler shut down")

# Now initialize the app after defining the lifespan function
app = FastAPI(
    title="Enhanced RAG API with Always-On Hybrid Search", 
    version="2.0.0",
    description="RAG API with always-active hybrid search using vector similarity + BM25 keyword matching",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://custom-gpt-frontend-nine.vercel.app", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
)

# Now initialize r2_storage after app is defined
r2_storage = CloudflareR2Storage()

active_rag_sessions: Dict[str, EnhancedRAG] = {}
sessions_lock = asyncio.Lock()

LOCAL_DATA_BASE_PATH = os.getenv("LOCAL_DATA_PATH", "local_rag_data")
LOCAL_KB_INDEX_PATH_TEMPLATE = os.path.join(LOCAL_DATA_BASE_PATH, "kb_indexes", "{gpt_id}")
LOCAL_USER_INDEX_BASE_PATH = os.path.join(LOCAL_DATA_BASE_PATH, "user_indexes")
TEMP_DOWNLOAD_PATH = os.path.join(LOCAL_DATA_BASE_PATH, "temp_downloads")

os.makedirs(os.path.join(LOCAL_DATA_BASE_PATH, "kb_indexes"), exist_ok=True)
os.makedirs(LOCAL_USER_INDEX_BASE_PATH, exist_ok=True)
os.makedirs(TEMP_DOWNLOAD_PATH, exist_ok=True)

# --- Pydantic Models ---
class BaseRAGRequest(BaseModel):
    user_email: str
    gpt_id: str
    gpt_name: Optional[str] = "default_gpt"

class ChatPayload(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = []
    user_document_keys: Optional[List[str]] = Field([], alias="user_documents")
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    web_search_enabled: Optional[bool] = False

class ChatStreamRequest(BaseRAGRequest, ChatPayload):
    memory: Optional[List[Dict[str, str]]] = []

class ChatRequest(BaseRAGRequest, ChatPayload):
    pass

class GptContextSetupRequest(BaseRAGRequest):
    kb_document_urls: Optional[List[str]] = []
    default_model: Optional[str] = None
    default_system_prompt: Optional[str] = None

class FileUploadInfoResponse(BaseModel):
    filename: str
    stored_url_or_key: str
    status: str
    error_message: Optional[str] = None

class GptOpenedRequest(BaseModel):
    user_email: str
    gpt_id: str
    gpt_name: str
    file_urls: List[str] = []
    config_schema: Optional[Dict[str, Any]] = Field(default=None, alias="schema")
    api_keys: Optional[Dict[str, str]] = Field(default_factory=dict)

# --- Helper Functions ---
def get_session_id(user_email: str, gpt_id: str) -> str:
    email_part = user_email.replace('@', '_').replace('.', '_')
    return f"user_{email_part}_gpt_{gpt_id}"

async def get_or_create_rag_instance(
    user_email: str,
    gpt_id: str,
    gpt_name: Optional[str] = "default_gpt",
    default_model: Optional[str] = None,
    default_system_prompt: Optional[str] = None,
    api_keys: Optional[Dict[str, str]] = None
) -> EnhancedRAG:
    """Create or retrieve RAG instance with always-on hybrid search."""
    async with sessions_lock:
        if gpt_id not in active_rag_sessions:
            logger.info(f"Creating new EnhancedRAG instance with hybrid search for gpt_id: {gpt_id}")
            
            # Use API keys from frontend if available, otherwise fallback to environment
            openai_api_key = api_keys.get('openai') if api_keys and 'openai' in api_keys else os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY not set in environment or not provided by frontend.")
                
            qdrant_url = os.getenv("QDRANT_URL")
            qdrant_api_key = os.getenv("QDRANT_API_KEY")
            
            if not qdrant_url:
                raise ValueError("QDRANT_URL not set in environment.")
                
            # Get optional API keys for other providers from frontend or environment
            tavily_api_key = api_keys.get('tavily') if api_keys and 'tavily' in api_keys else os.getenv("TAVILY_API_KEY")

            # Create RAG instance without default_use_hybrid_search parameter (always active now)
            active_rag_sessions[gpt_id] = EnhancedRAG(
                gpt_id=gpt_id,
                r2_storage_client=r2_storage,
                openai_api_key=openai_api_key,
                default_llm_model_name=default_model or os.getenv("DEFAULT_OPENAI_MODEL", "gpt-4o"),
                qdrant_url=qdrant_url,
                qdrant_api_key=qdrant_api_key,
                temp_processing_path=TEMP_DOWNLOAD_PATH,
                default_system_prompt=default_system_prompt,
                tavily_api_key=tavily_api_key
            )
            
            logger.info(f"✅ RAG instance created with always-on hybrid search (BM25: {HYBRID_SEARCH_AVAILABLE})")
            
            # Update API keys for other providers if available
            rag_instance = active_rag_sessions[gpt_id]
            
            # Update Claude API key
            claude_api_key = api_keys.get('claude') if api_keys and 'claude' in api_keys else os.getenv("ANTHROPIC_API_KEY")
            if claude_api_key and hasattr(rag_instance, "claude_api_key"):
                rag_instance.claude_api_key = claude_api_key
                # Reinitialize Anthropic client if possible
                if hasattr(rag_instance, "anthropic_client") and CLAUDE_AVAILABLE:
                    import anthropic
                    rag_instance.anthropic_client = anthropic.AsyncAnthropic(api_key=claude_api_key)
                    logger.info("✅ Claude client reinitialized with user-provided API key")
                
            # Update Gemini API key
            gemini_api_key = api_keys.get('gemini') if api_keys and 'gemini' in api_keys else os.getenv("GOOGLE_API_KEY")
            if gemini_api_key and hasattr(rag_instance, "gemini_api_key"):
                rag_instance.gemini_api_key = gemini_api_key
                # Reinitialize Gemini client if possible
                if hasattr(rag_instance, "gemini_client") and GEMINI_AVAILABLE:
                    import google.generativeai as genai
                    genai.configure(api_key=gemini_api_key)
                    rag_instance.gemini_client = genai
                    logger.info("✅ Gemini client reinitialized with user-provided API key")
            
            # Update Groq API key
            groq_api_key = api_keys.get('groq') if api_keys and 'groq' in api_keys else os.getenv("GROQ_API_KEY")
            if groq_api_key and hasattr(rag_instance, "groq_api_key"):
                rag_instance.groq_api_key = groq_api_key
                # Reinitialize Groq client if possible
                if hasattr(rag_instance, "groq_client") and GROQ_AVAILABLE:
                    from groq import AsyncGroq
                    rag_instance.groq_client = AsyncGroq(api_key=groq_api_key)
                    logger.info("✅ Groq client reinitialized with user-provided API key")

            # Update OpenRouter API key
            openrouter_api_key = api_keys.get('openrouter') if api_keys and 'openrouter' in api_keys else os.getenv("OPENROUTER_API_KEY")
            if openrouter_api_key and hasattr(rag_instance, "openrouter_api_key"):
                rag_instance.openrouter_api_key = openrouter_api_key
                # Reinitialize OpenRouter client if possible
                if hasattr(rag_instance, "openrouter_client"):
                    try:
                        import httpx
                        from openai import AsyncOpenAI
                        timeout_config = httpx.Timeout(connect=15.0, read=180.0, write=15.0, pool=15.0)
                        rag_instance.openrouter_client = AsyncOpenAI(
                            api_key=openrouter_api_key,
                            base_url="https://openrouter.ai/api/v1",
                            timeout=timeout_config,
                            max_retries=1
                        )
                        logger.info("✅ OpenRouter client reinitialized with user-provided API key")
                    except Exception as e:
                        logger.error(f"❌ Error reinitializing OpenRouter client: {e}")

        else:
            rag_instance = active_rag_sessions[gpt_id]
            if default_model:
                rag_instance.default_llm_model_name = default_model
            if default_system_prompt:
                rag_instance.default_system_prompt = default_system_prompt
                
            # Update API keys if a RAG instance already exists
            if api_keys:
                # Update OpenAI API key if provided
                if 'openai' in api_keys and api_keys['openai']:
                    old_key = rag_instance.openai_api_key
                    new_key = api_keys['openai']
                    if old_key != new_key:
                        rag_instance.openai_api_key = new_key
                        # Update OpenAI client with new key
                        if hasattr(rag_instance, "async_openai_client"):
                            try:
                                import httpx
                                from openai import AsyncOpenAI
                                timeout_config = httpx.Timeout(connect=15.0, read=180.0, write=15.0, pool=15.0)
                                rag_instance.async_openai_client = AsyncOpenAI(
                                    api_key=new_key,
                                    timeout=timeout_config,
                                    max_retries=1
                                )
                                logger.info("✅ OpenAI client reinitialized with user-provided API key")
                            except Exception as e:
                                logger.error(f"❌ Error reinitializing OpenAI client: {e}")
                
                # Update other API keys similarly
                for key_name in ['claude', 'gemini', 'groq', 'tavily', 'openrouter']:
                    if key_name in api_keys and api_keys[key_name] and hasattr(rag_instance, f"{key_name}_api_key"):
                        attr_name = f"{key_name}_api_key"
                        if getattr(rag_instance, attr_name) != api_keys[key_name]:
                            setattr(rag_instance, attr_name, api_keys[key_name])
                            logger.info(f"✅ Updated {key_name} API key for RAG instance {gpt_id}")
                
            logger.info(f"Reusing EnhancedRAG instance for gpt_id: {gpt_id}. Hybrid search always active.")

        return active_rag_sessions[gpt_id]

async def _process_uploaded_file_to_r2(
    file: UploadFile,
    is_user_doc: bool
) -> FileUploadInfoResponse:
    """Process uploaded file to R2 storage with enhanced error handling."""
    try:
        file_content = await file.read()
        file_bytes_io = BytesIO(file_content)
        
        # Log file type detection
        filename = file.filename or "unknown"
        _, ext = os.path.splitext(filename)
        ext = ext.lower()
        is_image = ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
        
        if is_image:
            logger.info(f"📸 Processing image file: {filename}")
        else:
            logger.info(f"📄 Processing document file: {filename}")
        
        success, r2_path_or_error = await asyncio.to_thread(
            r2_storage.upload_file,
            file_data=file_bytes_io,
            filename=filename,
            is_user_doc=is_user_doc
        )

        if success:
            logger.info(f"✅ File '{filename}' stored at: {r2_path_or_error}")
            return FileUploadInfoResponse(
                filename=filename,
                stored_url_or_key=r2_path_or_error,
                status="success"
            )
        else:
            logger.error(f"❌ Failed to store file '{filename}': {r2_path_or_error}")
            return FileUploadInfoResponse(
                filename=filename,
                stored_url_or_key="", status="failure", error_message=r2_path_or_error
            )
    except Exception as e:
        logger.error(f"❌ Exception processing file '{file.filename}': {e}")
        return FileUploadInfoResponse(
            filename=file.filename or "unknown",
            stored_url_or_key="", status="failure", error_message=str(e)
        )

# --- API Endpoints ---

@app.post("/setup-gpt-context", summary="Initialize/update a GPT's knowledge base from URLs with hybrid search")
async def setup_gpt_context_endpoint(request: GptContextSetupRequest, background_tasks: BackgroundTasks):
    """Setup GPT context with always-on hybrid search."""
    try:
        rag_instance = await get_or_create_rag_instance(
            user_email=request.user_email,
            gpt_id=request.gpt_id,
            gpt_name=request.gpt_name,
            default_model=request.default_model,
            default_system_prompt=request.default_system_prompt
        )

        if request.kb_document_urls:
            async def _process_kb_urls_task(urls: List[str], rag: EnhancedRAG):
                logger.info(f"📥 BG Task: Processing {len(urls)} KB URLs for gpt_id '{rag.gpt_id}' with hybrid search...")
                r2_kb_keys_or_urls_for_indexing = []
                for url in urls:
                    if not (url.startswith('http://') or url.startswith('https://')):
                        logger.warning(f"⚠️ Skipping invalid KB URL: {url}")
                        continue
                    success, r2_path = await asyncio.to_thread(
                        r2_storage.download_file_from_url, url=url
                    )
                    if success:
                        r2_kb_keys_or_urls_for_indexing.append(r2_path)
                        logger.info(f"✅ KB URL '{url}' processed to R2: {r2_path}")
                    else:
                        logger.error(f"❌ Failed to process KB URL '{url}': {r2_path}")
                
                if r2_kb_keys_or_urls_for_indexing:
                    try:
                        await rag.update_knowledge_base_from_r2(r2_kb_keys_or_urls_for_indexing)
                        logger.info(f"✅ KB hybrid search indexing completed for gpt_id '{rag.gpt_id}'")
                    except Exception as e:
                        logger.error(f"❌ Error indexing KB documents for gpt_id '{rag.gpt_id}': {e}")

            background_tasks.add_task(_process_kb_urls_task, request.kb_document_urls, rag_instance)
            return JSONResponse(status_code=202, content={
                "message": f"KB processing with hybrid search for gpt_id '{request.gpt_id}' initiated for {len(request.kb_document_urls)} URLs.",
                "gpt_id": request.gpt_id,
                "hybrid_search_active": True,
                "bm25_available": HYBRID_SEARCH_AVAILABLE
            })
        else:
            return JSONResponse(status_code=200, content={
                "message": f"No KB URLs provided. RAG instance with hybrid search for gpt_id '{request.gpt_id}' is ready.",
                "gpt_id": request.gpt_id,
                "hybrid_search_active": True,
                "bm25_available": HYBRID_SEARCH_AVAILABLE
            })
    except Exception as e:
        logger.error(f"❌ Error in setup-gpt-context: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/upload-documents", summary="Upload documents (KB or User-specific) with hybrid search indexing")
async def upload_documents_endpoint(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    user_email: str = Form(...),
    gpt_id: str = Form(...),
    is_user_document: str = Form("false"),
):
    """Upload and index documents with always-on hybrid search."""
    try:
        is_user_doc_bool = is_user_document.lower() == "true"
        processing_results: List[FileUploadInfoResponse] = []
        r2_keys_or_urls_for_indexing: List[str] = []

        logger.info(f"📤 Processing {len(files)} files for {'user' if is_user_doc_bool else 'KB'} documents with hybrid search")

        for file_upload in files:
            result = await _process_uploaded_file_to_r2(file_upload, is_user_doc_bool)
            processing_results.append(result)
            if result.status == "success" and result.stored_url_or_key:
                r2_keys_or_urls_for_indexing.append(result.stored_url_or_key)

        if not r2_keys_or_urls_for_indexing:
            return JSONResponse(status_code=400, content={
                "message": "No files were successfully uploaded to R2.",
                "upload_results": [r.model_dump() for r in processing_results],
                "hybrid_search_active": True
            })

        rag_instance = await get_or_create_rag_instance(user_email=user_email, gpt_id=gpt_id)
        
        async def _index_documents_task(rag: EnhancedRAG, keys_or_urls: List[str], is_user_specific: bool, u_email: str, g_id: str):
            doc_type = "user-specific" if is_user_specific else "knowledge base"
            s_id = get_session_id(u_email, g_id)
            logger.info(f"📚 BG Task: Hybrid search indexing {len(keys_or_urls)} {doc_type} documents for gpt_id '{rag.gpt_id}' (session '{s_id}')...")
            try:
                if is_user_specific:
                    await rag.update_user_documents_from_r2(session_id=s_id, r2_keys_or_urls=keys_or_urls)
                else:
                    await rag.update_knowledge_base_from_r2(keys_or_urls)
                logger.info(f"✅ BG Task: Hybrid search indexing complete for {doc_type} documents.")
            except Exception as e:
                logger.error(f"❌ BG Task: Error indexing {doc_type} documents for gpt_id '{rag.gpt_id}': {e}")

        background_tasks.add_task(_index_documents_task, rag_instance, r2_keys_or_urls_for_indexing, is_user_doc_bool, user_email, gpt_id)

        return JSONResponse(status_code=202, content={
            "message": f"{len(r2_keys_or_urls_for_indexing)} files accepted for {'user-specific' if is_user_doc_bool else 'knowledge base'} hybrid search indexing. Processing in background.",
            "upload_results": [r.model_dump() for r in processing_results],
            "hybrid_search_active": True,
            "bm25_available": HYBRID_SEARCH_AVAILABLE
        })
    except Exception as e:
        logger.error(f"❌ Error in upload-documents: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/chat-stream", summary="Stream chat responses using hybrid search")
async def chat_stream(request: ChatStreamRequest):
    """Stream chat responses with always-on hybrid search."""
    try:
        # Initialize rag_instance
        rag_instance = await get_or_create_rag_instance(
            user_email=request.user_email,
            gpt_id=request.gpt_id,
            gpt_name=request.gpt_name,
            default_model=request.model,
            default_system_prompt=request.system_prompt
        )
        
        session_id = get_session_id(request.user_email, request.gpt_id)

        logger.info(f"\n{'='*50}")
        logger.info(f"📝 New streaming chat request from user: {request.user_email}")
        logger.info(f"🔍 GPT ID: {request.gpt_id}")
        logger.info(f"💬 Query: '{request.message}'")
        logger.info(f"🔄 Hybrid search: ALWAYS ACTIVE (BM25: {HYBRID_SEARCH_AVAILABLE})")
        logger.info(f"🌐 Web search: {'ENABLED' if request.web_search_enabled else 'DISABLED'}")
        logger.info(f"🧠 Model: {request.model or rag_instance.default_llm_model_name}")
        logger.info(f"{'='*50}")

        # Setup SSE headers
        headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }

        # Create streaming response generator
        async def generate():
            try:
                async for chunk in rag_instance.query_stream(
                    session_id=session_id,
                    query=request.message,
                    chat_history=request.history,
                    user_r2_document_keys=request.user_document_keys,
                    llm_model_name=request.model,
                    system_prompt_override=request.system_prompt,
                    enable_web_search=request.web_search_enabled
                ):
                    yield f"data: {json.dumps(chunk)}\n\n"
            except Exception as e:
                logger.error(f"❌ Error during streaming in /chat-stream: {e}")
                error_chunk = {
                    "type": "error",
                    "data": {"error": str(e)}
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"

        return StreamingResponse(generate(), headers=headers)
    
    except Exception as e:
        logger.error(f"❌ Error in /chat-stream endpoint: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/chat", summary="Handle non-streaming chat requests with hybrid search")
async def chat_endpoint(request: ChatRequest):
    """Handle non-streaming chat with always-on hybrid search."""
    try:
        rag_instance = await get_or_create_rag_instance(
            user_email=request.user_email, 
            gpt_id=request.gpt_id, 
            gpt_name=request.gpt_name,
            default_model=request.model,
            default_system_prompt=request.system_prompt
        )
        session_id = get_session_id(request.user_email, request.gpt_id)
        
        logger.info(f"💬 Chat request: '{request.message}' with hybrid search for session {session_id}")
        
        response_data = await rag_instance.query(
            session_id=session_id,
            query=request.message,
            chat_history=request.history,
            user_r2_document_keys=request.user_document_keys,
            llm_model_name=request.model,
            system_prompt_override=request.system_prompt,
            enable_web_search=request.web_search_enabled
        )
        
        # Add hybrid search info to response
        response_data["hybrid_search_active"] = True
        response_data["bm25_available"] = HYBRID_SEARCH_AVAILABLE
        
        return JSONResponse(content={"success": True, "data": response_data})
    except Exception as e:
        logger.error(f"❌ Error in /chat endpoint: {e}")
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

@app.post("/gpt-opened", summary="Notify backend when a GPT is opened with hybrid search")
async def gpt_opened_endpoint(request: GptOpenedRequest, background_tasks: BackgroundTasks):
    """Handle GPT opened event with always-on hybrid search setup."""
    try:
        logger.info(f"🚀 GPT opened: {request.gpt_name} (ID: {request.gpt_id}) for user: {request.user_email}")
        
        rag_instance = await get_or_create_rag_instance(
            user_email=request.user_email,
            gpt_id=request.gpt_id,
            gpt_name=request.gpt_name,
            default_model=request.config_schema.get("model") if request.config_schema else None,
            default_system_prompt=request.config_schema.get("instructions") if request.config_schema else None,
            api_keys=request.api_keys if hasattr(request, "api_keys") else None
        )
        
        sanitized_email = request.user_email.replace('@', '_').replace('.', '_')
        sanitized_gpt_name = (request.gpt_name or 'gpt').replace(' ', '_').replace('-', '_')
        collection_name = f"kb_{sanitized_email}_{sanitized_gpt_name}_{request.gpt_id}"
        
        if request.file_urls:
            async def _process_kb_urls_task(urls: List[str], rag: EnhancedRAG):
                logger.info(f"📥 Processing {len(urls)} KB URLs with hybrid search for GPT: {request.gpt_name}")
                r2_kb_keys_or_urls_for_indexing = []
                for url in urls:
                    if url.startswith('http://') or url.startswith('https://'):
                        success, r2_path = await asyncio.to_thread(
                            r2_storage.download_file_from_url, url=url
                        )
                        if success:
                            r2_kb_keys_or_urls_for_indexing.append(r2_path)
                            logger.info(f"✅ KB URL processed: {url}")
                        else:
                            logger.error(f"❌ Failed to process KB URL: {url}")
                
                if r2_kb_keys_or_urls_for_indexing:
                    try:
                        await rag.update_knowledge_base_from_r2(r2_kb_keys_or_urls_for_indexing)
                        logger.info(f"✅ Hybrid search KB indexing completed for {request.gpt_name}")
                    except Exception as e:
                        logger.error(f"❌ Error indexing KB documents for gpt_id '{rag.gpt_id}': {e}")
            
            background_tasks.add_task(_process_kb_urls_task, request.file_urls, rag_instance)
        
        return {
            "success": True, 
            "collection_name": collection_name,
            "hybrid_search_active": True,  # Always true now
            "bm25_available": HYBRID_SEARCH_AVAILABLE,
            "message": f"GPT '{request.gpt_name}' initialized with always-on hybrid search"
        }
    except Exception as e:
        logger.error(f"❌ Error in gpt-opened endpoint: {e}")
        return {"success": False, "error": str(e)}

@app.post("/upload-chat-files", summary="Upload files for chat with hybrid search indexing")
async def upload_chat_files_endpoint(
    files: List[UploadFile] = File(...),
    user_email: str = Form(...),
    gpt_id: str = Form(...),
    gpt_name: str = Form(...),
    collection_name: str = Form(...),
    is_user_document: str = Form("true"),
    optimize_pdfs: str = Form("false"),
):
    """Upload chat files with always-on hybrid search indexing."""
    try:
        is_user_doc_bool = is_user_document.lower() == "true"
        optimize_pdfs_bool = optimize_pdfs.lower() == "true"
        
        logger.info(f"📤 Uploading {len(files)} chat files with hybrid search for session: {get_session_id(user_email, gpt_id)}")
        
        processing_results = []
        file_urls = []

        for file_upload in files:
            result = await _process_uploaded_file_to_r2(file_upload, is_user_doc_bool)
            if result.status == "success" and result.stored_url_or_key:
                file_urls.append(result.stored_url_or_key)
            processing_results.append(result)

        rag_instance = await get_or_create_rag_instance(
            user_email=user_email, 
            gpt_id=gpt_id,
            gpt_name=gpt_name
        )
        
        if file_urls:
            session_id = get_session_id(user_email, gpt_id)
            
            try:
                if is_user_doc_bool:
                    await rag_instance.update_user_documents_from_r2(session_id=session_id, r2_keys_or_urls=file_urls)
                else:
                    await rag_instance.update_knowledge_base_from_r2(file_urls)
                logger.info(f"✅ Hybrid search indexing complete for {len(file_urls)} {'user-specific' if is_user_doc_bool else 'knowledge base'} documents for session '{session_id}'.")
            except Exception as e:
                logger.error(f"❌ Error indexing chat files for session '{session_id}': {e}")
                return {
                    "success": False,
                    "message": f"Failed to index {len(file_urls)} files with hybrid search: {str(e)}",
                    "file_urls": file_urls,
                    "processing": False,
                    "hybrid_search_active": True
                }
        
        return {
            "success": True,
            "message": f"Processed and indexed {len(file_urls)} files with hybrid search (BM25: {HYBRID_SEARCH_AVAILABLE})",
            "file_urls": file_urls,
            "processing": len(file_urls) > 0,
            "hybrid_search_active": True,
            "bm25_available": HYBRID_SEARCH_AVAILABLE
        }
    except Exception as e:
        logger.error(f"❌ Error in upload-chat-files: {e}")
        return {"success": False, "error": str(e)}

@app.get("/gpt-collection-info/{param1}/{param2}", summary="Get information about a GPT collection with hybrid search")
async def gpt_collection_info(param1: str, param2: str):
    """Get collection info with hybrid search status."""
    return {
        "status": "available",
        "timestamp": time.time(),
        "hybrid_search_active": True,
        "bm25_available": HYBRID_SEARCH_AVAILABLE,
        "search_type": "vector + bm25 ensemble"
    }

@app.get("/", include_in_schema=False)
async def root_redirect():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")

@app.get("/health", summary="Health check endpoint with hybrid search status", tags=["Monitoring"])
async def health_check():
    """Health check with comprehensive component status."""
    return {
        "status": "healthy", 
        "timestamp": time.time(),
        "hybrid_search_active": True,
        "hybrid_search_available": HYBRID_SEARCH_AVAILABLE,
        "search_capabilities": {
            "vector_search": True,
            "bm25_keyword_search": HYBRID_SEARCH_AVAILABLE,
            "ensemble_retriever": True
        },
        "components": {
            "bm25": HYBRID_SEARCH_AVAILABLE,
            "claude": CLAUDE_AVAILABLE,
            "gemini": GEMINI_AVAILABLE,
            "groq": GROQ_AVAILABLE
        }
    }

@app.post("/dev/reset-gpt-context", summary="DEVELOPMENT ONLY: Clear RAG context for a GPT", tags=["Development"])
async def dev_reset_gpt_context_endpoint(gpt_id: str = Form(...)):
    """Reset GPT context including hybrid search indexes."""
    if os.getenv("ENVIRONMENT_TYPE", "production").lower() != "development":
        return JSONResponse(status_code=403, content={"error": "Endpoint only available in development."})

    async with sessions_lock:
        if gpt_id in active_rag_sessions:
            try:
                rag_instance_to_reset = active_rag_sessions.pop(gpt_id)
                await rag_instance_to_reset.clear_all_context()
                
                kb_index_path_to_delete = LOCAL_KB_INDEX_PATH_TEMPLATE.format(gpt_id=gpt_id)
                if os.path.exists(kb_index_path_to_delete):
                    shutil.rmtree(kb_index_path_to_delete)
                
                logger.info(f"🧹 DEV: Cleared hybrid search context and local KB index for gpt_id '{gpt_id}'. R2 files not deleted.")
                return {
                    "status": "success", 
                    "message": f"RAG context with hybrid search for gpt_id '{gpt_id}' cleared from memory and local disk.",
                    "hybrid_search_active": True
                }
            except Exception as e:
                logger.error(f"❌ DEV: Error clearing context for gpt_id '{gpt_id}': {e}")
                return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
        else:
            return JSONResponse(status_code=404, content={
                "status": "not_found", 
                "message": f"No active RAG context for gpt_id '{gpt_id}'."
            })

@app.post("/maintenance/cleanup-r2", summary="Manually trigger cleanup of expired R2 files", tags=["Maintenance"])
async def manual_cleanup_r2():
    """Manually trigger R2 cleanup."""
    try:
        await asyncio.to_thread(r2_storage.cleanup_expired_files)
        logger.info("✅ Manual R2 cleanup completed")
        return {"status": "success", "message": "R2 cleanup completed"}
    except Exception as e:
        logger.error(f"❌ Error during manual R2 cleanup: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Error during R2 cleanup: {str(e)}"}
        )

if __name__ == "__main__":
    print(f"🚀 Starting Enhanced RAG API with Always-On Hybrid Search")
    print(f"🔄 Hybrid Search: ALWAYS ACTIVE")
    print(f"📊 BM25 Available: {HYBRID_SEARCH_AVAILABLE}")
    print(f"🔗 Vector + Keyword Ensemble Retrieval Enabled")
    uvicorn.run(app, host="0.0.0.0", port=8000)