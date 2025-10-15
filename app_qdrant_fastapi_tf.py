"""
RAG System with Qdrant Vector Database - FastAPI Version
Install: pip install fastapi uvicorn python-multipart anthropic transformers torch qdrant-client pypdf
Run: uvicorn app_qdrant_fastapi:app --reload
Then open http://localhost:8000 in your browser
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic
from transformers import AutoTokenizer, AutoModel
import torch
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Tuple, Optional
import os
import re
from pypdf import PdfReader
from io import BytesIO
import uuid
import numpy as np
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    Code before yield runs on startup, code after yield runs on shutdown.
    """
    # STARTUP
    if rag is None:
        print("\n⚠️  WARNING: Server started without valid API key!")
        print("⚠️  All API endpoints will return errors until configured.")
        print("⚠️  Visit http://localhost:8000/health to check status\n")
    else:
        example_docs = [
            """
            Artificial Intelligence (AI) is transforming healthcare in numerous ways. 
            Machine learning algorithms can now detect diseases from medical images with 
            accuracy rivaling human experts. AI-powered diagnostic tools analyze X-rays, 
            MRIs, and CT scans to identify conditions like cancer, pneumonia, and fractures.
            
            Natural language processing helps extract insights from medical records and 
            research papers. Predictive models forecast patient outcomes and identify 
            high-risk individuals who may benefit from early intervention.
            """,
            """
            Climate change is causing significant impacts on global ecosystems. Rising 
            temperatures are leading to more frequent and severe weather events, including 
            hurricanes, droughts, and floods. The Arctic ice is melting at an alarming rate, 
            contributing to sea level rise that threatens coastal communities.
            
            Carbon emissions from fossil fuels are the primary driver of climate change. 
            Renewable energy sources like solar and wind power offer sustainable alternatives 
            that can help reduce greenhouse gas emissions and mitigate climate impacts.
            """,
            """
            Quantum computing represents a paradigm shift in computational power. Unlike 
            classical computers that use bits (0 or 1), quantum computers use qubits that 
            can exist in multiple states simultaneously through superposition.
            
            This enables quantum computers to solve certain problems exponentially faster 
            than classical computers. Applications include cryptography, drug discovery, 
            optimization problems, and simulating quantum systems.
            """
        ]
        
        print("Initializing RAG system with example documents...")
        rag.add_documents(example_docs)
        print("RAG system ready!")
    
    yield  # Server is running
    
    # SHUTDOWN
    try:
        # Clean up embedding model
        if rag and hasattr(rag, 'embedding_model'):
            del rag.embedding_model
        
        # Close Qdrant client connection
        if rag and hasattr(rag, 'qdrant_client'):
            rag.qdrant_client.close()
        
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print("Resources cleaned up successfully")
    except Exception as e:
        print(f"Error during cleanup: {e}")


# Create FastAPI app with lifespan
app = FastAPI(
    title="RAG System with Qdrant",
    lifespan=lifespan  # Pass the lifespan context manager here
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom exception for configuration errors
class ConfigurationError(Exception):
    """Raised when there's a configuration issue."""
    pass

# Pydantic models for request/response
class DocumentsInput(BaseModel):
    documents: List[str]

class QuestionInput(BaseModel):
    question: str

class Source(BaseModel):
    text: str
    distance: float

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    cache_stats: dict

class StatusResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    configured: bool
    message: Optional[str] = None
    documents_count: Optional[int] = None
    instructions: Optional[str] = None

class EmbeddingModel:
    """Clean embedding model using transformers directly."""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        print(f"Loading embedding model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # Set to evaluation mode
        
        # Move to GPU if available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        print(f"Model loaded on device: {self.device}")
    
    def mean_pooling(self, model_output, attention_mask):
        """Perform mean pooling on token embeddings."""
        token_embeddings = model_output[0]  # First element contains token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def encode(self, texts: List[str], batch_size: int = 32, 
               convert_to_numpy: bool = True) -> np.ndarray:
        """Encode texts into embeddings."""
        if isinstance(texts, str):
            texts = [texts]
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            # Move to device
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
            
            # Generate embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            
            # Perform pooling
            embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
            
            # Normalize embeddings
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings.cpu())
        
        # Concatenate all batches
        all_embeddings = torch.cat(all_embeddings, dim=0)
        
        if convert_to_numpy:
            return all_embeddings.numpy()
        return all_embeddings
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class RAGSystem:
    def __init__(self, anthropic_api_key: str, embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.claude_client = anthropic.Anthropic(api_key=anthropic_api_key)
        self.embedding_model = EmbeddingModel(embedding_model)
        
        # Initialize Qdrant client (local mode)
        self.qdrant_client = QdrantClient(path="./qdrant_db")
        # in-memory for development/testing
        self.qdrant_client = QdrantClient(":memory:")
        
        self.collection_name = "documents"
        self.vector_size = 384  # all-MiniLM-L6-v2 dimension
        
        # Create collection if it doesn't exist
        try:
            self.qdrant_client.get_collection(self.collection_name)
            print(f"Collection '{self.collection_name}' already exists")
        except:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
            print(f"Created collection '{self.collection_name}'")
        
    def semantic_chunk_text(self, text: str, max_chunk_size: int = 1000) -> List[str]:
        """Split text on semantic boundaries (paragraphs, sentences)."""
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if len(current_chunk) + len(para) + 2 <= max_chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                if len(para) > max_chunk_size:
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    current_chunk = ""
                    
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
                            if current_chunk:
                                current_chunk += " " + sentence
                            else:
                                current_chunk = sentence
                        else:
                            if current_chunk:
                                chunks.append(current_chunk)
                            current_chunk = sentence
                else:
                    current_chunk = para
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def add_documents(self, documents: List[str], max_chunk_size: int = 1000):
        """Add documents to the RAG system using Qdrant."""
        print("Chunking documents semantically...")
        all_chunks = []
        for doc in documents:
            chunks = self.semantic_chunk_text(doc, max_chunk_size)
            all_chunks.extend(chunks)
        
        print(f"Created {len(all_chunks)} semantic chunks")
        
        print("Generating embeddings...")
        embeddings = self.embedding_model.encode(all_chunks, batch_size=32)
        
        print("Adding to Qdrant...")
        # Create points for Qdrant
        points = []
        for i, (chunk, embedding) in enumerate(zip(all_chunks, embeddings)):
            point_id = str(uuid.uuid4())
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload={"text": chunk}
                )
            )
        
        # Upload to Qdrant in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
        
        collection_info = self.qdrant_client.get_collection(self.collection_name)
        print(f"Added {len(all_chunks)} chunks to Qdrant")
        print(f"Total points in collection: {collection_info.points_count}")
        
    def retrieve(self, query: str, top_k: int = 5, similarity_threshold: float = None) -> List[Tuple[str, float]]:
        """Retrieve relevant chunks using Qdrant."""
        collection_info = self.qdrant_client.get_collection(self.collection_name)
        if collection_info.points_count == 0:
            raise ValueError("No documents added.")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Search in Qdrant (retrieve more for filtering)
        search_k = min(top_k * 3, collection_info.points_count)
        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=search_k
        )
        
        # Extract results (Qdrant returns similarity scores, not distances)
        documents = [hit.payload["text"] for hit in search_results]
        distances = [1 - hit.score for hit in search_results]
        
        # Auto-set threshold based on gap in distances if not provided
        if similarity_threshold is None and len(distances) > 1:
            gaps = [distances[i+1] - distances[i] for i in range(len(distances)-1)]
            if gaps:
                max_gap_idx = gaps.index(max(gaps))
                if gaps[max_gap_idx] > distances[max_gap_idx] * 0.3:
                    similarity_threshold = distances[max_gap_idx] + gaps[max_gap_idx] * 0.5
        
        # Filter and deduplicate
        filtered_results = []
        seen_starts = set()
        
        for doc, distance in zip(documents, distances):
            if similarity_threshold is not None and distance > similarity_threshold:
                continue
            
            chunk_start = doc[:100].strip()
            
            if chunk_start not in seen_starts:
                filtered_results.append((doc, float(distance)))
                seen_starts.add(chunk_start)
                
            if len(filtered_results) >= top_k:
                break
        
        return filtered_results
    
    def get_all_chunks(self) -> List[str]:
        """Get all chunks from Qdrant for caching."""
        collection_info = self.qdrant_client.get_collection(self.collection_name)
        if collection_info.points_count == 0:
            return []
        
        # Scroll through all points
        all_chunks = []
        offset = None
        limit = 100
        
        while True:
            records, next_offset = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            all_chunks.extend([record.payload["text"] for record in records])
            
            if next_offset is None:
                break
            offset = next_offset
        
        return all_chunks
    
    def query(self, question: str, top_k: int = 5) -> dict:
        print(f"Query: {question}")
        results = self.retrieve(question, top_k)
        
        # Get all chunks for caching
        all_chunks = self.get_all_chunks()
        
        # Get indices of retrieved chunks for highlighting
        retrieved_indices = []
        for chunk, _ in results:
            try:
                retrieved_indices.append(all_chunks.index(chunk))
            except ValueError:
                pass
        
        # Cache ALL chunks in system prompt
        all_chunks_text = "\n\n---\n\n".join([
            f"[Chunk {i}]\n{chunk}" 
            for i, chunk in enumerate(all_chunks)
        ])
        
        relevant_chunks_str = ", ".join([str(i) for i in retrieved_indices[:top_k]])
        
        response = self.claude_client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=2048,
            system=[
                {
                    "type": "text",
                    "text": "You are a helpful assistant that answers questions based on provided context. If the answer cannot be found in the context, say so clearly.",
                },
                {
                    "type": "text", 
                    "text": f"Here is the complete knowledge base:\n\n{all_chunks_text}",
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            messages=[{
                "role": "user",
                "content": f"Question: {question}\n\nMost relevant chunks: {relevant_chunks_str}\n\nFocus primarily on the chunks listed above, but you may reference other chunks if needed. Answer:"
            }]
        )
        
        usage = response.usage
        cache_stats = {
            "input_tokens": usage.input_tokens,
            "cache_creation_input_tokens": getattr(usage, 'cache_creation_input_tokens', 0),
            "cache_read_input_tokens": getattr(usage, 'cache_read_input_tokens', 0),
            "output_tokens": usage.output_tokens
        }
        print(f"Cache stats: {cache_stats}")
        
        return {
            "answer": response.content[0].text,
            "sources": [{"text": chunk, "distance": dist} for chunk, dist in results],
            "cache_stats": cache_stats
        }
    
    def clear_database(self):
        """Clear all documents from Qdrant."""
        try:
            self.qdrant_client.delete_collection(self.collection_name)
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
            print("Database cleared")
        except Exception as e:
            print(f"Error clearing database: {e}")

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF bytes."""
    pdf_file = BytesIO(file_bytes)
    reader = PdfReader(pdf_file)
    
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n\n"
    
    return text

def get_api_key() -> str:
    """Get API key with proper validation."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    
    if not api_key or api_key == "your-api-key-here":
        raise ConfigurationError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Please set it with: export ANTHROPIC_API_KEY='your-key-here'"
        )
    
    return api_key

# Initialize RAG system with proper error handling
rag = None
try:
    API_KEY = get_api_key()
    rag = RAGSystem(anthropic_api_key=API_KEY)
except ConfigurationError as e:
    print(f"\n{'='*60}")
    print("CONFIGURATION ERROR")
    print(f"{'='*60}")
    print(f"\n{str(e)}\n")
    print("To fix this:")
    print("1. Get your API key from: https://console.anthropic.com/")
    print("2. Set it as an environment variable:")
    print("   export ANTHROPIC_API_KEY='your-api-key-here'")
    print("3. Restart the server\n")
    print(f"{'='*60}\n")

# Dependency to check configuration
def require_configured_rag() -> RAGSystem:
    """Dependency that ensures RAG system is configured."""
    if rag is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Service not configured",
                "message": "ANTHROPIC_API_KEY environment variable is not set",
                "instructions": "Please configure the API key and restart the server"
            }
        )
    return rag

# HTML template with health status
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG System with Qdrant</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .badge {
            display: inline-block;
            background: rgba(255,255,255,0.2);
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.85em;
            margin-top: 10px;
        }
        
        .health-status {
            background: white;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        .health-status.error {
            border-left: 4px solid #dc2626;
        }
        
        .health-status.healthy {
            border-left: 4px solid #16a34a;
        }
        
        .health-status h3 {
            margin-bottom: 10px;
        }
        
        .health-status.error h3 {
            color: #dc2626;
        }
        
        .health-status.healthy h3 {
            color: #16a34a;
        }
        
        .card {
            background: white;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 20px;
        }
        
        .input-section {
            margin-bottom: 20px;
        }
        
        .input-section label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
        }
        
        .input-section textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 1em;
            resize: vertical;
            transition: border-color 0.3s;
        }
        
        .input-section textarea:focus {
            outline: none;
            border-color: #dc2626;
        }
        
        .file-upload {
            border: 2px dashed #dc2626;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            margin-bottom: 15px;
        }
        
        .file-upload:hover {
            background: #fef2f2;
            border-color: #991b1b;
        }
        
        .file-upload input[type="file"] {
            display: none;
        }
        
        .file-upload-label {
            display: block;
            color: #dc2626;
            font-weight: 600;
            cursor: pointer;
        }
        
        .file-name {
            margin-top: 10px;
            color: #666;
            font-size: 0.9em;
        }
        
        .button-group {
            display: flex;
            gap: 10px;
            margin-top: 15px;
        }
        
        button {
            flex: 1;
            padding: 12px 24px;
            font-size: 1em;
            font-weight: 600;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
            color: white;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(220, 38, 38, 0.4);
        }
        
        .btn-secondary {
            background: #f0f0f0;
            color: #333;
        }
        
        .btn-secondary:hover {
            background: #e0e0e0;
        }
        
        .btn-danger {
            background: #7f1d1d;
            color: white;
        }
        
        .btn-danger:hover {
            background: #991b1b;
            transform: translateY(-2px);
        }
        
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none !important;
        }
        
        .loading {
            text-align: center;
            padding: 20px;
            color: #dc2626;
            font-weight: 600;
        }
        
        .answer-section {
            margin-top: 20px;
            padding: 20px;
            background: #fef2f2;
            border-radius: 8px;
            border-left: 4px solid #dc2626;
        }
        
        .answer-section h3 {
            color: #dc2626;
            margin-bottom: 10px;
        }
        
        .answer-text {
            line-height: 1.6;
            color: #333;
        }
        
        .answer-text h1, .answer-text h2, .answer-text h3 {
            margin-top: 1em;
            margin-bottom: 0.5em;
            color: #dc2626;
        }
        
        .answer-text h1 { font-size: 1.5em; }
        .answer-text h2 { font-size: 1.3em; }
        .answer-text h3 { font-size: 1.1em; }
        
        .answer-text p {
            margin-bottom: 1em;
        }
        
        .answer-text ul, .answer-text ol {
            margin-left: 2em;
            margin-bottom: 1em;
            padding-left: 0.5em;
        }
        
        .answer-text li {
            margin-bottom: 0.5em;
            margin-left: 0;
        }
        
        .answer-text strong {
            color: #dc2626;
            font-weight: 600;
        }
        
        .answer-text code {
            background: #f0f0f0;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }
        
        .answer-text pre {
            background: #f0f0f0;
            padding: 12px;
            border-radius: 6px;
            overflow-x: auto;
            margin-bottom: 1em;
        }
        
        .answer-text pre code {
            background: none;
            padding: 0;
        }
        
        .sources {
            margin-top: 20px;
        }
        
        .sources h4 {
            color: #dc2626;
            margin-bottom: 10px;
        }
        
        .source-item {
            background: white;
            padding: 12px;
            border-radius: 6px;
            margin-bottom: 8px;
            border: 1px solid #e0e0e0;
            font-size: 0.9em;
            color: #666;
        }
        
        .status {
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-weight: 500;
        }
        
        .status.success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        
        .status.error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            border-bottom: 2px solid #e0e0e0;
        }
        
        .tab {
            padding: 10px 20px;
            background: none;
            border: none;
            border-bottom: 3px solid transparent;
            cursor: pointer;
            font-weight: 600;
            color: #666;
            transition: all 0.3s;
        }
        
        .tab.active {
            color: #dc2626;
            border-bottom-color: #dc2626;
        }
        
        .tab:hover {
            color: #dc2626;
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 RAG System</h1>
            <p>Powered by Claude & Qdrant Vector DB (FastAPI)</p>
            <span class="badge">🚀 Production Ready</span>
        </div>
        
        <div id="healthStatus"></div>
        
        <div class="card">
            <div class="tabs">
                <button class="tab active" onclick="switchTab('text')">Text Input</button>
                <button class="tab" onclick="switchTab('pdf')">PDF Upload</button>
            </div>
            
            <div id="text-tab" class="tab-content active">
                <div class="input-section">
                    <label>Add Documents (one per line or separated by blank lines):</label>
                    <textarea id="documents" rows="6" placeholder="Paste your documents here..."></textarea>
                    <div class="button-group">
                        <button class="btn-secondary" onclick="addDocuments()">Add Documents</button>
                        <button class="btn-danger" onclick="clearDatabase()">Clear Database</button>
                    </div>
                </div>
            </div>
            
            <div id="pdf-tab" class="tab-content">
                <div class="input-section">
                    <label>Upload PDF Documents:</label>
                    <div class="file-upload" id="fileUploadArea">
                        <label class="file-upload-label" for="pdfInput">
                            📄 Click to select PDF files
                        </label>
                        <input type="file" id="pdfInput" accept=".pdf" multiple onchange="handleFileSelect(event)">
                        <div id="fileName" class="file-name"></div>
                    </div>
                    <div class="button-group">
                        <button class="btn-secondary" onclick="uploadPDFs()">Upload & Process PDFs</button>
                    </div>
                </div>
            </div>
            
            <div id="status"></div>
            
            <div class="input-section">
                <label>Ask a Question:</label>
                <textarea id="question" rows="3" placeholder="What would you like to know?"></textarea>
                <div class="button-group">
                    <button class="btn-primary" onclick="askQuestion()">Ask Question</button>
                </div>
            </div>
            
            <div id="loading" style="display: none;" class="loading">
                Processing your query...
            </div>
            
            <div id="response"></div>
        </div>
    </div>
    
    <script>
        let selectedFiles = [];
        
        // Check health on page load
        window.addEventListener('load', checkHealth);
        
        async function checkHealth() {
            try {
                const response = await fetch('/health');
                const data = await response.json();
                
                const healthDiv = document.getElementById('healthStatus');
                
                if (data.configured) {
                    healthDiv.innerHTML = `
                        <div class="health-status healthy">
                            <h3>✅ System Ready</h3>
                            <p>Documents in database: ${data.documents_count || 0}</p>
                        </div>
                    `;
                } else {
                    healthDiv.innerHTML = `
                        <div class="health-status error">
                            <h3>⚠️ Configuration Required</h3>
                            <p><strong>Error:</strong> ${data.message}</p>
                            <p><strong>Instructions:</strong> ${data.instructions}</p>
                        </div>
                    `;
                }
            } catch (error) {
                console.error('Health check failed:', error);
            }
        }
        
        function switchTab(tab) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            
            if (tab === 'text') {
                document.querySelector('.tab:first-child').classList.add('active');
                document.getElementById('text-tab').classList.add('active');
            } else {
                document.querySelector('.tab:last-child').classList.add('active');
                document.getElementById('pdf-tab').classList.add('active');
            }
        }
        
        function handleFileSelect(event) {
            selectedFiles = Array.from(event.target.files);
            const fileNameDiv = document.getElementById('fileName');
            
            if (selectedFiles.length > 0) {
                const names = selectedFiles.map(f => f.name).join(', ');
                fileNameDiv.textContent = `Selected: ${names}`;
            } else {
                fileNameDiv.textContent = '';
            }
        }
        
        async function uploadPDFs() {
            const statusDiv = document.getElementById('status');
            
            if (selectedFiles.length === 0) {
                statusDiv.innerHTML = '<div class="status error">Please select PDF files first.</div>';
                return;
            }
            
            const formData = new FormData();
            selectedFiles.forEach(file => {
                formData.append('files', file);
            });
            
            statusDiv.innerHTML = '<div class="status">Uploading and processing PDFs...</div>';
            
            try {
                const response = await fetch('/upload_pdfs', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.success) {
                    statusDiv.innerHTML = `<div class="status success">${data.message}</div>`;
                    selectedFiles = [];
                    document.getElementById('pdfInput').value = '';
                    document.getElementById('fileName').textContent = '';
                    checkHealth(); // Refresh health status
                } else {
                    statusDiv.innerHTML = `<div class="status error">Error: ${data.error}</div>`;
                }
            } catch (error) {
                statusDiv.innerHTML = `<div class="status error">Error: ${error.message}</div>`;
            }
        }
        
        async function addDocuments() {
            const docs = document.getElementById('documents').value;
            const statusDiv = document.getElementById('status');
            
            if (!docs.trim()) {
                statusDiv.innerHTML = '<div class="status error">Please enter some documents.</div>';
                return;
            }
            
            const docArray = docs.split(/\\n\\s*\\n/).filter(d => d.trim());
            
            try {
                const response = await fetch('/add_documents', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({documents: docArray})
                });
                
                const data = await response.json();
                
                if (data.success) {
                    statusDiv.innerHTML = `<div class="status success">${data.message}</div>`;
                    document.getElementById('documents').value = '';
                    checkHealth(); // Refresh health status
                } else {
                    statusDiv.innerHTML = `<div class="status error">Error: ${data.error}</div>`;
                }
            } catch (error) {
                statusDiv.innerHTML = `<div class="status error">Error: ${error.message}</div>`;
            }
        }
        
        async function clearDatabase() {
            if (!confirm('Are you sure you want to clear all documents from the database?')) {
                return;
            }
            
            const statusDiv = document.getElementById('status');
            
            try {
                const response = await fetch('/clear_database', {
                    method: 'POST'
                });
                
                const data = await response.json();
                
                if (data.success) {
                    statusDiv.innerHTML = `<div class="status success">${data.message}</div>`;
                    checkHealth(); // Refresh health status
                } else {
                    statusDiv.innerHTML = `<div class="status error">Error: ${data.error}</div>`;
                }
            } catch (error) {
                statusDiv.innerHTML = `<div class="status error">Error: ${error.message}</div>`;
            }
        }
        
        async function askQuestion() {
            const question = document.getElementById('question').value;
            const responseDiv = document.getElementById('response');
            const loadingDiv = document.getElementById('loading');
            
            if (!question.trim()) {
                alert('Please enter a question.');
                return;
            }
            
            loadingDiv.style.display = 'block';
            responseDiv.innerHTML = '';
            
            try {
                const response = await fetch('/query', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({question: question})
                });
                
                const data = await response.json();
                
                loadingDiv.style.display = 'none';
                
                if (data.answer) {
                    const formattedAnswer = marked.parse(data.answer);
                    
                    let html = `
                        <div class="answer-section">
                            <h3>Answer:</h3>
                            <div class="answer-text">${formattedAnswer}</div>
                        </div>
                    `;
                    
                    if (data.sources && data.sources.length > 0) {
                        html += '<div class="sources"><h4>Sources:</h4>';
                        data.sources.forEach((source, idx) => {
                            html += `<div class="source-item"><strong>Source ${idx + 1}:</strong> ${source.text.substring(0, 200)}...</div>`;
                        });
                        html += '</div>';
                    }
                    
                    responseDiv.innerHTML = html;
                } else if (data.detail) {
                    // Handle HTTPException detail format
                    const errorMsg = typeof data.detail === 'object' ? 
                        `${data.detail.message || data.detail.error}<br><small>${data.detail.instructions || ''}</small>` : 
                        data.detail;
                    responseDiv.innerHTML = `<div class="status error">${errorMsg}</div>`;
                } else {
                    responseDiv.innerHTML = `<div class="status error">Error: ${data.error || 'Unknown error'}</div>`;
                }
            } catch (error) {
                loadingDiv.style.display = 'none';
                responseDiv.innerHTML = `<div class="status error">Error: ${error.message}</div>`;
            }
        }
        
        document.getElementById('question').addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                askQuestion();
            }
        });
    </script>
</body>
</html>
"""

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the service is properly configured."""
    if rag is None:
        return HealthResponse(
            status="error",
            configured=False,
            message="ANTHROPIC_API_KEY not configured",
            instructions="Set ANTHROPIC_API_KEY environment variable and restart"
        )
    
    try:
        collection_info = rag.qdrant_client.get_collection(rag.collection_name)
        return HealthResponse(
            status="healthy",
            configured=True,
            documents_count=collection_info.points_count
        )
    except Exception as e:
        return HealthResponse(
            status="error",
            configured=True,
            message=str(e)
        )

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML_TEMPLATE

@app.post("/add_documents", response_model=StatusResponse)
async def add_documents(
    data: DocumentsInput,
    rag_system: RAGSystem = Depends(require_configured_rag)
):
    try:
        documents = data.documents
        
        if not documents:
            raise HTTPException(status_code=400, detail="No documents provided")
        
        rag_system.add_documents(documents)
        
        collection_info = rag_system.qdrant_client.get_collection(rag_system.collection_name)
        
        return StatusResponse(
            success=True,
            message=f"Successfully added {len(documents)} document(s). Total points: {collection_info.points_count}"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_pdfs", response_model=StatusResponse)
async def upload_pdfs(
    files: List[UploadFile] = File(...),
    rag_system: RAGSystem = Depends(require_configured_rag)
):
    try:
        documents = []
        
        for file in files:
            if file.filename and file.filename.endswith('.pdf'):
                pdf_bytes = await file.read()
                text = extract_text_from_pdf(pdf_bytes)
                documents.append(text)
        
        if not documents:
            raise HTTPException(status_code=400, detail="No valid PDF files found")
        
        rag_system.add_documents(documents)
        
        collection_info = rag_system.qdrant_client.get_collection(rag_system.collection_name)
        
        return StatusResponse(
            success=True,
            message=f"Successfully processed {len(files)} PDF(s). Total points: {collection_info.points_count}"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear_database", response_model=StatusResponse)
async def clear_database(
    rag_system: RAGSystem = Depends(require_configured_rag)
):
    try:
        rag_system.clear_database()
        return StatusResponse(
            success=True,
            message="Database cleared successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(
    data: QuestionInput,
    rag_system: RAGSystem = Depends(require_configured_rag)
):
    try:
        question = data.question
        
        if not question:
            raise HTTPException(status_code=400, detail="No question provided")
        
        result = rag_system.query(question)
        
        return QueryResponse(**result)
    except HTTPException:
        raise
    except anthropic.AuthenticationError:
        raise HTTPException(
            status_code=401,
            detail="Invalid Anthropic API key. Please check your configuration."
        )
    except anthropic.RateLimitError:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    print("\n" + "="*50)
    print("RAG System with Qdrant Vector Database (FastAPI)")
    print("="*50)
    print("\nStarting server at http://localhost:8000")
    print("Press CTRL+C to stop\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)