# Document RAG System

A modern Retrieval-Augmented Generation (RAG) system for querying documents using Milvus vector database, Gemini embeddings, and Claude for intelligent text generation. Features a beautiful full-screen chat interface and modular, maintainable code architecture.

## ✨ Features

- **📄 Document Processing**: Intelligent loading and chunking of PDF documents
- **🔍 Vector Search**: High-performance semantic search using Gemini embeddings and Milvus
- **🤖 RAG Pipeline**: Advanced retrieval-augmented generation with conversation history
- **🌐 Web Interface**: Beautiful, responsive chat UI served directly from FastAPI
- **📱 Full-Screen Design**: Modern, mobile-friendly interface with typing indicators
- **🔄 Conversation Memory**: Maintains chat history for contextual follow-up questions
- **⚡ FastAPI Backend**: High-performance API with automatic documentation
- **🏗️ Modular Architecture**: Clean, maintainable code with separated concerns
- **🔧 Debug Tools**: Comprehensive logging and monitoring capabilities

## 🏗️ Project Structure

```
research-papers-rag/
├── src/
│   ├── main.py                    # Core RAG logic and pipeline
│   ├── api.py                     # FastAPI server with UI integration
│   ├── db_config.py               # Milvus database configuration
│   ├── prompt_templates/
│   │   └── templates.py           # Custom prompt templates
│   ├── data/                      # Place your PDF files here
│   └── __pycache__/
├── ui/
│   └── index.html                 # Full-screen chat interface
├── requirement.txt                # Python dependencies
├── .gitignore                     # Git ignore rules (includes macOS files)
└── README.md                      # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirement.txt
```

### 2. Environment Setup

Create a `.env` file in the root directory:

```env
GEMINI_API_KEY=your_gemini_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### 3. Prepare Your Documents

Place your PDF files in the `src/data/` directory. The system will automatically process them.

### 4. Start the Application

```bash
cd src
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### 5. Open Your Browser

Visit `http://localhost:8000` to access the full-screen chat interface!

## 🎯 How It Works

### RAG Pipeline Architecture

1. **Document Ingestion**: PDFs are loaded and chunked into manageable pieces
2. **Vector Embeddings**: Gemini creates embeddings for each chunk
3. **Vector Storage**: Milvus stores embeddings for fast similarity search
4. **Query Processing**: User questions are embedded and searched
5. **Context Retrieval**: Most relevant chunks are retrieved with relevance scores
6. **Response Generation**: Claude generates answers using context and chat history

### Code Organization

- **`prepare_context_from_search_results()`**: Formats search results into readable context
- **`prepare_chat_history()`**: Converts conversation history to prompt format
- **`execute_rag_pipeline()`**: Orchestrates the complete RAG process
- **`ask_question()`**: Simple API interface for question answering

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves the full-screen chat interface |
| `/health` | GET | Health check endpoint |
| `/ask` | POST | Ask questions with conversation history |
| `/setup` | POST | Initialize database with documents |

### Example API Usage

```bash
# Ask a question
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "What is the main topic of the document?",
       "history": [
         {"role": "user", "content": "Tell me about this document"},
         {"role": "assistant", "content": "This document discusses..."}
       ]
     }'

# Setup database
curl -X POST "http://localhost:8000/setup"
```

## 🔧 Technical Details

### Core Technologies
- **Vector Database**: Milvus with COSINE similarity
- **Embeddings**: Google Gemini `models/embedding-001`
- **Language Model**: Anthropic Claude 3 Opus
- **Web Framework**: FastAPI with static file serving
- **Frontend**: Vanilla HTML/CSS/JavaScript
- **Document Processing**: LangChain PDF loader and text splitter

### Performance Optimizations
- **Batch Processing**: Efficient embedding generation in batches
- **Indexed Search**: Fast vector similarity search
- **Context Scoring**: Relevance-based context selection
- **Memory Management**: Optimized conversation history handling

### Configuration
- **Chunk Size**: 800 characters with 200 character overlap
- **Search Results**: Top 8 most relevant chunks
- **Embedding Dimension**: 768 (Gemini default)
- **Similarity Metric**: COSINE for semantic search

## 🐛 Debugging & Monitoring

## Troubleshooting

1. **"uvicorn command not found"**: Install with `pip install uvicorn`
2. **API connection errors**: Make sure the server is running on port 8000
3. **Setup failures**: Check your API keys and ensure PDF files are in the data directory
4. **Milvus connection issues**: Ensure Milvus is running and accessible

- **Environment Variables**: API keys stored securely in `.env`
- **CORS Configuration**: Properly configured for development
- **Error Handling**: Graceful error responses without exposing internals
- **Input Validation**: Pydantic models for request validation

- The `main.py` file contains all core logic and can be run independently for testing
- The `api.py` file wraps the main functions in FastAPI endpoints
- The HTML interface is self-contained and can be easily modified

### Local Development
```bash
cd src
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

## Functional Requirements

The project aims to achieve the following:

1.  **Vectorize PDF Documents**: Specifically, the "Attention is All You Need" paper will be processed.
2.  **Store Embeddings in MilvusDB**: Embeddings generated from the PDF will be persisted in a Milvus vector database.
3.  **Create a Streamlit Minimal UI**: A user-friendly interface will be developed using Streamlit for interaction with the RAG system.
4.  **Deploy the Application**: The complete RAG application will be deployed for accessibility.

---

## Technology Stack

- **Large Language Model (LLM)**: Claude-Sonnet-4
- **Embeddings Model**: Gemini `embedding-001`
- **Vector Index Type**: `AUTOINDEX` (Milvus will automatically select the optimal index algorithm)
- **Vector Metric Type**: `COSINE` similarity

---

## Milvus DB Schema

The Milvus collection will store vectorized document chunks with the following schema:

- **`chunk_id`** (Primary Key, INT64): A unique identifier for each text chunk, automatically generated by Milvus.
- **`document_id`** (VARCHAR): A unique identifier for the document to which the chunk belongs (e.g., document's filename or a UUID). This aids in grouping all chunks from the same source document.
- **`document_title`** (VARCHAR): The title of the source document, typically extracted from the PDF's metadata.
- **`chunk_text`** (VARCHAR): The actual textual content of the chunk (`page_content` from Langchain).
- **`embedding`** (FLOAT_VECTOR): The high-dimensional vector representation of the `chunk_text`, generated by the Gemini `embedding-001` model.
- **`page_number`** (INT64): The original page number in the PDF where the chunk was found, providing valuable contextual information.
- **`source_filename`** (VARCHAR): The name of the original PDF file (corresponding to the `source` metadata field), useful for identifying the document.

---
