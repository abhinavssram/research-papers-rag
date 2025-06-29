from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import google.generativeai as genai
from db_config import MilvusDBConfig
from prompt_templates.templates import get_system_prompt, get_user_prompt

# Load environment variables
load_dotenv()

# Initialize API keys
gemini_api_key = os.getenv("GEMINI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

# Initialize models
model = ChatAnthropic(model='claude-3-opus-20240229', api_key=anthropic_api_key)
genai.configure(api_key=gemini_api_key)

# Initialize database config
db_config = MilvusDBConfig()
client = db_config.get_client()

def chunk_and_split_documents():
    """Load and split documents into chunks"""
    file_path = "./data/IndusValleyReport2024.pdf"
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    return splits

def get_embeddings(texts_to_embed):
    """Generate embeddings for given texts"""
    EMBEDDING_MODEL = "models/embedding-001"
    batch_size = 100
    embeddings = []
    
    print(f"Generating embeddings for {len(texts_to_embed)} document chunks...")
    
    for i in range(0, len(texts_to_embed), batch_size):
        batch_texts = texts_to_embed[i:i + batch_size]
        print(f"Processing batch {int(i/batch_size) + 1}/{(len(texts_to_embed) + batch_size - 1) // batch_size}...")
        try:
            response = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=batch_texts,
                task_type="RETRIEVAL_DOCUMENT"
            )
            embeddings.extend([embedding for embedding in response['embedding']])
        except Exception as e:
            print(f"Error generating embeddings for batch starting at index {i}: {e}")
            break
    
    return embeddings

def setup_database():
    """Setup function to load documents, create embeddings, and insert into Milvus"""
    print("Starting database setup...")
    
    # Load and split documents
    splits = chunk_and_split_documents()
    print(f"Split documents into {len(splits)} chunks")
    
    # Generate embeddings
    texts_to_embed = [doc.page_content for doc in splits]
    embeddings = get_embeddings(texts_to_embed)
    
    if len(embeddings) != len(texts_to_embed):
        print(f"Warning: Only generated {len(embeddings)} embeddings out of {len(texts_to_embed)}")
        return False
    
    # Prepare data for insertion
    document_ids = ['IndusValleyReport2024']
    data_to_insert = []
    
    for i, doc in enumerate(splits):
        if i < len(embeddings):
            data_to_insert.append({
                "document_id": document_ids[0],
                "document_title": document_ids[0],
                "chunk_text": doc.page_content,
                "embedding": embeddings[i],
                "page_number": doc.metadata.get('page', -1),
                "source_filename": os.path.basename(doc.metadata.get('source', 'No Source'))
            })
    
    # Insert into Milvus
    print(f"Inserting {len(data_to_insert)} entities into Milvus...")
    if data_to_insert:
        res = client.insert(
            collection_name=db_config.COLLECTION_NAME,
            data=data_to_insert
        )
        print(f"Successfully inserted {res['insert_count']} entities.")
        
        # Create index
        print("Creating index for vector field...")
        index_params = client.prepare_index_params(
            field_name="embedding",
            index_type="AUTOINDEX",
            metric_type="COSINE"
        )
        client.create_index(
            collection_name=db_config.COLLECTION_NAME,
            index_params=index_params
        )
        print("Index creation request sent.")
        
        # Load collection
        db_config.load_collection()
        print("Database setup completed successfully!")
        return True
    else:
        print("No data to insert.")
        return False

def search_similar_chunks(question, top_k=8):
    """Search for similar chunks in the database"""
    # Generate embedding for the question
    question_embedding = genai.embed_content(
        model="models/embedding-001",
        content=[question],
        task_type="RETRIEVAL_QUERY"
    )['embedding'][0]
    
    # Search in Milvus - fixed parameter structure
    results = client.search(
        collection_name=db_config.COLLECTION_NAME,
        data=[question_embedding],
        anns_field="embedding",
        limit=top_k,
        output_fields=["chunk_text", "page_number", "document_title"],
        search_params={"metric_type": "COSINE", "params": {"nprobe": 10}}
    )
    
    return results[0]  # Return the first (and only) query result

def prepare_context_from_search_results(search_results):
    """Prepare formatted context from search results"""
    context_parts = []
    for i, result in enumerate(search_results):
        chunk_text = result.entity.get('chunk_text', '')
        page_num = result.entity.get('page_number', -1)
        score = result.score
        context_parts.append(f"[Page {page_num}, Relevance: {score:.3f}]\n{chunk_text}")
    
    return "\n\n---\n\n".join(context_parts)

def prepare_chat_history(history):
    """Format chat history for the prompt"""
    if not history:
        return ""
    
    history_text = "Previous conversation:\n"
    for msg in history:
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        history_text += f"{role.capitalize()}: {content}\n"
    
    return history_text

def debug_rag_components(question: str, search_results, context: str, history_text: str):
    """Debug function to monitor RAG pipeline components"""
    print("\n" + "="*50)
    print("RAG PIPELINE DEBUG INFO")
    print("="*50)
    print(f"Question: {question}")
    print(f"Search Results Count: {len(search_results)}")
    print(f"Context Length: {len(context)} characters")
    print(f"History Length: {len(history_text)} characters")
    
    if search_results:
        print(f"Top Result Score: {search_results[0].score:.3f}")
        print(f"Top Result Page: {search_results[0].entity.get('page_number', -1)}")
    
    print("="*50 + "\n")

def execute_rag_pipeline(question: str, history: list[dict[str, str]] = None):
    """Execute the complete RAG pipeline: search, prepare context, and generate response"""
    try:
        # Default to empty history if none provided
        if history is None:
            history = []
        
        # Step 1: Search for relevant chunks
        search_results = search_similar_chunks(question)
        
        # Step 2: Prepare context and history
        context = prepare_context_from_search_results(search_results)
        history_text = prepare_chat_history(history)
        
        # Step 3: Debug logging
        debug_rag_components(question, search_results, context, history_text)
        
        # Step 4: Create and execute prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", get_system_prompt()),
            ("human", get_user_prompt())
        ])
        
        chain = prompt | model
        
        # Step 5: Generate response
        response = chain.invoke({
            "chat_history": history_text,
            "context": context,
            "question": question
        })
        
        return response.content
        
    except Exception as e:
        print(f"Error in RAG pipeline: {e}")
        return f"Sorry, I encountered an error: {str(e)}"

def ask_question(question: str, history: list[dict[str, str]] = None):
    """Main function to answer questions using RAG"""
    return execute_rag_pipeline(question, history)

# For testing - only run setup if this file is run directly
if __name__ == "__main__":
    # Uncomment the line below to run setup (only needed once)
    setup_database()
    
    # Test the question answering
    test_question = "What is Indus Valley and how does it relate to the Indian startup ecosystem?"
    print(f"Testing with question: {test_question}")
    answer = ask_question(test_question, [])
    print(f"Answer: {answer}")