# research-papers-rag

This repository is to understand the RAG workflow end-to-end with deployment

Functional Requirement

1. Vectorise the PDF - Attention is All you Need
2. Store in MilvusDB
3. Create streamlit minimal UI
4. Deploy the App

Model : Claude-Sonnet-4
Embeddings Model : Gemini embedding-001
Similarity Algorithm : Cosine-similarity

# Milvus DB Schema

chunk_id (Primary Key, int64): A unique identifier for each text chunk.
document_id (varchar): A unique identifier for the document the chunk belongs to. This is useful for grouping all chunks from the same document.(document's filename or a UUID.)
document_title (varchar): The title of the source document from the PDF's metadata.
chunk_text (varchar): The actual text content of the chunk (page_content).
embedding (Vector, float_vector): The vector representation of the chunk_text.
page_number (int64): The page number where the chunk originated. This is valuable context to show the user.
source_filename (varchar): The name of the original file (source field), which helps in identifying the document.
