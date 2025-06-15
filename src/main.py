from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_and_split_documents():
    file_path = "./data/IndusValleyReport2024.pdf"
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    return splits

splits = chunk_and_split_documents()
print("split 7:\n", splits[7])


from dotenv import load_dotenv
import os

load_dotenv()  # Loads variables from .env into environment

# Now you can access your keys anywhere in your code:
gemini_api_key = os.getenv("GEMINI_API_KEY")


from langchain_anthropic import ChatAnthropic
model = ChatAnthropic(model='claude-3-opus-20240229',api_key=os.getenv("ANTHROPIC_API_KEY"))

from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that answers questions about the Indus Valley Civilization."),
    ("human", "{question}")
])

chain = prompt | model

# Function to ask a question and get a response from the model
def ask_question(question): 
    response = chain.invoke({"question": question})
    return response.content

ask_question("What were the main features of the Indus Valley Civilization's urban planning?")

import google.generativeai as genai

EMBEDDING_MODEL = "models/embedding-001"
EMBEDDING_DIM = 768
embeddings = []
batch_size = 100 # Maximum batch size allowed by Gemini Embeddings API
genai.configure(api_key=gemini_api_key)
print(f"Generating embeddings for document chunks using {EMBEDDING_MODEL}...")

# create embeddings and store in the database
def get_embbedings():
    print("Generating embeddings for document chunks...")
    texts_to_embed = [doc.page_content for doc in splits]
    # print((texts_to_embed[89]))
    print(f"Successfully generated {len(embeddings)} embeddings.")
    for i in range(0, len(texts_to_embed), batch_size):
        batch_texts = texts_to_embed[i:i + batch_size]
        print(f"Processing batch {int(i/batch_size) + 1}/{(len(texts_to_embed) + batch_size - 1) // batch_size} (Texts {i+1} to {min(i + batch_size, len(texts_to_embed))})...")
        try:
            response = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=batch_texts,
                task_type="RETRIEVAL_DOCUMENT"
            )
            embeddings.extend([embedding for embedding in response['embedding']])
        except Exception as e:
            print(f"Error generating embeddings for batch starting at index {i}: {e}")
            # Depending on your error handling strategy, you might want to break,
            # log more details, or retry.
            break # Stop processing if an error occurs

    if len(embeddings) == len(texts_to_embed):
        print(f"Successfully generated {len(embeddings)} embeddings.")
    else:
        print(f"Warning: Only generated {len(embeddings)} embeddings out of {len(texts_to_embed)} due to error or partial processing.")


get_embbedings()

