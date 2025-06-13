from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents():
    file_path = "./data/IndusValleyReport2024.pdf"
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    return docs

docs = chunk_documents()
splits = split_documents(docs)
print(splits[2].page_content)


from dotenv import load_dotenv
import os

load_dotenv()  # Loads variables from .env into environment

# Now you can access your keys anywhere in your code:
openai_api_key = os.getenv("OPENAI_API_KEY")


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