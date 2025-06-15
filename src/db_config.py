from pymilvus import CollectionSchema, DataType, MilvusClient, FieldSchema

class MilvusDBConfig: 
    DB_URI = "research_papers.db"
    COLLECTION_NAME = "research_papers_collection"
    EMBEDDING_DIM = 768
    
    def __init__(self, db_path: str = DB_URI):
        self.client = MilvusClient(uri=db_path)
        self._initialise_collection()    
        
    def _initialise_collection(self):
        fields = [
            # Primary Key field
            FieldSchema(name="chunk_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            # Other metadata fields
            FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="document_title", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=4096), # Adjust max_length as needed
            FieldSchema(name="page_number", dtype=DataType.INT64),
            FieldSchema(name="source_filename", dtype=DataType.VARCHAR, max_length=256),
            # Vector field
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.EMBEDDING_DIM)
        ]

        schema = CollectionSchema(fields, description="Document chunks with Gemini embeddings for RAG")
        
        # Check if the collection already exists and drop it if we're creating with a new schema
        if self.client.has_collection(collection_name=self.COLLECTION_NAME):
            print(f"Collection '{self.COLLECTION_NAME}' already exists. Dropping and recreating with new schema.")
            self.client.drop_collection(collection_name=self.COLLECTION_NAME)
        else:
            print(f"Collection '{self.COLLECTION_NAME}' does not exist. Creating...")

        # Create the collection with the defined schema
        self.client.create_collection(
            collection_name=self.COLLECTION_NAME,
            schema=schema
        )
        print("Collection created successfully with custom schema.")
        
    def get_client(self):
        return self.client
    
    def get_collection_name(self):
        return self.COLLECTION_NAME
    
    def load_collection(self):
        print(f"Loading collection '{self.COLLECTION_NAME}' into memory...")
        self.client.load_collection(collection_name=self.COLLECTION_NAME)
        print("Collection loaded successfully.")
        
    def close(self):
        print("Closing Milvus client connection.")
        self.client.close()