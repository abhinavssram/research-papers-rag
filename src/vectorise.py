class Vectorize:
    
    def __init__(self, dbConnection):
        pass
    
    def read(self, name):
        # read documents from file path
        pass
    
    def splitIntoChunks(self, document):
        # Recursively split into chunks 
        chunks = []
        return chunks
        
    def storeChunksInDB(self,chunks):
        self.dbconnection.save(chunks)