import os
import chromadb
from chromadb import EmbeddingFunction, Embeddings, Documents
from sentence_transformers import SentenceTransformer
import logging

# Класс функции создания эмбеддингов
class MyEmbeddingFunction(EmbeddingFunction):
    def __init__(self, embedding_model: SentenceTransformer):
        self.embedding_model = embedding_model
        
    def __call__(self, input: Documents) -> Embeddings:
        batch_embeddings = self.embedding_model.encode(input)
        return batch_embeddings.tolist()
    
# Initialize the chromadb directory, and client.
client = chromadb.PersistentClient(path="./chromadb")
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    logging.info(f"Initializeing embedding model")
    embedding_model = SentenceTransformer("cointegrated/rubert-tiny2")

    embed_fn = MyEmbeddingFunction(embedding_model=embedding_model)

    # create collection
    collection = client.get_or_create_collection(
        name=f"test-collection"
)
    logging.info(f"Process finished")