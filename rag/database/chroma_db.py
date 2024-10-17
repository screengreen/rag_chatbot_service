import chromadb
from chromadb import EmbeddingFunction, Embeddings, Documents
from sentence_transformers import SentenceTransformer
import logging
import pandas as pd
import os
from tqdm import tqdm
import random
from typing import List

# Define logging rules
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Define embedding function class
class MyEmbeddingFunction(EmbeddingFunction):
    def __init__(self, embedding_model: SentenceTransformer):
        self.embedding_model = embedding_model
        
    def __call__(self, input: Documents) -> Embeddings:
        batch_embeddings = self.embedding_model.encode(input)
        return batch_embeddings.tolist()
    
    
# Constants
chromadb_path: str = f"../chromadb"
sentence_transformer: str = f"cointegrated/rubert-tiny2"
collection_name: str = f"test-collection"
dataset_path: str = f"../ml-potw-10232023.csv"
batch_size: int = 10
query_texts: List[str] = ["LLM", "python", "object detection"]

# Code
if __name__ == "__main__":
    assert os.path.isfile(dataset_path)
    assert batch_size > 0
    assert len(query_texts) > 0
    assert len(chromadb_path) > 0
    assert len(sentence_transformer) > 0
    assert len(collection_name) > 0
    assert len(dataset_path) > 0
    if not os.path.exists(chromadb_path):
        os.makedirs(chromadb_path)
    
    logging.info(f"Reading dataset")
    dataset_df = pd.read_csv(dataset_path, header=0)
    dataset_df = dataset_df.dropna(subset=["Title", "Description"])
    logging.info(f"Dataset size: {dataset_df.shape}")
    dataset_dict = dataset_df.to_dict(orient="records")
    
    logging.info(f"Initializing ChromaDB client")
    client = chromadb.PersistentClient(path=chromadb_path)

    logging.info(f"Initializing embedding model")
    embedding_model = SentenceTransformer(sentence_transformer)

    logging.info(f"Initializing embedding function with selected embedding model")
    embed_fn = MyEmbeddingFunction(embedding_model=embedding_model)

    logging.info(f"Creating collection")
    collection = client.get_or_create_collection(name=collection_name, embedding_function=embed_fn, metadata={"hnsw:space":"cosine"})
    
    # Loop through batches and generate + store embeddings
    for i in tqdm(range(0, len(dataset_dict), batch_size)):

        i_end = min(i + batch_size, len(dataset_dict))
        batch = dataset_dict[i : i + batch_size]

        # Replace title with "No Title" if empty string
        batch_titles = [str(paper["Title"]) if str(paper["Title"]) != "" else "No Title" for paper in batch]
        batch_ids = [str(sum(ord(c) + random.randint(1, 10000) for c in paper["Title"])) for paper in batch]
        batch_metadata = [dict(url=paper["PaperURL"],
                            abstract=paper['Abstract'])
                            for paper in batch]

        # Generate embeddings
        batch_embeddings = embedding_model.encode(batch_titles)

        # Upsert to chromadb
        collection.upsert(
            ids=batch_ids,
            metadatas=batch_metadata,
            documents=batch_titles,
            embeddings=batch_embeddings.tolist(),
        )
    
    retriever_results = collection.query(query_texts=query_texts, n_results=2)
    
    logging.info(retriever_results["documents"])
    
    logging.info(f"Process finished")