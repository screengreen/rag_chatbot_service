from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import logging
import os

from router import db_router
# from chromadb import EmbeddingFunction, Embeddings, Documents


# Define logging rules
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')



# Constants
chromadb_path: str = f"../chromadb"
sentence_transformer: str = f"cointegrated/rubert-tiny2"
collection_name: str = f"test-collection"
dataset_path: str = f"../ml-potw-10232023.csv"
batch_size: int = 10
# query_texts: List[str] = ["LLM", "python", "object detection"]




@asynccontextmanager
async def lifespan(app: FastAPI):
    #check if db else create one
   
    #insert base docs if they are not inserted

    # logging.info(f"Loading embedding model")
    # EmbModelLoader.load_model()

    yield
    logging.info(f"shutting down chromadb")

app = FastAPI(lifespan=lifespan)
app.include_router(db_router)



# Запуск приложения
if __name__ == "__main__":
    # assert os.path.isfile(dataset_path)
    # assert batch_size > 0
    # # assert len(query_texts) > 0
    # assert len(chromadb_path) > 0
    # assert len(sentence_transformer) > 0
    # assert len(collection_name) > 0
    # assert len(dataset_path) > 0

    if not os.path.exists(chromadb_path):
        os.makedirs(chromadb_path)


    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
