from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import logging
import os

# from chromadb import EmbeddingFunction, Embeddings, Documents


# Define logging rules
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')




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



    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
