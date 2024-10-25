import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from database import ChromaDBManager

# Define logging rules
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Define default constant values
DEFAULT_DB_PATH: str = "../../../chromadb"  # Заменить на "/app/db"
DEFAULT_EMBEDDING_MODEL: str = "cointegrated/rubert-tiny2"
DEFAULT_COLLECTION_NAME: str = "rag_collection"  # Заменить на что угодно
DEFAULT_DATASET_PATH: str = "../../../ml-potw-10232023.csv"
DEFAULT_DOCS_DIR: str = ...

# Define manager and collection
manager = ChromaDBManager(db_path=DEFAULT_DB_PATH)
collection = manager.get_or_create_collection(collection_name=DEFAULT_COLLECTION_NAME,
                                           embedding_model=DEFAULT_EMBEDDING_MODEL)

# Define lifespan function
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Define database manager and client
    # manager = ChromaDBManager(db_path=DEFAULT_DB_PATH)
    # collection = manager.get_or_create_collection(collection_name=DEFAULT_COLLECTION_NAME,
    #                                        embedding_model=DEFAULT_EMBEDDING_MODEL)
    # Client and collection are already created
    # You only need to use them with existing manager
    # 1. Check if client is available
    logging.info(f"Client heartbeat: {manager.client.heartbeat()}")
    # 2. Check if collection is available
    logging.info(f"List of collections: {manager.list_collections()}.")
    logging.info(f"Name of current collection is: {DEFAULT_COLLECTION_NAME}.")
    logging.info(f"{'TRUE' if DEFAULT_COLLECTION_NAME in manager.list_collections() else 'FALSE'}.")
    # 2. Insert base docs if they are not inserted
    logging.info("Checking default docs")
    # default_docs_present = ...
    # if default_docs_present:
    #     logging.info("Default docs are already in the database")
    # else:
    #     ...
        # Some function to read the DEFAULT_DOCS
        # When you have List[TextInput] of DEFAULT_DOCS
        # default_documents = ...
        # for document in default_documents:
        #     manager.add_document(document)

    yield
    logging.info("Shutting down ChromaDB client...")
    exit(52)


# Define FastAPI app
app = FastAPI(lifespan=lifespan)
# app.include_router(db_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8003, reload=True)
