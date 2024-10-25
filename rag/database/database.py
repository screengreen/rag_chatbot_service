import json
import logging
from pathlib import Path
from typing import Union, Dict

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from rag.database.datatypes import TextInput


class ChromaDBManager:
    DEFAULT_DB_NAME: str = "rag_database"
    DEFAULT_DB_PATH: str = "../chromadb"
    DEFAULT_COLLECTION_NAME: str = "rag_collection"
    DEFAULT_METADATA: Dict[str, str] = {"hnsw:space": "cosine"}

    # Creates client on instance
    def __init__(
            self,
            db_path: Union[str, Path] = DEFAULT_DB_PATH,
            verbose: bool = False
    ):
        # Define default/custom variables
        self.db_path = db_path
        self.verbose = verbose

        # Initialize client
        self.client = chromadb.PersistentClient(path=self.db_path, settings=Settings(anonymized_telemetry=False))

        # Define future elements
        self.collection = None
        self.embedding_fn = None

    def get_or_create_collection(self, collection_name: str, embedding_model: Union[str, None] = None):
        if self.verbose:
            logging.info("Getting or creating collection")
        embd_fn = self._load_embedding_fn(embedding_model)
        self.collection = self.client.get_or_create_collection(collection_name,
                                                               metadata={"hnsw:space": "cosine"})
        return self.collection

    def get_collection(self, collection_name: str):
        if self.verbose:
            logging.info("Getting collection")
        assert collection_name in [col.name for col in self.list_collections()]
        return self.client.get_collection(collection_name)

    def add_document(self, text: TextInput):
        if self.verbose:
            logging.info("Adding document")
        assert self.collection is not None
        self.collection.upsert(
            embeddings=self.embedding_fn(text.text),
            # documents=text.document,
            metadatas={"language": text.language, "origin": text.origin},
            ids=str(hash(text.text)),
        )

    def remove_document(self):
        if self.verbose:
            logging.info("Removing document")
        assert self.collection is not None
        # TODO: Add removal func
        pass

    def get_similar_texts(self, text: TextInput, n_results: int = 2):
        if self.verbose:
            logging.info("Getting similar texts")
        assert self.collection is not None
        query_embd = self._get_embeddings(text.text)
        # TODO: Add 'where=' possibility
        return self.collection.query(
            query_embeddings=query_embd,
            n_results=n_results,
            # where=...
        )

    def list_collections(self):
        if self.verbose:
            logging.info("Listing collections")
        return self.client.list_collections()

    def delete_collection(self, collection_name: str):
        if self.verbose:
            logging.info("Deleting collection")
        self.client.delete_collection(collection_name)

    def _load_embedding_fn(self, model_name: Union[str, None] = None):
        if self.verbose:
            logging.info(f"Loading embedding model")
        if model_name is None:
            # TODO: Change device
            self.embedding_fn = SentenceTransformer(model_name_or_path="cointegrated/rubert-tiny2", device="cpu").encode
        else:
            self.embedding_fn = SentenceTransformer(model_name_or_path=model_name, device="cpu").encode
        return self.embedding_fn

    def _get_embeddings(self, text: str):
        if self.verbose:
            logging.info("Getting embeddings")
        return self.embedding_fn(text)

    def count_collection(self):
        if self.verbose:
            logging.info("Counting collection")
        assert self.collection is not None
        return self.collection.count()

    def rename_collection(self, new_name: str):
        if self.verbose:
            logging.info("Renaming collection")
        assert self.collection is not None
        self.collection.modify(name=new_name)

    # Function to load default documents to database
    # Should be without args
    def load_defaults(self):
        if self.verbose:
            logging.info("Loading default documents")
        assert self.client is not None
        assert self.collection is not None
        # TODO: Implement default docs loading
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    path = "/Users/fffgson/Desktop/Coding/turbohack/"

    text1 = TextInput(text="я дебил", language="en", origin="pukpuk", default=True, document="privetpoka")
    # text2 = TextInput(text="ti durak", language="ru", origin="text2", default=True)

    manager = ChromaDBManager(db_path=path)
    manager.get_or_create_collection("rag_collection")
    logging.info(manager.list_collections())
    logging.info(manager.get_collection("rag_collection"))
    manager.add_document(text1)
    # manager.add_document(text2)
    logging.info(manager.count_collection())
    query_result = manager.get_similar_texts(TextInput("дурак"), n_results=5)
    json = json.dumps(query_result, indent=4)
    logging.info(json)
    logging.info(manager.collection.get(where={"language": "ru"}))
    logging.info(len(query_result))
    manager.delete_collection("rag_collection")
    logging.info(manager.list_collections())
    exit(52)
