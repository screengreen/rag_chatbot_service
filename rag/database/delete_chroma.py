import chroma_db
import os
import shutil
import logging

# Define logging rules
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Constants
chromadb_path: str = f"../chromadb"
collection_name: str = f"test-collection"

# Code
if __name__ == "__main__":
    assert len(chromadb_path) > 0
    assert len(collection_name) > 0
    
    if os.path.exists(chromadb_path):
        client = chroma_db.PersistentClient(path=chromadb_path)
        collection = client.get_collection(name=collection_name)
        collection.count()
        choice = input("Do you really want to delete the whole database? (y/n): ")
        while choice not in ("y", "n"):
            choice = input("Do you really want to delete the whole database? (y/n): ")
        if choice.lower().strip() == "y":
            logging.info(f"Deleting ChromaDB")
            client.delete_collection(name=collection_name)
            logging.info(f"ChromaDb deleted successfully")
            logging.info(f"Exiting")
            exit()
        elif choice.lower().strip() == "n":
            logging.info(f"Exiting")
            exit(0)