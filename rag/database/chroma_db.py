import chromadb

from models import embed_fn


collection_name: str = f"test-collection"
client = chromadb.Client()
collection = client.get_or_create_collection(name=collection_name, embedding_function=embed_fn, metadata={"hnsw:space":"cosine"})
#create db

def create_or_connect_chromadb(database_name):
    if client.database_exists(database_name):
        print(f"Connecting to existing database: {database_name}")
        db = client.get_database(database_name)
    else:
        print(f"Creating new database: {database_name}")
        db = client.create_database(database_name)
    return db