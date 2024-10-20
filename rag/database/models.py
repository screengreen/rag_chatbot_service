from sentence_transformers import SentenceTransformer
from chromadb import EmbeddingFunction, Embeddings, Documents
from chromadb.utils import embedding_functions



model_name = 'paraphrase-MiniLM-L6-v2'


class EmbedderModelLoader:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(EmbedderModelLoader, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name):
        if not self._initialized:
            self.model_name = model_name
            self.model = None
            self._initialized = True

    def load_model(self):
        if self.model is None:
            self.model = SentenceTransformer(model_name)
        return self.model
    
    def get_model(self):
        return self.model
    


# Define embedding function class
class MyEmbeddingFunction(EmbeddingFunction):
    def __init__(self, embedding_model: SentenceTransformer):
        self.embedding_model = embedding_model
        
    def __call__(self, input: Documents) -> Embeddings:
        batch_embeddings = self.embedding_model.encode(input)
        return batch_embeddings.tolist()
    



EmbModelLoader = EmbedderModelLoader(model_name)
emb_model = EmbModelLoader.get_model()


embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")