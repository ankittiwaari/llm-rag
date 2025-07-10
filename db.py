from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from os import getenv
from dotenv import load_dotenv
load_dotenv()

embeddings = OllamaEmbeddings(model=getenv("EMBEDDING_MODEL"))

vector_store = Chroma(
    collection_name=getenv("EMBEDDING_COLLECTION"),
    embedding_function=embeddings,
    persist_directory=getenv("LOCAL_VECTOR_DIR"),  # Where to save data locally, remove if not necessary
)   