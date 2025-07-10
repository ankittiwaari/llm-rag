
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from db import vector_store
from os import getenv
from dotenv import load_dotenv
load_dotenv()
loader = PyPDFLoader(getenv("SAMPLE_PDF"))
pages = []

for page in loader.lazy_load():
    pages.append(page)

print(f"{pages[0].metadata}\n")

docs = loader.load()
print("Splitting")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Index chunks
print("Adding")
_ = vector_store.add_documents(documents=all_splits)
print("Added!!!")
