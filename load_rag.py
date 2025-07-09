
import bs4
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from db import vector_store

loader = PyPDFLoader('/path/to/pdf/file')
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
