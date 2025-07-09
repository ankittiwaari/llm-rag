
import bs4
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from db import vector_store

# Load and chunk contents of the blog
# loader = WebBaseLoader(
#     web_paths=("https://www.one.com/en/hosting",),
#     bs_kwargs=dict(
#         parse_only=bs4.SoupStrainer(
#             class_=("tabbed-plans-content-wrapper", "acb-itd-wrapper")
#         )
#     ),
# )

loader = PyPDFLoader('/Users/ankitt/Downloads/Intro+to+AI+-+Course+notes.pdf')
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
