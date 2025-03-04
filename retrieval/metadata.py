from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import ZhipuAIEmbeddings

# load the web page
loader = WebBaseLoader("https://en.d2l.ai/")
docs = loader.load()

# split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
)
splits = text_splitter.split_documents(docs)
# print(len(splits))

# set the embeddings models
embeddings = ZhipuAIEmbeddings(
    model="embedding-3",
    api_key="your_api_key"
)

# set the persist directory
persist_directory = r'.'

# create the vector database
vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory=persist_directory
)
# print(vectordb._collection.count())

# add new documents from another website
new_loader = WebBaseLoader("https://www.deeplearning.ai/")
new_docs = new_loader.load()

# split the text into chunks
new_splits = text_splitter.split_documents(new_docs)

# add to the existing vector database
vectordb.add_documents(new_splits)

# Get all documents
all_docs = vectordb.similarity_search("What is the difference between a neural network and a deep learning model?", k=20)

# Print the metadata of the documents
for i, doc in enumerate(all_docs):
    print(f"Document {i+1} metadata: {doc.metadata}")
# Document 1 metadata: {'language': 'en', 'source': 'https://en.d2l.ai/', 'title': 'Dive into Deep Learning — Dive into Deep Learning 1.0.3 documentation'}
# Document 2 metadata: {'language': 'en', 'source': 'https://en.d2l.ai/', 'title': 'Dive into Deep Learning — Dive into Deep Learning 1.0.3 documentation'}
# Document 3 metadata: {'language': 'en', 'source': 'https://en.d2l.ai/', 'title': 'Dive into Deep Learning — Dive into Deep Learning 1.0.3 documentation'}
# Document 4 metadata: {'description': 'DeepLearning.AI | Andrew Ng | Join over 7 million people learning how to use and build AI through our online courses. Earn certifications, level up your skills, and stay ahead of the industry.', 'language': 'en', 'source': 'https://www.deeplearning.ai/', 'title': 'DeepLearning.AI: Start or Advance Your Career in AI'}

question = "how the neural network works?"
# filter the documents from the specific website
docs_meta = vectordb.similarity_search(question, k=1, filter={"source": "https://www.deeplearning.ai/"})
print(docs_meta[0].page_content[:100])