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

# query the vector database with MMR
question = "How the neural network works?"
# fetch the 8 most similar documents, and then choose the 2 most relevant documents
docs_mmr = vectordb.max_marginal_relevance_search(question, fetch_k=8, k=2)
print(docs_mmr[0].page_content[:100])
print(docs_mmr[1].page_content[:100])