from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

loader = WebBaseLoader("https://en.d2l.ai/")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
)
splits = text_splitter.split_documents(docs)

persist_directory = '.'

# initialize the embeddings
embeddings = ZhipuAIEmbeddings(
    model="embedding-3",
    api_key="your_api_key"
)

# initialize the vector database
vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory=persist_directory
)

chat = ChatZhipuAI(
    model="glm-4-flash",
    temperature=0.5,
    api_key =  "your_api_key"
)

# initialize the memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# create the ConversationalRetrievalChain
retriever = vectordb.as_retriever()
qa = ConversationalRetrievalChain.from_llm(
    chat,
    retriever=retriever,
    memory=memory
)

# First question
question = "What is the main topic of this book?"
result = qa.invoke({"question": question})
print(result['answer'])

# Second question
question = "Can you tell me more about it?"
result = qa.invoke({"question": question})
print(result['answer'])