from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate # You can also import the PromptTemplate

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

# Now you can ask the question about the web to the model
question = "What is this book about?"

# You can also create a prompt template
template = """
Please answer the question based on the following context.
If you don't know the answer, just say you don't know, don't try to make up an answer.
Answer in at most three sentences. Please answer as concisely as possible. Finally, always say "Thank you for asking!"
Context: {context}
Question: {question}
Helpful answer:
"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

qa_chain = RetrievalQA.from_chain_type(
    chat,
    retriever=vectordb.as_retriever(),
    return_source_documents=True, # Return the source documents(optional)
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT} # Add the prompt template to the chain
)
result = qa_chain({"query": question})
print(result["result"])
print(result["source_documents"][0]) # If you set return_source_documents to True, you can get the source documents