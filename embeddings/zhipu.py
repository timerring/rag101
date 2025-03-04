from langchain_community.embeddings import ZhipuAIEmbeddings

embed = ZhipuAIEmbeddings(
    model="embedding-3",
    api_key="your own api key"
)

input_texts = ["This is a test query1.", "This is a test query2."]
print(embed.embed_documents(input_texts))