from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://zh.d2l.ai/")
pages = loader.load()
print(pages[0].page_content[:500])

# You can also use json as the post processing
# import json
# convert_to_json = json.loads(pages[0].page_content)