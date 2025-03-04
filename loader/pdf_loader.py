from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("ProbRandProc_Notes2004_JWBerkeley.pdf")
pages = loader.load()

print(type(pages))
# <class 'list'>
print(len(pages))
# Print the total num of pages

# Using the first page as an example
page = pages[0]
print(type(page))
# <class 'langchain_core.documents.base.Document'>

# What is inside the page:
# 1. page_content
# 2. meta_data: the description of the page

print(page.page_content[0:500])
print(page.metadata)