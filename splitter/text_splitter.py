from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

example_text = """When writing documents, writers will use document structure to group content. \
This can convey to the reader, which idea's are related. For example, closely related ideas \
are in sentances. Similar ideas are in paragraphs. Paragraphs form a document. \n\n  \
Paragraphs are often delimited with a carriage return or two carriage returns. \
Carriage returns are the "backslash n" you see embedded in this string. \
Sentences have a period at the end, but also, have a space.\
and words are separated by space."""

c_splitter = CharacterTextSplitter(
    chunk_size=450, # the size of the chunk
    chunk_overlap=0, # the overlap of the chunk, which can be shared with the previous chunk
    separator = ' '
)
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=0, 
    separators=["\n\n", "\n", " ", ""] # priority of the separators
)

print(c_splitter.split_text(example_text))
# split at 450 characters
print(r_splitter.split_text(example_text))
# split at first \n\n
