from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents=[
    
    "hi my name is Niks",
    "Delhi is so polluted",
    "Kokata is the captial of West Bengal"
]

vector = embeddings.embed_documents(documents)

print(str(vector))