from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()


embedding = OpenAIEmbeddings(model="text-embedding-3-large",dimensions=32)

documents = [
    "Hii this is mayank",
    "Weather looks great today",
    "This is our day"
    
]
result  = embedding.embed_documents(documents)



print(str(result))




