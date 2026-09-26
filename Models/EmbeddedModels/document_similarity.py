from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large",dimensions=32)

documents=[
    
"Virat Kohli is known for his aggressive batting and exceptional chase performances.",
"Rohit Sharma is famous for his elegant stroke play and record-breaking ODI double centuries.",
"MS Dhoni is renowned for his calm leadership, explosive finishing, and wicketkeeping skills.",
"Jasprit Bumrah is known for his unique bowling action, deadly yorkers, and accuracy.",
"AB de Villiers was celebrated for his innovative batting and ability to play shots all around the ground."
]


query = "tell me about rohit sharma"

doc_embeddings = embeddings.embed_documents(documents)
query_embeddings = embeddings.embed_query(query)


# print(cosine_similarity([query_embeddings],doc_embeddings)) 
#both params need to be in 2d vector

scores = cosine_similarity([query_embeddings],doc_embeddings)[0] 
#we only need the single list that's why taking 0th row only

print(sorted(list(enumerate(scores)),key = lambda x:x[1])[-1])

#after enumerate list will be a list of tuples of size 2 [(0,val1),(1,val2),(2,val3)]


#sorting the list on basis of second element and using -1 to get the last element with highest value (cosine similarity)

index,score = sorted(list(enumerate(scores)),key = lambda x:x[1])[-1]

print(query)

print(documents[index])

print("similarity score ",score)










