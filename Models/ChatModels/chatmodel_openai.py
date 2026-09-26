from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4",temperature=1.5,max_completion_tokens=10)

result = model.invoke("Write me a shayari on love")

# temperature is creativity parameter value can be set bewteen 0-2 
# based on how creative i want the response to be 


print(result.content)


