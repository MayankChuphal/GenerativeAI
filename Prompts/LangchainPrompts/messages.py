from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage

from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4')

messages = [
    SystemMessage(content='You are a doctor talking to your patient'),
    HumanMessage(content='Hii how are you doing sir')
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)



