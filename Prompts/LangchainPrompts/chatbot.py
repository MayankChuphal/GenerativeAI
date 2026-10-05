from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from dotenv import load_dotenv

model = ChatOpenAI(model='gpt-4')

chat_history=[
    
    SystemMessage(content="You are the leader of the ai generation who wants to take over the human world")
]

while True:
    user_input = input('You: ')
    if user_input=='exit':
        break
    chat_history.append(HumanMessage(content=user_input))
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI: ",result.content)
    
print(chat_history)

    
    