from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4')


chat_prompt = ChatPromptTemplate(
    
    [
    
    ('system','you are a {domain} expert'),
    
    ('human','can you explain what is {topic}')
    
    ]
)
prompt = chat_prompt.invoke({
    'domain':'cricket',
    'topic':'flipper'
})

result = model.invoke(prompt);

print(prompt)

print(result.content)



