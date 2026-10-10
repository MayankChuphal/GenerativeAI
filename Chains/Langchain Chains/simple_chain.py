from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


model = ChatOpenAI(model='gpt-4')

prompt = PromptTemplate(
    template="Give me 5 facts about the {topic}",
    input_variables=['topic']
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({
    'topic':'Cricket'
})

print(result)

chain.get_graph().print_ascii()