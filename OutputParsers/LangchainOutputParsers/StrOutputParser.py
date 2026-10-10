from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = HuggingFaceEndpoint(
    repo_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0:featherless-ai',
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

template1=PromptTemplate(
    template="Write a report on the {topic}",
    input_variables = ['topic']
)

template2 = PromptTemplate(
    template="Write a 5 line summary on the following text.\n {text}",
    input_variables=['text']
)

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({
    'topic':'BlackHole'
})

print(result)



