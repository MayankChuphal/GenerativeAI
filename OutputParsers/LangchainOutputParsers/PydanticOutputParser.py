from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from pydantic import BaseModel,Field
from typing import Optional

model = ChatOpenAI(model='gpt-4')

class Person(BaseModel):
    name:str = Field(description='name of the person')
    age:int = Field(gt=18,description='age of the person')
    city:str = Field(description='name of the city person belongs to ')
    year:Optional[int] = None


parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Generate name,age,city,year of a famous athlete from {place}.\n {format_instruction}",
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

prompt = template.invoke({
    'place':'Australia'
})
store = model.invoke(prompt)
result_prev = parser.parse(store.content)

print(result_prev)


chain = template | model | parser

result = chain.invoke({
    'place':'India'
})

print(result)


