from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct:novita',
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)
parser = JsonOutputParser()

template1 = PromptTemplate(
    template='Give me the name,age,city of a famous athlete.\n {format_instructions}',
    input_variables=[],
    partial_variables={'format_instructions':parser.get_format_instructions()}
)

template2 = PromptTemplate(
    template='Give me 5 facts about {topic}.\n {format_instructions}',
    input_variables=['topic'],
    partial_variables={'format_instructions':parser.get_format_instructions()}
)

chain2 = template2 | model | parser

result2 = chain2.invoke({
    'topic':'BlackHole'
})


# prompt = template1.format()

chain = template1 | model | parser

result = chain.invoke({
})

print(result)

print(result2)

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)



# print(result.content)
# print(prompt)
# print(final_result)