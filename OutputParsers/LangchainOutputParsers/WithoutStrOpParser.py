from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
load_dotenv()

llm = HuggingFaceEndpoint(
    
    repo_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0:featherless-ai',
    task ='text-generation'
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template="Write a report on the {topic}",
    input_variables=['topic']
)

template2 = PromptTemplate(
    template="Write a 2 line summary on the given text.\n {text}",
    input_variables=['text']
)

prompt1 = template1.invoke({
    'topic':'BlackHole'
})

result = model.invoke(prompt1)

prompt2 = template2.invoke({
    'text':result.content
})

final_result = model.invoke(prompt2)

print(final_result.content)
