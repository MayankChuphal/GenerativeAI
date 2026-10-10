# from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint,HuggingFacePipeline
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser,ResponseSchema

# llm = HuggingFaceEndpoint(
#     repo_id='meta-llama/Llama-3.1-8B-Instruct:novita',
#     task='text-generation'
# )
# llm = HuggingFacePipeline.from_model_id(
    
#     model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
#     task="text-generation",
#     pipeline_kwargs=dict(
#          temperature = 0.5,
#          max_new_tokens = 100
# )
    
# llm = HuggingFacePipeline.from_model_id(
#     model_id='Qwen/Qwen2.5-0.5B-Instruct',
#     task='text-generation'
# )

# model = ChatHuggingFace(llm=llm)


model = ChatOpenAI(model='gpt-4')

schema=[
    ResponseSchema(name='fact 1',description='Fact1 about the topic'),
    ResponseSchema(name='fact 2',description='Fact2 about the topic'),
    ResponseSchema(name='fact 3',description='Fact2 about the topic')
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give me the 3 facts about {topic},\n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

prompt = template.invoke({
    'topic':"BlackHole"
})

result = model.invoke(prompt)

final_result = parser.parse(result.content);

print(final_result)



