from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser,ResponseSchema

from langchain_openai import ChatOpenAI

model = ChatOpenAI(model='gpt-4')

    
# llm = HuggingFacePipeline.from_model_id(
#     model_id='Qwen/Qwen2.5-0.5B-Instruct',
#     task='text-generation'
# )

# ## Qwen/Qwen2.5-0.5B-Instruct

# model = ChatHuggingFace(llm=llm)

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
chain = template | model | parser


result = chain.invoke({'topic':'Anime'})

print(result)



