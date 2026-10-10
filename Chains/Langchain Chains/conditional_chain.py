from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser

from pydantic import BaseModel,Field
from typing import Literal
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel,RunnableBranch,RunnableLambda



from dotenv import load_dotenv
load_dotenv()


model = ChatOpenAI()

parser = StrOutputParser()

class FeedBack(BaseModel):
    sentiment:Literal['positive','negative'] = Field(description="Give me the sentiment of the feedback")


parser2 = PydanticOutputParser(pydantic_object=FeedBack)


prompt1 = PromptTemplate(
    template="Classify the sentiment of the given feedback into positive or negative.\n {feedback} \n {format_instructions}",
    input_variables=['feedback'],
    partial_variables={'format_instructions':parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

# print(classifier_chain.invoke({'feedback':'This watch is terrible'}))

prompt2 = PromptTemplate(
    template="Give me the response for this positive feedback.\n {feedback}",
    input_variables=['feedback']
)
prompt3 = PromptTemplate(
    template="Give me the response for this negative feedback.\n {feedback}",
    input_variables=['feedback']
)


branch_chain = RunnableBranch(
    (lambda x:x.sentiment=='positive',prompt2 | model | parser),
    (lambda x:x.sentiment=='negative',prompt3 | model | parser),
    (RunnableLambda(lambda x:"We can not process this request"))
    
)

final_chain = classifier_chain | branch_chain

result = final_chain.invoke({
    'feedback':"This is the worst product ever"
})

print(result)

final_chain.get_graph().print_ascii()