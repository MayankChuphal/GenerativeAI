from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict,Annotated

class Review(TypedDict):
    summary:Annotated[str,"A brief summary of the review"]
    sentiment:Annotated[str,"Rreturn sentiment of the review either negative,positive or neutral"]
    
    

model = ChatOpenAI(model='gpt-4o-mini')
structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""
I’m disappointed with this laptop, especially considering its price. The **battery life is poor**, lasting only a few hours with normal use, and the **keyboard feels cheap and lacks good key feedback**. The **display is another major issue**—the colors look dull and the brightness is not sufficient for outdoor use. It also **heats up noticeably during multitasking**, and the **fan becomes quite loud**. Overall, the build and performance do not feel worth the price.
""")

print(result)
print(result['summary'])
print(result['sentiment'])

