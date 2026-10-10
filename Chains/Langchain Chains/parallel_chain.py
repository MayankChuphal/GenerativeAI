from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel

from dotenv import load_dotenv
load_dotenv()


model1 = ChatOpenAI(model='gpt-4')

model2 = ChatGoogleGenerativeAI(model='gemini-3.8-flash')

prompt1 = PromptTemplate(
    template="Generate a short anc clear notes from the text.\n {text}",
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template = "Generate 5 questions and answers quiz from the text.\n {text}",
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template = "Merge the provided notes and quiz into a single document \n {notes} \n {quiz}",
    input_variables=['notes','quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes':prompt1 | model1 | parser,
    'quiz':prompt2 | model2 | parser
})

merge_chain = prompt3 | model2 | parser

final_chain = parallel_chain | merge_chain

text="""Cricket is an incredibly nuanced, strategically deep, and globally celebrated sport 
that functions as both a
prestigious athletic pursuit and a profound cultural phenomenon for billions of people across
the world. At its fundamental core, the game is played on a vast, oval-shaped grass field 
centered around a meticulously prepared 22-yard rectangular strip known as the pitch, which
features three wooden stumps called wickets at either end. The match unfolds as an intense,
high-stakes battle of wits and physical mastery between a bowler, who hurls a hard, 
cork-and-leather ball at speeds exceeding 150 kilometers per hour or with deceptive, 
aerodynamic spin, and a batsman, who has mere milliseconds to read the ball's trajectory, 
bounce, and deviation to execute a precise counter-strike using a willow bat. The batting 
team aims to score runs either by physically sprinting between the wickets after a hit or by 
sending the ball racing across or over the boundary rope for an automatic four or six runs. 
Conversely, the fielding team acts as a cohesive defensive unit, precisely positioned across 
the vast outfield based on data-driven strategies to choke the flow of runs and orchestrate
dismissals (wickets) via athletic catches, direct-hit run-outs, leg-before-wicket (LBW) 
decisions, or by clean-bowling the batsman.
What truly elevates cricket above many other global sports is
its unique multi-format structure, which accommodates vastly different paces of play and
tactical philosophies within the exact same sport. At the pinnacle of tradition sits Test 
Cricket, a grueling five-day epic requiring monumental physical stamina, mental resilience,
and long-term tactical planning, where teams wear traditional whites, conditions shift 
dynamically with the weather, and a single match
can last thirty hours without a guaranteed winner. Serving as a bridge is One Day
International (ODI) cricket, a 50-over format that balances the patient accumulation of
runs with aggressive late-game acceleration. Meanwhile, the modern era has been completely 
revolutionized by Twenty20 (T20) cricket and newer short-form innovations, which condense 
the game into a three-hour, high-octane spectacle of non-stop boundaries, unorthodox batting 
innovations like the switch-hit, and theatrical stadium entertainment. This format has birthed 
massive global franchises like the Indian Premier League (IPL), turning cricket into a 
multi-billion-dollar commercial powerhouse. Ultimately, from historic rivalries like the
Ashes to the high-stakes passion of India-Pakistan clashes, cricket transcends simple recreation;
it is a unifying religion, a treasure trove of intricate statistics, and a timeless
testament to sportsmanship that shapes the identity of entire nations."""

result = final_chain.invoke({
    'text':text
})

print(result)

final_chain.get_graph().print_ascii()



