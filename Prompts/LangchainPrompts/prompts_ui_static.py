from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

import streamlit as st   

st.header("Prompts Langchain")

user_input = st.text_input("Enter your prompt")

model = ChatOpenAI(model='gpt-4')

if st.button("Summarize"):
    result = model.invoke(user_input)
    st.write(result.content)
    
    
