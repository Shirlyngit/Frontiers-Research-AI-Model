#LCEL style:
import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

st.title('Medium Article Generator')
topic = st.text_input('Input your topic of interest:') 
language = st.text_input('Input language')

title_template = PromptTemplate.from_template(
        'Give me a unique medium-like title on {topic} in {language}'
)

llm = OpenAI(temperature= 0.9)

title_chain = title_template | llm 

title_chain = title_chain.with_config({'verbose': True})

if topic:
    response = title_chain.invoke({'topic':topic, 'language':language})
    st.write(response)  
