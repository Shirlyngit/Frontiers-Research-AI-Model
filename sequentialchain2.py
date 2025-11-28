import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openai import OpenAI
import streamlit as st
from dotenv import load_dotenv 


load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

llm = OpenAI(temperature = 0.9)

st.title('Medium Article Generator')
topic = st.text_input('Input your topic of interest')
language = st.text_input('Input language')


title_template = PromptTemplate.from_template(
        'Give me a catchy title on {topic} in {language}'
)

article_template = PromptTemplate.from_template(
        'Expand this title into a medium-like article {title}'
)

chain1 = title_template | llm | StrOutputParser()
chain2 = article_template | llm | StrOutputParser()

# Replacement for RunnablePassThrough()
passthrough = RunnableLambda(lambda x: x)

overall_chain = (
        {
                'title': chain1,
                'topic': passthrough,
                'language': passthrough
        }
        | chain2
)

if topic:
        response = overall_chain.invoke({'topic': topic, 'language': 'english'})
        st.subheader("Generated Title: ")
        st.write(response)
