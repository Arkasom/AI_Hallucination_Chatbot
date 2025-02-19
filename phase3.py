import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

st.title("ArkasomGPT!")

#creating a session state variable to store the past messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

#displaying the past messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

prompt = st.chat_input("Enter your prompt here...")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({'role':'user','content':prompt})

    #system prompt
    groq_sys_prompt = ChatPromptTemplate.from_template("""You are very smart at everything.
                                                       You always gives best and accurate answers for everthing.
                                                       Answer the following question: {user_prompt}. Start the answers precisely
                                                       and directly no small talk""")
    #APIkey and model selection
    groq_Chat = ChatGroq(groq_api_key = os.environ.get("GROQ_API_KEY"),
                         model_name = "llama3-8b-8192")
    chain = groq_sys_prompt | groq_Chat | StrOutputParser()



    response = chain.invoke({'user_prompt':prompt})
    st.chat_message('assistant').markdown(response)
    st.session_state.messages.append({'role':'assistant','content':response})