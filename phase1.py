import streamlit as st

st.title("Arkasom Chatbot!")

#setup a session state variable to hold all past messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

#display all the past messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

prompt = st.chat_input("Enter your prompt here...")

if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({"role":'user',"content":prompt})
    response = "I am your chat assistant"
    st.chat_message('assistant').markdown(response)
    st.session_state.messages.append({"role":'assistant',"content":response})