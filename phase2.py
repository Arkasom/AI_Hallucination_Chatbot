import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

st.title("AI Hallucination Chatbot!")

# Check for API Key
if not os.environ.get("GROQ_API_KEY"):
    st.error("GROQ API Key is missing!")

# Creating a session state variable to store past messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display past messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

@st.cache_resource
def get_vectorstore():
    pdf_path = "D:\AI_CHATBOT\cureus-0015-00000035179.pdf"
    
    if not os.path.exists(pdf_path):
        st.error(f"Error: PDF file not found at {pdf_path}")
        return None
    
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        if not documents:
            st.error("Error: No text extracted from the PDF.")
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        
        if not texts:
            st.error("Error: Document splitting failed.")
            return None

        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")

        # Create and cache FAISS index
        vectorstore = FAISS.from_documents(texts, embedding)
        return vectorstore

    except Exception as e:
        st.error(f"Error loading document: {e}")
        return None

prompt = st.chat_input("Enter your prompt here...")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    # System prompt
    groq_sys_prompt = ChatPromptTemplate.from_template("""
        You are very smart at everything.
        You always give the best and most accurate answers for everything.
        Answer the following question: {user_prompt}.
        Answers should be precise and concise, no small talk.
    """)

    # API key and model selection
    groq_Chat = ChatGroq(
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        model_name="llama3-8b-8192"
    )

    try:
        vectorstore = get_vectorstore()
        
        if vectorstore is None:
            st.error("Failed to load the document")
            st.stop()  # Stop execution if vectorstore is not available

        chain = RetrievalQA.from_chain_type(
            llm=groq_Chat,
            chain_type='stuff',
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True
        )

        result = chain.invoke({"query": prompt})
        
        # Debugging: Print result dictionary
        #st.write(result)  

        response = result.get("answer", result.get("result", "No response!!"))

        st.chat_message('assistant').markdown(response)
        st.session_state.messages.append({'role': 'assistant', 'content': response})

    except Exception as e:
        st.error(f"Error: [{str(e)}]")


