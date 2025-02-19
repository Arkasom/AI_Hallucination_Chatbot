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



        
# try:
#     import streamlit as st
# except ModuleNotFoundError:
#     raise ModuleNotFoundError("Streamlit module not found. Please install it using 'pip install streamlit'.")
# import os
# import pandas as pd
# from langchain.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
# from langchain_groq import ChatGroq
# import tempfile

# if not os.environ.get("GROQ_API_KEY"):
#     st.error("GROQ API Key is missing!")

# def load_pdfs(pdf_files):
#     docs = []
#     for pdf in pdf_files:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
#             temp_pdf.write(pdf.read())
#             temp_pdf_path = temp_pdf.name
#         loader = PyPDFLoader(temp_pdf_path)
#         docs.extend(loader.load())
#     return docs

# def split_docs(docs, chunk_size=500, chunk_overlap=50):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#     return text_splitter.split_documents(docs)

# def create_vector_db(documents):
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     return FAISS.from_documents(documents, embeddings)

# def load_dataset(csv_file):
#     return pd.read_csv(csv_file)

# def query_data(user_query, vector_db, dataset):
#     llm = ChatGroq(groq_api_key=os.environ.get("GROQ_API_KEY"), model_name="mixtral-8x7b-32768")
#     retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
#     qa_chain = RetrievalQA(llm=llm, retriever=retriever)

#     dataset_keywords = ["table", "csv", "column", "row", "trend", "statistics", "value", "alert"]
#     if any(keyword in user_query.lower() for keyword in dataset_keywords):
#         return dataset_query(user_query, dataset)
#     return qa_chain.run(user_query)

# def dataset_query(user_query, dataset):
#     try:
#         if "columns" in user_query.lower():
#             return f"Columns in the dataset: {list(dataset.columns)}"
#         if "rows" in user_query.lower():
#             return f"The dataset has {len(dataset)} rows."
#         if "show" in user_query.lower():
#             return dataset.head().to_string()
#         return "Invalid query format. Try asking about columns, rows, or showing data."
#     except Exception as e:
#         return f"Couldn't process dataset query: {str(e)}"

# st.set_page_config(page_title="AI Health Chatbot", layout="wide")
# st.title("🩺 AI Health Assistant for Patient Monitoring")

# uploaded_pdfs = st.file_uploader("Upload Disease Medication PDFs", accept_multiple_files=True, type=["pdf"])
# uploaded_dataset = st.file_uploader("Upload Patient Monitoring Data (CSV)", type=["csv"])

# if uploaded_pdfs and uploaded_dataset:
#     st.write("Processing files...")
#     pdf_docs = load_pdfs(uploaded_pdfs)
#     split_documents = split_docs(pdf_docs)
#     vector_db = create_vector_db(split_documents)
#     dataset = load_dataset(uploaded_dataset)

#     user_query = st.text_input("Ask a question about patient health:")

#     if user_query:
#         response = query_data(user_query, vector_db, dataset)
#         st.write("**Chatbot Response:**", response)

#     st.write("### 📊 Patient Monitoring Data Overview")
#     st.dataframe(dataset.head())
