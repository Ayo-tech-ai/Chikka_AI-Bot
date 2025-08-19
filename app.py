import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Load FAISS index
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("rag_assets/faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

# LLM setup
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]  # from Streamlit Cloud secrets
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile")

# Retrieval QA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Streamlit UI
st.title("🐔 Chikka AI Assistant")
query = st.text_input("Ask me about broilers:")

if query:
    response = qa_chain.invoke(query)
    st.write("**Answer:**", response["result"])
