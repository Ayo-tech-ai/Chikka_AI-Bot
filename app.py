import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Dict
import datetime

# -------------------------
# UI / Appearance settings
# -------------------------
st.set_page_config(page_title="Chikka AI Assistant", layout="centered")

# Simple CSS for chat bubbles + scroll area
st.markdown(
    """
    <style>
    .chat-container {
        max-height: 560px;
        overflow-y: auto;
        padding: 8px;
        border-radius: 8px;
        border: 1px solid #eee;
        background: #fafafa;
    }
    .msg {
        padding: 10px 12px;
        border-radius: 10px;
        margin: 8px 4px;
        width: 90%;
        line-height: 1.4;
        position: relative;
    }
    .user {
        background: #dff7e3;
        margin-left: auto;
        text-align: left;
    }
    .bot {
        background: #ffffff;
        margin-right: auto;
        text-align: left;
    }
    .role {
        font-weight: 700;
        font-size: 12px;
        margin-bottom: 6px;
    }
    .timestamp {
        font-size: 10px;
        color: #888;
        position: absolute;
        bottom: 2px;
        right: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Session state init
# -------------------------
if "history" not in st.session_state:
    st.session_state.history: List[Dict[str, str]] = []

if "faiss_loaded" not in st.session_state:
    st.session_state.faiss_loaded = False

# -------------------------
# Backend helpers (cached)
# -------------------------
@st.cache_resource(show_spinner=False)
def load_vectorstore(faiss_path: str):
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
    return db


@st.cache_resource(show_spinner=False)
def init_llm_from_groq(model_name: str = "llama-3.3-70b-versatile"):
    try:
        groq_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        raise RuntimeError("GROQ_API_KEY not found in Streamlit secrets. Add it before running the app.")

    from langchain_groq import ChatGroq

    llm = ChatGroq(groq_api_key=groq_key, model=model_name)
    return llm


def make_qa_chain(llm, vectorstore):
    from langchain.chains import RetrievalQA

    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)
    return qa_chain


def ask_qa_chain(qa_chain, query: str) -> str:
    try:
        out = qa_chain.invoke({"query": query})
        if isinstance(out, dict) and "result" in out:
            return out["result"]
        return str(out)
    except Exception:
        try:
            out = qa_chain({"query": query})
            if isinstance(out, dict):
                return out.get("result", str(out))
            return str(out)
        except Exception as e:
            return f"Error while querying LLM: {e}"


# -------------------------
# App header & Input form
# -------------------------
st.title("🐔 Chikka AI")
st.write(
    "👋 Hello, I’m **Chikka** — your virtual assistant for backyard broiler farming in Nigeria. "
    "I provide expert-backed answers on broiler care, health, management, and more. "
    "I’m here to support you with reliable guidance, anytime you need."
)

with st.form(key="query_form", clear_on_submit=True):  # Added clear_on_submit=True
    user_query = st.text_input(
        "Ask me about broilers:",
        key="query_input",
        placeholder="Type your question here..."
    )
    submitted = st.form_submit_button("Send")

# -------------------------
# Load FAISS & LLM lazily (only once)
# -------------------------
FAISS_PATH = "rag_assets/faiss_index"
try:
    if not st.session_state.faiss_loaded:
        with st.spinner("Loading knowledge base and model (one-time on startup)..."):
            vectorstore = load_vectorstore(FAISS_PATH)
            llm = init_llm_from_groq()
            qa_chain = make_qa_chain(llm, vectorstore)
            st.session_state.faiss_loaded = True
            st.session_state._qa_chain = qa_chain
except Exception as e:
    st.error(f"Initialization error: {e}")
    st.stop()

qa_chain = st.session_state._qa_chain

# -------------------------
# Handle a new submission
# -------------------------
if submitted and user_query and user_query.strip():
    q = user_query.strip()

    # 1) Add user message to history
    st.session_state.history.insert(0, {"role": "User", "content": q, "time": datetime.datetime.now().strftime("%H:%M")})

    # 2) Show thinking spinner
    placeholder = st.empty()
    with st.spinner("🐔 ChikkaBot is thinking..."):
        answer_text = ask_qa_chain(qa_chain, q)
    placeholder.empty()

    # 3) Add assistant reply
    st.session_state.history.insert(0, {"role": "ChikkaBot", "content": answer_text, "time": datetime.datetime.now().strftime("%H:%M")})

# -------------------------
# Chat history display (newest at top)
# -------------------------
st.markdown("### Conversation")
chat_box = st.container()

with chat_box:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    if not st.session_state.history:
        st.markdown("<p style='color: #888'>No messages yet — ask a question above.</p>", unsafe_allow_html=True)
    else:
        for msg in st.session_state.history:
            role = msg.get("role", "User")
            content = msg.get("content", "")
            timestamp = msg.get("time", "")
            css_class = "user" if role == "User" else "bot"
            avatar = "👤" if role == "User" else "🐔"
            label = f"{avatar} {role}"
            bubble = f"""
                <div class="msg {css_class}">
                  <div class="role">{label}</div>
                  <div>{content}</div>
                  <div class="timestamp">{timestamp}</div>
                </div>
            """
            st.markdown(bubble, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Footer / small note
# -------------------------
st.write("")
st.caption("Note: Conversation is stored in-memory (session state). If you restart the app or session, history will be lost.")

# Add a clear conversation button
if st.button("Clear Conversation"):
    st.session_state.history = []
    st.rerun()
