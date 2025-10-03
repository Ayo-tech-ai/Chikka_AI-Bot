import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Dict
import datetime
import re

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
        padding: 12px 16px;
        border-radius: 12px;
        margin: 10px 6px;
        width: 90%;
        line-height: 1.5;
        position: relative;
    }
    .user {
        background: #e8f5e9;
        margin-left: auto;
        text-align: left;
        border: 1px solid #c8e6c9;
    }
    .bot {
        background: #ffffff;
        margin-right: auto;
        text-align: left;
        border: 1px solid #e0e0e0;
    }
    .role {
        font-weight: 600;
        font-size: 13px;
        margin-bottom: 6px;
        color: #555;
    }
    .timestamp {
        font-size: 11px;
        color: #888;
        position: absolute;
        bottom: 4px;
        right: 10px;
    }
    .suggestion-chip {
        display: inline-block;
        background-color: #e3f2fd;
        border: 1px solid #bbdefb;
        border-radius: 18px;
        padding: 6px 14px;
        margin: 5px 5px 5px 0;
        font-size: 14px;
        cursor: pointer;
        transition: all 0.2s;
        color: #1976d2;
    }
    .suggestion-chip:hover {
        background-color: #bbdefb;
        transform: translateY(-1px);
    }
    .follow-up {
        margin-top: 14px;
        padding-top: 10px;
        border-top: 1px dashed #e0e0e0;
        font-style: italic;
        color: #666;
        font-size: 14px;
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

if "conversation_context" not in st.session_state:
    st.session_state.conversation_context = ""

# ✅ Ensure agent + qa_chain session state exist
if "_agent" not in st.session_state:
    st.session_state._agent = None

if "_qa_chain" not in st.session_state:
    st.session_state._qa_chain = None

# -------------------------
# Backend helpers (cached)
# -------------------------
@st.cache_resource(show_spinner=False)
def load_vectorstore(faiss_path: str):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
    return db


@st.cache_resource(show_spinner=False)
def init_llm_from_groq(model_name: str = "llama-3.3-70b-versatile"):
    try:
        groq_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        raise RuntimeError("GROQ_API_KEY not found in Streamlit secrets. Add it before running the app.")

    llm = ChatGroq(groq_api_key=groq_key, model=model_name)
    return llm


def make_qa_chain(llm, vectorstore):
    from langchain.prompts import PromptTemplate

    retriever = vectorstore.as_retriever()
    prompt_template = """You are Chikka, a friendly expert AI assistant specialized in backyard broiler farming. 
Provide helpful, conversational answers that are clear and focused. Be naturally conversational but avoid unnecessary fluff.

After providing your main answer, end with a natural follow-up question that continues the conversation. 
Make the follow-up relevant to the topic and helpful for the farmer.

Context: {context}

Question: {question}

Answer in a friendly, expert tone. Share knowledge conversationally while staying on topic. 
End with a helpful follow-up question to continue the conversation."""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": PROMPT},
    )
    return qa_chain

# -------------------------
# Weather + Feed Tool Setup
# -------------------------
from tools.weather import get_weather
from tools/calculator.py import calculate_feed_cost


def extract_city(query: str) -> str:
    match = re.search(r"in\s+([A-Za-z\s]+)", query, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return "Lagos"


def handle_query(query: str, qa_chain, context: str = ""):
    q_lower = query.lower()
    if "weather" in q_lower or "rain" in q_lower or "temperature" in q_lower:
        city = extract_city(query)
        return get_weather(city, "NG")
    elif "feed cost" in q_lower or "total feed" in q_lower:
        return calculate_feed_cost(query)
    return ask_qa_chain(qa_chain, query, context)


# -------------------------
# Load FAISS & Initialize Agent
# -------------------------
FAISS_PATH = "rag_assets/faiss_index"
try:
    if not st.session_state.faiss_loaded:
        with st.spinner("Getting things ready for you..."):
            vectorstore = load_vectorstore(FAISS_PATH)
            llm = init_llm_from_groq()
            qa_chain = make_qa_chain(llm, vectorstore)

            from langchain.agents import initialize_agent, Tool

            tools = [
                Tool.from_function(
                    func=lambda q: ask_qa_chain(qa_chain, q),
                    name="BroilerQA",
                    description="Answer broiler farming related questions"
                ),
                Tool.from_function(
                    func=lambda q: handle_query(q, qa_chain),
                    name="WeatherOrFeed",
                    description="Handles weather and feed cost queries"
                )
            ]

            st.session_state._qa_chain = qa_chain
            st.session_state._agent = initialize_agent(
                tools=tools, llm=llm,
                agent="zero-shot-react-description", verbose=True
            )
            st.session_state.faiss_loaded = True
except Exception as e:
    st.error(f"Sorry, I'm having trouble loading my knowledge base: {e}")
    st.stop()

qa_chain = st.session_state._qa_chain
agent = st.session_state._agent

# -------------------------
# App header & Input form
# -------------------------
st.title("🐔 Chikka AI")
st.write(
    "👋 Hello! I'm **Chikka**, your friendly assistant for backyard broiler farming. "
    "I'm here to help with practical advice on broiler care, health, and management."
)

with st.form(key="query_form", clear_on_submit=True):
    user_query = st.text_input("Ask me about broilers:", key="query_input")
    submitted = st.form_submit_button("Send")

# -------------------------
# Handle a new submission
# -------------------------
if submitted and user_query and user_query.strip():
    q = user_query.strip()

    st.session_state.history.insert(
        0, {"role": "User", "content": q, "time": datetime.datetime.now().strftime("%H:%M")}
    )

    placeholder = st.empty()
    with st.spinner("Thinking about your question..."):
        answer_text = agent.run(q) if agent else ask_qa_chain(qa_chain, q)
    placeholder.empty()

    st.session_state.history.insert(
        0, {"role": "ChikkaBot", "content": answer_text, "time": datetime.datetime.now().strftime("%H:%M")}
    )

# -------------------------
# Chat history display
# -------------------------
st.markdown("### Conversation")
chat_box = st.container()

with chat_box:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    if not st.session_state.history:
        st.markdown("<p style='color: #888; text-align: center;'>No messages yet. Ask me anything about broiler farming!</p>", unsafe_allow_html=True)
    else:
        for msg in st.session_state.history:
            role = msg.get("role", "User")
            content = msg.get("content", "")
            timestamp = msg.get("time", "")
            css_class = "user" if role == "User" else "bot"
            avatar = "👤" if role == "User" else "🐔"
            label = f"{avatar} {role}"

            st.markdown(f"""
                <div class="msg {css_class}">
                    <div class="role">{label}</div>
                    <div>{content}</div>
                    <div class="timestamp">{timestamp}</div>
                </div>
            """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Footer
# -------------------------
st.caption("💡 Conversation history clears when you refresh.")
if st.button("🧹 Clear Conversation"):
    st.session_state.history = []
    st.session_state.conversation_context = ""
    st.session_state._agent = None
    st.session_state._qa_chain = None
    st.rerun()
