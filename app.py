import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.agents import initialize_agent
from typing import List, Dict
import datetime
import re

# -------------------------
# UI / Appearance settings
# -------------------------
st.set_page_config(page_title="Chikka AI Assistant", layout="centered")

# (CSS styles remain unchanged)
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


# -------------------------
# TOOLS (Weather + Feed Cost)
# -------------------------
from tools.weather import get_weather
from langchain.tools import tool

@tool
def feed_cost_calculator(bird_count: int, feed_per_bird_kg: float, price_per_kg: float) -> str:
    """
    Calculate total feed cost for broilers.
    Args:
        bird_count (int): Number of birds.
        feed_per_bird_kg (float): Estimated feed consumption per bird in kg.
        price_per_kg (float): Cost of feed per kg.
    """
    try:
        total_feed = bird_count * feed_per_bird_kg
        total_cost = total_feed * price_per_kg
        return f"For {bird_count} birds, with {feed_per_bird_kg} kg feed per bird at ₦{price_per_kg}/kg, total feed cost = ₦{total_cost:,.2f}."
    except Exception as e:
        return f"Error calculating feed cost: {str(e)}"

# -------------------------
# QA + Agent Setup
# -------------------------
def make_agent(llm, vectorstore):
    retriever = vectorstore.as_retriever()
    
    from langchain.prompts import PromptTemplate
    prompt_template = """You are Chikka, a friendly expert AI assistant specialized in backyard broiler farming.
Provide clear, conversational answers. When appropriate, use available tools (weather, feed calculator). 
Always end with a natural follow-up question to keep the conversation flowing.

Context: {context}

Question: {question}

Answer as Chikka, and ask a follow-up at the end.
"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        retriever=retriever, 
        return_source_documents=False,
        chain_type_kwargs={"prompt": PROMPT}
    )

    tools = [get_weather, feed_cost_calculator]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent="chat-conversational-react-description",
        verbose=True,
        handle_parsing_errors=True
    )

    return qa_chain, agent

# -------------------------
# App header & Input form
# -------------------------
st.title("🐔 Chikka AI")
st.write(
    "👋 Hello! I'm **Chikka**, your friendly assistant for backyard broiler farming. "
    "I'm here to help with practical advice on broiler care, health, and management."
)

# Suggestion chips (unchanged)
if "suggestions" in st.session_state and st.session_state.suggestions:
    st.markdown("**You might want to ask:**")
    cols = st.columns(2)
    for i, suggestion in enumerate(st.session_state.suggestions[:4]):
        with cols[i % 2]:
            if st.button(suggestion, key=f"sugg_{i}", use_container_width=True):
                st.session_state.query_input = suggestion
                st.session_state.auto_submit = True

with st.form(key="query_form", clear_on_submit=True):
    user_query = st.text_input(
        "Ask me about broilers:",
        key="query_input",
        placeholder="What would you like to know about broiler farming?"
    )
    submitted = st.form_submit_button("Send")

if "auto_submit" in st.session_state and st.session_state.auto_submit:
    submitted = True
    st.session_state.auto_submit = False

# -------------------------
# Load FAISS & Agent lazily
# -------------------------
FAISS_PATH = "rag_assets/faiss_index"
try:
    if not st.session_state.faiss_loaded:
        with st.spinner("Getting things ready for you..."):
            vectorstore = load_vectorstore(FAISS_PATH)
            llm = init_llm_from_groq()
            qa_chain, agent = make_agent(llm, vectorstore)
            st.session_state.faiss_loaded = True
            st.session_state._qa_chain = qa_chain
            st.session_state._agent = agent
except Exception as e:
    st.error(f"Sorry, I'm having trouble loading my knowledge base: {e}")
    st.stop()

qa_chain = st.session_state._qa_chain
agent = st.session_state._agent

# -------------------------
# Handle user query
# -------------------------
if submitted and user_query and user_query.strip():
    q = user_query.strip()

    # Maintain conversation context
    if st.session_state.history:
        recent_messages = st.session_state.history[:3]
        context_text = " ".join([msg["content"] for msg in recent_messages if msg["role"] == "User"])
        entities = []  # left as-is
        if entities:
            st.session_state.conversation_context = f"We've been discussing: {', '.join(entities)}"

    # Add user message
    st.session_state.history.insert(0, {"role": "User", "content": q, "time": datetime.datetime.now().strftime("%H:%M")})

    # Get response (try tools first)
    placeholder = st.empty()
    with st.spinner("Thinking about your question..."):
        try:
            answer_text = agent.run(q)
        except Exception:
            answer_text = qa_chain.invoke({"query": q})["result"]
    placeholder.empty()

    # Add bot reply
    st.session_state.history.insert(0, {"role": "ChikkaBot", "content": answer_text, "time": datetime.datetime.now().strftime("%H:%M")})

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
st.caption("💡 Conversation history is temporary and will clear when you refresh the page.")

if st.button("🧹 Clear Conversation"):
    st.session_state.history = []
    st.session_state.conversation_context = ""
    if "suggestions" in st.session_state:
        del st.session_state.suggestions
    st.rerun()
