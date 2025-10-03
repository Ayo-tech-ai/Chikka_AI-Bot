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
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
    return db

@st.cache_resource(show_spinner=False)
def init_llm_from_groq(model_name: str = "llama-3.3-70b-versatile"):
    try:
        groq_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        raise RuntimeError("GROQ_API_KEY not found in Streamlit secrets. Add it before running the app.")
    return ChatGroq(groq_api_key=groq_key, model=model_name)

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
    return RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, return_source_documents=False,
        chain_type_kwargs={"prompt": PROMPT}
    )

# -------------------------
# Entity extraction + Suggestions + Naturalization
# -------------------------
def extract_key_entities(text):
    patterns = [
        r'\b(Newcastle|Gumboro|Coccidiosis|Marek\'s|IBD|Avian Influenza|AI|CRD|Fowl Cholera|Fowl Pox)\b',
        r'\b(broiler|chick|poultry|farm|feed|vaccine|ventilation|temperature|humidity)\b',
        r'\b(symptom|treatment|prevention|cause|diagnosis|spread|transmission)\b'
    ]
    entities = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        entities.extend(matches)
    return list(set(entities))

def generate_suggestions(last_query, last_response):
    suggestions = []
    if any(term in last_query.lower() for term in ['symptom', 'disease', 'newcastle', 'gumboro', 'coccidiosis']):
        disease_match = re.search(r'\b(Newcastle|Gumboro|Coccidiosis|Marek\'s|IBD|Avian Influenza)\b', last_query, re.IGNORECASE)
        if disease_match:
            disease = disease_match.group(1)
            suggestions = [f"How is {disease} treated?", f"What prevents {disease}?", f"What causes {disease}?", f"How do I diagnose {disease}?"]
        else:
            suggestions = ["What are common broiler diseases?", "How to prevent disease outbreaks?", "What are Newcastle disease symptoms?", "How is Coccidiosis treated?"]
    elif any(term in last_query.lower() for term in ['feed', 'housing', 'management', 'care']):
        suggestions = ["What's the ideal feeding program?", "How should I set up broiler housing?", "What temperature is best for broilers?", "How to manage broiler waste?"]
    elif any(term in last_query.lower() for term in ['breed', 'type', 'variety', 'strain']):
        suggestions = ["Which breed is best for small farms?", "What are Cobb breed advantages?", "How do Ross breeds handle heat?", "How is Hubbard different from others?"]
    else:
        suggestions = ["What are best broiler farming practices?", "How to maximize growth rate?", "What vaccines are essential?", "How to manage ventilation properly?"]
    return suggestions

def ensure_follow_up_question(response, query):
    if response.strip().endswith('?'):
        return response
    follow_ups = {
        'breed': "Are you considering a particular breed for your farm?",
        'disease': "Are you currently dealing with sick birds in your flock?",
        'feed': "What type of feeding system are you using currently?",
        'housing': "How is your current housing setup arranged?",
        'management': "How many birds are you managing right now?",
        'general': "Is there anything else you'd like to know about this?"
    }
    query_lower = query.lower()
    if any(t in query_lower for t in ['breed', 'type', 'variety', 'strain']):
        follow_up = follow_ups['breed']
    elif any(t in query_lower for t in ['disease', 'symptom', 'treatment', 'vaccine']):
        follow_up = follow_ups['disease']
    elif any(t in query_lower for t in ['feed', 'food', 'nutrition']):
        follow_up = follow_ups['feed']
    elif any(t in query_lower for t in ['housing', 'shelter', 'coop']):
        follow_up = follow_ups['housing']
    elif any(t in query_lower for t in ['manage', 'care', 'practice']):
        follow_up = follow_ups['management']
    else:
        follow_up = follow_ups['general']
    return response + f"\n\n<div class='follow-up'>{follow_up}</div>"

def naturalize_response(response):
    impersonal_phrases = ["based on the information provided","according to the context","the context mentions that","based on the context provided","the information states that"]
    for phrase in impersonal_phrases:
        response = re.sub(phrase, "from my experience", response, flags=re.IGNORECASE)
    naturalizations = {"is described as": "is known to be","are described as": "are typically","this suggests that": "this means","it is suggested that": "I've found that","additionally": "also","furthermore": "plus","moreover": "and"}
    for pattern, replacement in naturalizations.items():
        response = re.sub(pattern, replacement, response, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", response).strip()

def ask_qa_chain(qa_chain, query: str, context: str = "") -> str:
    enhanced_query = f"{context} {query}" if context else query
    try:
        out = qa_chain.invoke({"query": enhanced_query})
        result = out["result"].strip() if isinstance(out, dict) and "result" in out else str(out).strip()
    except Exception:
        out = qa_chain({"query": enhanced_query})
        result = out.get("result", str(out)).strip() if isinstance(out, dict) else str(out).strip()
    result = naturalize_response(result)
    if not result or any(p in result.lower() for p in ["i don't know","not in the context","no information"]):
        result = "I specialize in backyard broiler farming topics like health management, feeding practices, housing setup, and disease prevention. Feel free to ask me about any of these areas!"
    return ensure_follow_up_question(result, query)

# -------------------------
# Weather + Feed Cost + RAG Routing
# -------------------------
from tools.weather import get_weather
from tools.feed_calculator import feed_cost_calculator

def handle_query(query: str, qa_chain, context: str = ""):
    q_lower = query.lower()
    # Weather
    if "weather" in q_lower or "rain" in q_lower or "temperature" in q_lower:
        city_match = re.search(r"in\s+([a-zA-Z\s]+)", query, re.IGNORECASE)
        city = city_match.group(1).strip() if city_match else "Lagos"
        return get_weather(city, "NG")
    # Feed cost
    if "feed cost" in q_lower or ("feed" in q_lower and "kg" in q_lower and "₦" in q_lower):
        bird_match = re.search(r'(\d+)\s*(?:birds|chickens)', q_lower)
        feed_match = re.search(r'(\d+(?:\.\d+)?)\s*kg', q_lower)
        price_match = re.search(r'₦?\s*(\d+(?:\.\d+)?)\s*/?\s*kg', q_lower)
        if bird_match and feed_match and price_match:
            bird_count = int(bird_match.group(1))
            feed_per_bird_kg = float(feed_match.group(1))
            price_per_kg = float(price_match.group(1))
            return feed_cost_calculator.run(bird_count, feed_per_bird_kg, price_per_kg)
        return "Please provide the number of birds, feed per bird in kg, and feed price per kg so I can calculate."
    # Default → RAG
    return ask_qa_chain(qa_chain, query, context)

# -------------------------
# App header & Input form
# -------------------------
st.title("🐔 Chikka AI")
st.write("👋 Hello! I'm **Chikka**, your friendly assistant for backyard broiler farming.")

if "suggestions" in st.session_state and st.session_state.suggestions:
    st.markdown("**You might want to ask:**")
    cols = st.columns(2)
    for i, suggestion in enumerate(st.session_state.suggestions[:4]):
        with cols[i % 2]:
            if st.button(suggestion, key=f"sugg_{i}", use_container_width=True):
                st.session_state.query_input = suggestion
                st.session_state.auto_submit = True

with st.form(key="query_form", clear_on_submit=True):
    user_query = st.text_input("Ask me about broilers:", key="query_input", placeholder="What would you like to know about broiler farming?")
    submitted = st.form_submit_button("Send")

if "auto_submit" in st.session_state and st.session_state.auto_submit:
    submitted = True
    st.session_state.auto_submit = False

# -------------------------
# Load FAISS & LLM
# -------------------------
FAISS_PATH = "rag_assets/faiss_index"
try:
    if not st.session_state.faiss_loaded:
        with st.spinner("Getting things ready for you..."):
            vectorstore = load_vectorstore(FAISS_PATH)
            llm = init_llm_from_groq()
            qa_chain = make_qa_chain(llm, vectorstore)
            st.session_state.faiss_loaded = True
            st.session_state._qa_chain = qa_chain
except Exception as e:
    st.error(f"Sorry, I'm having trouble loading my knowledge base: {e}")
    st.stop()

qa_chain = st.session_state._qa_chain

# -------------------------
# Handle a new submission
# -------------------------
if submitted and user_query and user_query.strip():
    q = user_query.strip()
    if st.session_state.history:
        recent_messages = st.session_state.history[:3]
        context_text = " ".join([msg["content"] for msg in recent_messages if msg["role"] == "User"])
        entities = extract_key_entities(context_text)
        if entities:
            st.session_state.conversation_context = f"We've been discussing: {', '.join(entities)}"
    st.session_state.history.insert(0, {"role": "User", "content": q, "time": datetime.datetime.now().strftime("%H:%M")})
    placeholder = st.empty()
    with st.spinner("Thinking about your question..."):
        answer_text = handle_query(q, qa_chain, st.session_state.conversation_context)
    placeholder.empty()
    st.session_state.history.insert(0, {"role": "ChikkaBot", "content": answer_text, "time": datetime.datetime.now().strftime("%H:%M")})
    st.session_state.suggestions = generate_suggestions(q, answer_text)

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
            role, content, timestamp = msg.get("role", "User"), msg.get("content", ""), msg.get("time", "")
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
st.write("")
st.caption("💡 Conversation history is temporary and will clear when you refresh the page.")
if st.button("🧹 Clear Conversation"):
    st.session_state.history = []
    st.session_state.conversation_context = ""
    if "suggestions" in st.session_state: del st.session_state.suggestions
    st.rerun()
