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
    .suggestion-chip {
        display: inline-block;
        background-color: #f0f8ff;
        border: 1px solid #d0e8ff;
        border-radius: 16px;
        padding: 4px 12px;
        margin: 4px 4px 4px 0;
        font-size: 14px;
        cursor: pointer;
        transition: background-color 0.2s;
    }
    .suggestion-chip:hover {
        background-color: #e0f0ff;
    }
    .follow-up {
        margin-top: 12px;
        padding-top: 8px;
        border-top: 1px dashed #e0e0e0;
        font-style: italic;
        color: #666;
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


def make_qa_chain(llm, vectorstore):
    retriever = vectorstore.as_retriever()
    
    # Custom prompt to make responses more personalized
    from langchain.prompts import PromptTemplate
    prompt_template = """You are Chikka, an expert AI assistant specialized in backyard broiler farming. 
    You provide friendly, personalized advice based on your extensive knowledge.

    Context: {context}

    Question: {question}

    Answer as if you're sharing your own expertise. Avoid phrases like "based on the information provided" 
    or "according to the context". Instead, present the information as your own knowledge.

    Provide a helpful, personalized response:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        retriever=retriever, 
        return_source_documents=False,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain


def extract_key_entities(text):
    """Extract potential entities (diseases, topics) from text for context tracking"""
    # Simple pattern matching for disease names and important terms
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
    """Generate follow-up suggestions based on the last query and response"""
    suggestions = []
    
    # Disease-related suggestions
    if any(term in last_query.lower() for term in ['symptom', 'disease', 'newcastle', 'gumboro', 'coccidiosis']):
        disease_match = re.search(r'\b(Newcastle|Gumboro|Coccidiosis|Marek\'s|IBD|Avian Influenza)\b', last_query, re.IGNORECASE)
        if disease_match:
            disease = disease_match.group(1)
            suggestions = [
                f"How is {disease} treated?",
                f"How can I prevent {disease} in my flock?",
                f"What are the causes of {disease}?",
                f"How is {disease} diagnosed?"
            ]
        else:
            suggestions = [
                "What are common broiler diseases?",
                "How to prevent disease outbreaks?",
                "What are the symptoms of Newcastle disease?",
                "How to treat Coccidiosis?"
            ]
    
    # Management-related suggestions
    elif any(term in last_query.lower() for term in ['feed', 'housing', 'management', 'care']):
        suggestions = [
            "What is the ideal broiler feeding program?",
            "How to properly house broiler chickens?",
            "What is the ideal temperature for broilers?",
            "How to manage broiler waste?"
        ]
    
    # Breed-related suggestions
    elif any(term in last_query.lower() for term in ['breed', 'type', 'variety', 'strain']):
        suggestions = [
            "Which breed is best for small-scale farming?",
            "What are the advantages of Cobb breeds?",
            "How do Ross breeds perform in hot climates?",
            "What's the difference between Hubbard and other breeds?"
        ]
    
    # General broiler farming suggestions
    else:
        suggestions = [
            "What are the best practices for broiler farming?",
            "How to maximize broiler growth rate?",
            "What vaccines are essential for broilers?",
            "How to manage broiler ventilation?"
        ]
    
    return suggestions


def add_follow_up_prompt(response, query):
    """Add a natural follow-up question based on the response content"""
    follow_ups = {
        'breed': "Would you like me to compare different breeds for your specific situation?",
        'disease': "Would you like more details about preventing or treating this condition?",
        'feed': "Should I provide more specific feeding recommendations for your flock?",
        'housing': "Do you need advice on optimizing your housing setup?",
        'management': "Would you like tips on implementing this in your operation?",
        'default': "Is there anything else you'd like to know about this topic?"
    }
    
    # Determine the most relevant follow-up
    response_lower = response.lower()
    query_lower = query.lower()
    
    if any(term in query_lower for term in ['breed', 'type', 'variety', 'strain']) or any(term in response_lower for term in ['breed', 'hubbard', 'cobb', 'ross']):
        follow_up = follow_ups['breed']
    elif any(term in query_lower for term in ['disease', 'symptom', 'treatment', 'vaccine']) or any(term in response_lower for term in ['disease', 'symptom', 'treatment', 'vaccine']):
        follow_up = follow_ups['disease']
    elif any(term in query_lower for term in ['feed', 'food', 'nutrition', 'diet']) or any(term in response_lower for term in ['feed', 'food', 'nutrition', 'diet']):
        follow_up = follow_ups['feed']
    elif any(term in query_lower for term in ['housing', 'shelter', 'coop', 'ventilation']) or any(term in response_lower for term in ['housing', 'shelter', 'coop', 'ventilation']):
        follow_up = follow_ups['housing']
    elif any(term in query_lower for term in ['manage', 'care', 'practice', 'operation']) or any(term in response_lower for term in ['manage', 'care', 'practice', 'operation']):
        follow_up = follow_ups['management']
    else:
        follow_up = follow_ups['default']
    
    # Add the follow-up to the response
    return response + f"\n\n*{follow_up}*"


def personalize_response(response):
    """Remove impersonal phrases and make the response more natural"""
    impersonal_phrases = [
        "based on the information provided",
        "according to the context",
        "the context mentions",
        "based on the context",
        "the information states",
        "according to the information"
    ]
    
    for phrase in impersonal_phrases:
        response = re.sub(phrase, "based on my knowledge", response, flags=re.IGNORECASE)
        response = re.sub(phrase, "I recommend", response, flags=re.IGNORECASE)
        response = re.sub(phrase, "in my experience", response, flags=re.IGNORECASE)
    
    # Additional personalization
    response = re.sub(r"is described as", "is known to be", response, flags=re.IGNORECASE)
    response = re.sub(r"are described as", "are known to be", response, flags=re.IGNORECASE)
    response = re.sub(r"this suggests that", "this means", response, flags=re.IGNORECASE)
    response = re.sub(r"it is suggested that", "I've found that", response, flags=re.IGNORECASE)
    
    return response


def ask_qa_chain(qa_chain, query: str, context: str = "") -> str:
    """
    Ask the QA chain with enhanced context handling and fallback responses.
    """
    # Enhance query with conversation context
    enhanced_query = f"{context} {query}" if context else query
    
    try:
        out = qa_chain.invoke({"query": enhanced_query})
        if isinstance(out, dict) and "result" in out:
            result = out["result"].strip()
        else:
            result = str(out).strip()
    except Exception:
        try:
            out = qa_chain({"query": enhanced_query})
            if isinstance(out, dict):
                result = out.get("result", str(out)).strip()
            else:
                result = str(out).strip()
        except Exception as e:
            result = f"Error while querying LLM: {e}"

    # Personalize the response
    result = personalize_response(result)
    
    # Improved fallback response
    no_knowledge_phrases = [
        "i don't know", "i don't have information", "not in the context", 
        "not provided in the context", "no information", "not covered"
    ]
    
    if not result or any(phrase in result.lower() for phrase in no_knowledge_phrases):
        result = (
            "I don't have specific knowledge about that topic in my training. "
            "As Chikka AI, I'm specialized in providing information about backyard broiler farming, "
            "including health management, feeding practices, housing requirements, and disease prevention. "
            "Feel free to ask me about any of these broiler-related topics!"
        )
    else:
        # Add a natural follow-up question
        result = add_follow_up_prompt(result, query)

    return result


# -------------------------
# App header & Input form
# -------------------------
st.title("🐔 Chikka AI")
st.write(
    "👋 Hello, I'm **Chikka** — your virtual assistant for backyard broiler farming in Nigeria. "
    "I provide expert-backed answers on broiler care, health, management, and more. "
    "I'm here to support you with reliable guidance, anytime you need."
)

# Display suggestion chips if available
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
        placeholder="Type your question here..."
    )
    submitted = st.form_submit_button("Send")

# Handle auto-submission from suggestions
if "auto_submit" in st.session_state and st.session_state.auto_submit:
    submitted = True
    st.session_state.auto_submit = False

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
    
    # Update conversation context with entities from previous messages
    if st.session_state.history:
        # Get the last few messages to maintain context
        recent_messages = st.session_state.history[:3]  # Last 3 messages
        context_text = " ".join([msg["content"] for msg in recent_messages if msg["role"] == "User"])
        entities = extract_key_entities(context_text)
        if entities:
            st.session_state.conversation_context = f"Recent discussion mentioned: {', '.join(entities)}."

    # 1) Add user message to history
    st.session_state.history.insert(
        0,
        {"role": "User", "content": q, "time": datetime.datetime.now().strftime("%H:%M")}
    )

    # 2) Show thinking spinner
    placeholder = st.empty()
    with st.spinner("🐔 ChikkaBot is thinking..."):
        answer_text = ask_qa_chain(qa_chain, q, st.session_state.conversation_context)
    placeholder.empty()

    # 3) Add assistant reply
    st.session_state.history.insert(
        0,
        {"role": "ChikkaBot", "content": answer_text, "time": datetime.datetime.now().strftime("%H:%M")}
    )
    
    # 4) Generate follow-up suggestions
    st.session_state.suggestions = generate_suggestions(q, answer_text)

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
    st.session_state.conversation_context = ""
    if "suggestions" in st.session_state:
        del st.session_state.suggestions
    st.rerun()
