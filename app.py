import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Dict
import datetime
import re

# Import tools
from tools.weather import get_weather
from tools.calculator import calculate_feed_cost
from tools.reminder import create_vaccination_reminder

# -------------------------
# ReAct Agent Class with Enhanced State Management
# -------------------------

class ReActPoultryAgent:
    def __init__(self, qa_chain):
        self.qa_chain = qa_chain
        self.reasoning_history = []
        self.conversation_context = ""
        self.last_intent = None
        self.last_topic = None
        self.entity_memory = {}
        self.pending_action = None
        self.pending_intent = None
        self.pending_parameters = {}  # Store parameters collected during multi-turn

    def _update_conversation_memory(self, query: str, reasoning: Dict):
        """Remember important context for ALL conversation types"""
        self.last_intent = reasoning["intent"]
    
        # Remember broader topics
        if reasoning["intent"] == "vaccination_reminder":
            self.last_topic = "vaccination"
            if "newcastle" in query.lower():
                self.entity_memory["vaccine_type"] = "Newcastle"
            elif "gumboro" in query.lower():
                self.entity_memory["vaccine_type"] = "Gumboro"
            
        elif reasoning["intent"] == "feed_calculation":
            self.last_topic = "feeding"
            num_birds, feed_per_bird, price_per_kg = extract_feed_parameters(query)
            if num_birds:
                self.entity_memory["bird_count"] = num_birds
            if feed_per_bird:
                self.entity_memory["feed_amount"] = feed_per_bird
            
        elif reasoning["intent"] == "health_advice":
            self.last_topic = "health"
            diseases = ['newcastle', 'gumboro', 'coccidiosis', 'marek', 'ibd']
            for disease in diseases:
                if disease in query.lower():
                    self.entity_memory["disease"] = disease.title()
                    break
                
        elif reasoning["intent"] == "weather_inquiry":
            self.last_topic = "weather"
            city = extract_city(query)
            if city != "Lagos":
                self.entity_memory["location"] = city
     
        context_parts = []
        if self.last_topic:
            context_parts.append(f"Discussing {self.last_topic}")
        if self.entity_memory:
            for key, value in self.entity_memory.items():
                context_parts.append(f"{key}: {value}")
    
        self.conversation_context = ". ".join(context_parts) if context_parts else ""
    
    def process_query(self, query: str, context: str = "") -> Dict:
        """Enhanced ReAct pattern with persistent state management"""
        
        # STEP 1: Check if this is a follow-up to a pending action
        if self.pending_action and self._is_follow_up_response(query):
            return self._handle_follow_up_response(query)
        
        # STEP 2: Normal reasoning for new queries
        reasoning = self._reason_about_query(query, context)
        self.reasoning_history.append(reasoning)
        
        # STEP 3: Execute action
        action_result = self._execute_action(reasoning, query, context)
        
        return {
            "reasoning": reasoning,
            "result": action_result,
            "requires_follow_up": reasoning.get("missing_info")
        }
    
    def _is_follow_up_response(self, query: str) -> bool:
        """Determine if the query is answering a previous clarification question"""
        if not self.pending_action:
            return False
        
        query_lower = query.lower().strip()
        
        # For weather location - short responses likely location names
        if self.pending_action == "weather_location":
            return (len(query_lower.split()) <= 3 and 
                    not any(word in query_lower for word in ['weather', 'temperature', 'rain']))
            
        # For vaccine type - specific vaccine names or short responses
        if self.pending_action == "vaccine_type":
            vaccine_terms = ['newcastle', 'gumboro', 'coccidiosis', 'marek', 'ibd']
            return (any(vaccine in query_lower for vaccine in vaccine_terms) or 
                    len(query_lower.split()) <= 2)
            
        # For feed parameters - contains numbers
        if self.pending_action == "feed_parameters":
            return bool(re.search(r'\d+', query))
            
        # For timing - time-related words
        if self.pending_action == "timing":
            time_terms = ['tomorrow', 'monday', 'tuesday', 'wednesday', 'thursday', 
                         'friday', 'saturday', 'sunday', 'next week', 'next month']
            return any(term in query_lower for term in time_terms)
            
        return False
    
    def _handle_follow_up_response(self, query: str) -> Dict:
        """Process follow-up responses to complete pending actions"""
        result = ""
        original_intent = self.pending_intent
        original_action = self.pending_action
        
        try:
            if original_action == "weather_location":
                city = query.strip()
                if city:
                    self.entity_memory["location"] = city
                    result = get_weather(city, "NG")
                    self.pending_action = None
                    self.pending_intent = None
                else:
                    result = "I still need to know which city you want weather information for."
            
            elif original_action == "vaccine_type":
                vaccine_type = self._extract_vaccine_type(query)
                if vaccine_type:
                    self.entity_memory["vaccine_type"] = vaccine_type
                    self.pending_parameters["vaccine_type"] = vaccine_type
                    
                    # Check if we have all parameters for reminder
                    if self._has_all_reminder_parameters():
                        result = self._create_reminder_from_parameters()
                    else:
                        # Still need timing
                        self.pending_action = "timing"
                        result = self._ask_clarification("timing")
                else:
                    result = "I still need to know which vaccine you want me to remind you about."
            
            elif original_action == "timing":
                date_info = self._extract_timing(query)
                if date_info:
                    self.pending_parameters["timing"] = date_info
                    
                    if self._has_all_reminder_parameters():
                        result = self._create_reminder_from_parameters()
                    else:
                        # Still need vaccine type
                        self.pending_action = "vaccine_type"
                        result = self._ask_clarification("vaccine_type")
                else:
                    result = "I still need to know when to set the reminder for."
            
            elif original_action == "feed_parameters":
                num_birds, feed_per_bird, price_per_kg = extract_feed_parameters(query)
                if all([num_birds, feed_per_bird, price_per_kg]):
                    result = calculate_feed_cost(num_birds, feed_per_bird, price_per_kg)
                    self.pending_action = None
                    self.pending_intent = None
                else:
                    result = "I still need all three parameters: number of birds, daily feed per bird (kg), and price per kg."
        
        except Exception as e:
            result = f"I encountered an error while processing your response: {str(e)}"
        
        # Return follow-up response format
        return {
            "reasoning": {
                "intent": "follow_up_completion", 
                "original_intent": original_intent,
                "follow_up_type": original_action,
                "reasoning_steps": [f"Processing follow-up for {original_action}: {query}"]
            },
            "result": result,
            "requires_follow_up": bool(self.pending_action)
        }
    
    def _extract_vaccine_type(self, query: str) -> str:
        """Extract vaccine type from query"""
        query_lower = query.lower()
        vaccines = {
            'newcastle': 'Newcastle',
            'gumboro': 'Gumboro', 
            'coccidiosis': 'Coccidiosis',
            'marek': 'Marek',
            'ibd': 'IBD'
        }
        
        for vaccine_key, vaccine_name in vaccines.items():
            if vaccine_key in query_lower:
                return vaccine_name
        
        # If no specific vaccine found, use the query as-is for short responses
        if len(query.strip().split()) <= 2:
            return query.strip().title()
        
        return None
    
    def _extract_timing(self, query: str) -> str:
        """Extract timing information from query"""
        query_lower = query.lower()
        
        timing_patterns = [
            r'(tomorrow|next week|in 2 weeks|in 1 month)',
            r'(\d{1,2}[/-]\d{1,2}[/-]?\d{0,4})',
            r'(monday|tuesday|wednesday|thursday|friday|saturday|sunday)',
            r'(next monday|next tuesday|next wednesday|next thursday|next friday|next saturday|next sunday)'
        ]
        
        for pattern in timing_patterns:
            match = re.search(pattern, query_lower)
            if match:
                return match.group(1)
        
        # Default for very short responses
        if len(query.strip().split()) <= 2:
            return query.strip()
        
        return None
    
    def _has_all_reminder_parameters(self) -> bool:
        """Check if we have all parameters needed for a reminder"""
        return (self.pending_parameters.get("vaccine_type") and 
                self.pending_parameters.get("timing"))
    
    def _create_reminder_from_parameters(self) -> str:
        """Create reminder using collected parameters"""
        vaccine_type = self.pending_parameters["vaccine_type"]
        timing = self.pending_parameters["timing"]
        bird_count = self.entity_memory.get("bird_count")
        
        # Clear pending state
        self.pending_action = None
        self.pending_intent = None
        self.pending_parameters = {}
        
        return create_vaccination_reminder(vaccine_type, timing, bird_count)
    
    def _reason_about_query(self, query: str, context: str) -> Dict:
        """Analyze the query and plan actions"""
        reasoning_steps = []
        
        # Step 1: Identify primary intent
        intent = self._identify_intent(query)
        reasoning_steps.append(f"Primary intent: {intent}")
        
        # Step 2: Check for required information
        missing_info = self._check_required_info(query, intent)
        reasoning_steps.append(f"Missing information: {missing_info}")
        
        # Step 3: Plan tool usage
        tool_plan = self._plan_tool_usage(intent, missing_info)
        reasoning_steps.append(f"Tool plan: {tool_plan}")
        
        return {
            "intent": intent,
            "missing_info": missing_info,
            "reasoning_steps": reasoning_steps,
            "tool_plan": tool_plan,
            "requires_clarification": bool(missing_info)
        }
    
    def _identify_intent(self, query: str) -> str:
        """Enhanced intent recognition"""
        query_lower = query.lower()
        
        if any(term in query_lower for term in ['weather', 'temperature', 'rain']):
            return "weather_inquiry"
        elif any(term in query_lower for term in ['feed cost', 'calculate', 'budget']):
            return "feed_calculation"
        elif any(term in query_lower for term in ['remind', 'vaccine', 'schedule']):
            return "vaccination_reminder"
        elif any(term in query_lower for term in ['disease', 'symptom', 'treatment']):
            return "health_advice"
        elif any(term in query_lower for term in ['housing', 'ventilation', 'shelter']):
            return "housing_advice"
        else:
            return "general_inquiry"
    
    def _check_required_info(self, query: str, intent: str) -> List[str]:
        """Reason about missing information needed to complete the task"""
        missing = []
        
        if intent == "feed_calculation":
            numbers = re.findall(r'\d+\.?\d*', query)
            if len(numbers) < 3:
                missing.append("feed_parameters")
        
        elif intent == "vaccination_reminder":
            if not any(vaccine in query.lower() for vaccine in ['newcastle', 'gumboro', 'coccidiosis']):
                missing.append("vaccine_type")
            if not any(word in query.lower() for word in ['tomorrow', 'monday', 'next', 'date']):
                missing.append("timing")
        
        elif intent == "weather_inquiry":
            if "in " not in query.lower():
                missing.append("location")
        
        return missing
    
    def _plan_tool_usage(self, intent: str, missing_info: List[str]) -> Dict:
        """Plan which tools to use and in what order"""
        if missing_info:
            return {
                "primary_action": "clarify",
                "tools": [],
                "clarification_type": missing_info[0]
            }
        
        tool_map = {
            "weather_inquiry": {"primary_action": "get_weather", "tools": ["weather"]},
            "feed_calculation": {"primary_action": "calculate_feed", "tools": ["calculator"]},
            "vaccination_reminder": {"primary_action": "create_reminder", "tools": ["reminder"]},
            "health_advice": {"primary_action": "provide_advice", "tools": ["qa_chain"]},
            "housing_advice": {"primary_action": "provide_advice", "tools": ["qa_chain"]},
            "general_inquiry": {"primary_action": "provide_advice", "tools": ["qa_chain"]}
        }
        
        return tool_map.get(intent, {"primary_action": "provide_advice", "tools": ["qa_chain"]})
    
    def _execute_action(self, reasoning: Dict, query: str, context: str) -> str:
        """Execute the planned action based on reasoning"""
        tool_plan = reasoning["tool_plan"]
        
        if tool_plan["primary_action"] == "clarify":
            # Set pending action state
            self.pending_action = tool_plan["clarification_type"]
            self.pending_intent = reasoning["intent"]
            return self._ask_clarification(tool_plan["clarification_type"])
        
        # Route to appropriate tool
        if tool_plan["primary_action"] == "get_weather":
            city = extract_city(query) or self.entity_memory.get("location", "Lagos")
            return get_weather(city, "NG")
        
        elif tool_plan["primary_action"] == "calculate_feed":
            num_birds, feed_per_bird, price_per_kg = extract_feed_parameters(query)
            if all([num_birds, feed_per_bird, price_per_kg]):
                return calculate_feed_cost(num_birds, feed_per_bird, price_per_kg)
            else:
                self.pending_action = "feed_parameters"
                self.pending_intent = reasoning["intent"]
                return "I can help calculate feed costs! Please include: number of birds, daily feed per bird (kg), and price per kg of feed."
        
        elif tool_plan["primary_action"] == "create_reminder":
            vaccine_type, date_info, bird_count = extract_reminder_parameters(query)
            # Use memory fallbacks
            vaccine_type = vaccine_type or self.entity_memory.get("vaccine_type")
            bird_count = bird_count or self.entity_memory.get("bird_count")
            
            if vaccine_type and date_info:
                return create_vaccination_reminder(vaccine_type, date_info, bird_count)
            else:
                # Determine what's missing
                if not vaccine_type:
                    self.pending_action = "vaccine_type"
                else:
                    self.pending_action = "timing"
                self.pending_intent = reasoning["intent"]
                return self._ask_clarification(self.pending_action)

        else:  # provide_advice - use your existing RAG
            return ask_qa_chain(self.qa_chain, query, context)
    
    def _ask_clarification(self, clarification_type: str) -> str:
        """Ask intelligent clarification questions based on reasoning"""
        clarifications = {
            "feed_parameters": "I'd be happy to calculate feed costs! Could you tell me: how many birds, how much feed each bird gets daily (in kg), and the price per kg of feed?",
            "vaccine_type": "Which vaccine would you like me to remind you about? (Newcastle, Gumboro, Coccidiosis, etc.)",
            "timing": "When should I set this vaccination reminder for? (tomorrow, next Monday, specific date)",
            "location": "Which city would you like the weather information for?"
        }
        return clarifications.get(clarification_type, "Could you provide a bit more detail so I can help you better?")

# -------------------------
# UI / Appearance settings
# -------------------------

st.set_page_config(page_title="Chikka AI Assistant", layout="centered")

# Updated CSS for chat bubbles + scroll area with auto-scroll
st.markdown(
    """
    <style>
    .chat-container {
        max-height: 400px;
        overflow-y: auto;
        padding: 8px;
        border-radius: 8px;
        border: 1px solid #eee;
        background: #fafafa;
        display: flex;
        flex-direction: column;
        margin-bottom: 20px;
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
    .pending-action {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 8px 12px;
        border-radius: 6px;
        margin: 5px 0;
        font-size: 12px;
        color: #856404;
    }
    .input-container {
        position: sticky;
        bottom: 0;
        background: white;
        padding: 10px 0;
        border-top: 1px solid #e0e0e0;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# JavaScript for auto-scroll
st.markdown(
    """
    <script>
    function scrollToBottom() {
        const container = document.querySelector('.chat-container');
        if (container) {
            container.scrollTop = container.scrollHeight;
        }
    }
    
    // Scroll when page loads
    window.addEventListener('load', scrollToBottom);
    
    // Scroll when new messages are added
    const observer = new MutationObserver(scrollToBottom);
    const config = { childList: true, subtree: true };
    window.addEventListener('load', function() {
        const container = document.querySelector('.chat-container');
        if (container) {
            observer.observe(container, config);
        }
    });
    </script>
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

    from langchain.prompts import PromptTemplate
    prompt_template = """You are Chikka, a friendly expert AI assistant specialized in backyard broiler farming.

Provide helpful, conversational answers that are clear and focused. Be naturally conversational but avoid unnecessary fluff.

After providing your main answer, end with a natural follow-up question that continues the conversation.
Make the follow-up relevant to the topic and helpful for the farmer.

Context: {context}

Question: {question}

Answer in a friendly, expert tone. Share knowledge conversationally while staying on topic.
End with a helpful follow-up question to continue the conversation."""

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
            suggestions = [
                f"How is {disease} treated?",
                f"What prevents {disease}?",
                f"What causes {disease}?",
                f"How do I diagnose {disease}?"
            ]
        else:
            suggestions = [
                "What are common broiler diseases?",
                "How to prevent disease outbreaks?",
                "What are Newcastle disease symptoms?",
                "How is Coccidiosis treated?"
            ]
    elif any(term in last_query.lower() for term in ['feed', 'housing', 'management', 'care']):
        suggestions = [
            "What's the ideal feeding program?",
            "How should I set up broiler housing?",
            "What temperature is best for broilers?",
            "How to manage broiler waste?"
        ]
    elif any(term in last_query.lower() for term in ['breed', 'type', 'variety', 'strain']):
        suggestions = [
            "Which breed is best for small farms?",
            "What are Cobb breed advantages?",
            "How do Ross breeds handle heat?",
            "How is Hubbard different from others?"
        ]
    else:
        suggestions = [
            "What are best broiler farming practices?",
            "How to maximize growth rate?",
            "What vaccines are essential?",
            "How to manage ventilation properly?"
        ]

    return suggestions

def ensure_follow_up_question(response, query):
    if response.strip().endswith('?'):
        return response

    response_lower = response.lower()
    query_lower = query.lower()
    
    follow_ups = {
        'breed': "Are you considering a particular breed for your farm?",
        'disease': "Are you currently dealing with sick birds in your flock?",
        'feed': "What type of feeding system are you using currently?",
        'housing': "How is your current housing setup arranged?",
        'management': "How many birds are you managing right now?",
        'general': "Is there anything else you'd like to know about this?"
    }
    
    if any(term in query_lower for term in ['breed', 'type', 'variety', 'strain']) or any(term in response_lower for term in ['breed', 'hubbard', 'cobb', 'ross']):
        follow_up = follow_ups['breed']
    elif any(term in query_lower for term in ['disease', 'symptom', 'treatment', 'vaccine', 'sick', 'ill']) or any(term in response_lower for term in ['disease', 'symptom', 'treatment', 'vaccine', 'sick', 'ill']):
        follow_up = follow_ups['disease']
    elif any(term in query_lower for term in ['feed', 'food', 'nutrition', 'diet']) or any(term in response_lower for term in ['feed', 'food', 'nutrition', 'diet']):
        follow_up = follow_ups['feed']
    elif any(term in query_lower for term in ['housing', 'shelter', 'coop', 'ventilation']) or any(term in response_lower for term in ['housing', 'shelter', 'coop', 'ventilation']):
        follow_up = follow_ups['housing']
    elif any(term in query_lower for term in ['manage', 'care', 'practice', 'operation']) or any(term in response_lower for term in ['manage', 'care', 'practice', 'operation']):
        follow_up = follow_ups['management']
    else:
        follow_up = follow_ups['general']
    
    return response + f"\n\n<div class='follow-up'>{follow_up}</div>"

def naturalize_response(response):
    impersonal_phrases = [
        "based on the information provided",
        "according to the context",
        "the context mentions that",
        "based on the context provided",
        "the information states that"
    ]
    for phrase in impersonal_phrases:
        response = re.sub(phrase, "from my experience", response, flags=re.IGNORECASE)

    naturalizations = {
        r"is described as": "is known to be",
        r"are described as": "are typically",
        r"this suggests that": "this means",
        r"it is suggested that": "I've found that",
        r"additionally": "also",
        r"furthermore": "plus",
        r"moreover": "and"
    }
    for pattern, replacement in naturalizations.items():
        response = re.sub(pattern, replacement, response, flags=re.IGNORECASE)
    
    response = re.sub(r"\s+", " ", response).strip()
    return response

def ask_qa_chain(qa_chain, query: str, context: str = "") -> str:
    enhanced_query = f"{context} {query}" if context else query

    try:
        out = qa_chain.invoke({"query": enhanced_query})
        result = out["result"].strip() if isinstance(out, dict) and "result" in out else str(out).strip()
    except Exception:
        try:
            out = qa_chain({"query": enhanced_query})
            result = out.get("result", str(out)).strip() if isinstance(out, dict) else str(out).strip()
        except Exception as e:
            result = f"I encountered an error: {str(e)}"
    
    result = naturalize_response(result)
    
    no_knowledge_phrases = [
        "i don't know", "i don't have information", "not in the context", 
        "not provided in the context", "no information", "not covered"
    ]
    if not result or any(phrase in result.lower() for phrase in no_knowledge_phrases):
        result = "I specialize in backyard broiler farming topics like health management, feeding practices, housing setup, and disease prevention. Feel free to ask me about any of these areas!"
        result = ensure_follow_up_question(result, query)
    else:
        result = ensure_follow_up_question(result, query)
    
    return result

def extract_city(query: str) -> str:
    match = re.search(r"in\s+([A-Za-z\s]+)", query, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return "Lagos"  # fallback default

def extract_feed_parameters(query: str) -> tuple:
    """
    Extract num_birds, feed_per_bird, and price_per_kg from natural language query.
    Returns (num_birds, feed_per_bird, price_per_kg) or (None, None, None) if not found.
    """
    try:
        # Find all numbers in the query (including decimals)
        numbers = re.findall(r'\d+\.?\d*', query)
        numbers = [float(num) for num in numbers]
        
        if len(numbers) < 3:
            return None, None, None
        
        # Simple heuristic: assume order is birds, feed_per_bird, price
        # But we can make it smarter based on context clues
        num_birds = None
        feed_per_bird = None
        price_per_kg = None
        
        # Look for context clues to assign numbers correctly
        query_lower = query.lower()
        
        # Try to identify which number is which based on context
        for i, num in enumerate(numbers):
            # Look for bird count indicators
            if any(word in query_lower for word in ['bird', 'chicken', 'broiler', 'flock', 'stock']):
                # The number near these words is likely the bird count
                if num_birds is None and (20 <= num <= 10000):  # Reasonable bird count range
                    num_birds = int(num)
                    continue
            
            # Look for feed amount indicators
            if any(word in query_lower for word in ['kg', 'kilogram', 'feed', 'consum', 'eat']):
                if feed_per_bird is None and (0.1 <= num <= 5.0):  # Reasonable feed per bird range
                    feed_per_bird = num
                    continue
            
            # Look for price indicators
            if any(word in query_lower for word in ['price', 'cost', '₦', 'naira', 'per kg', 'per kilogram']):
                if price_per_kg is None and (100 <= num <= 5000):  # Reasonable price range
                    price_per_kg = num
                    continue
    
        # If we couldn't identify all parameters by context, use positional fallback
        if num_birds is None and len(numbers) >= 1:
            num_birds = int(numbers[0])
        if feed_per_bird is None and len(numbers) >= 2:
            feed_per_bird = numbers[1]
        if price_per_kg is None and len(numbers) >= 3:
            price_per_kg = numbers[2]
        
        return num_birds, feed_per_bird, price_per_kg
        
    except Exception as e:
        return None, None, None

def extract_reminder_parameters(query: str) -> tuple:
    """
    Extract vaccine type and date from natural language query
    """
    try:
        query_lower = query.lower()
        
        # Extract vaccine type
        vaccines = ['newcastle', 'gumboro', 'coccidiosis', 'marek', 'ibd', 'avian influenza', 
                   'fowl pox', 'fowl cholera', 'crd']
        vaccine_type = None
        for vaccine in vaccines:
            if vaccine in query_lower:
                vaccine_type = vaccine.title()
                break
        
        if not vaccine_type:
            vaccine_type = "Poultry Vaccine"  # Default
        
        # Extract bird count
        bird_match = re.search(r'(\d+)\s*(birds?|chickens?|broilers?)', query_lower)
        bird_count = int(bird_match.group(1)) if bird_match else None
        
        # Extract date (simple pattern)
        date_patterns = [
            r'(tomorrow|next week|in 2 weeks|in 1 month)',
            r'(\d{1,2}[/-]\d{1,2}[/-]?\d{0,4})',
            r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}',
            r'(monday|tuesday|wednesday|thursday|friday|saturday|sunday)'
        ]
        
        date_info = None
        for pattern in date_patterns:
            match = re.search(pattern, query_lower)
            if match:
                date_info = match.group(1)
                break
        
        if not date_info:
            date_info = "soon"  # Default
        
        return vaccine_type, date_info, bird_count
        
    except Exception as e:
        return None, None, None

def handle_query(query: str, qa_chain, context: str = ""):
    # Always use ReAct agent for consistent state management
    if "react_agent" in st.session_state:
        result = st.session_state.react_agent.process_query(query, context)
        
        # Show reasoning in expander (optional) - with safe access
        with st.expander("🔍 See my thought process"):
            reasoning = result.get("reasoning", {})
            reasoning_steps = reasoning.get("reasoning_steps", ["Processing your request..."])
            for step in reasoning_steps:
                st.write(f"• {step}")
        
        return result["result"]
    else:
        # Fallback - should not happen with proper initialization
        return ask_qa_chain(qa_chain, query, context)

# -------------------------
# App header & Introduction
# -------------------------

st.title("🐔 Chikka AI")
st.write(
    "👋 Hello! I'm Chikka, your friendly assistant for backyard broiler farming. "
    "I'm here to help with practical advice on broiler care, health, and management."
)

# Show pending action status if any - with safe attribute checking
if "react_agent" in st.session_state:
    agent = st.session_state.react_agent
    # Safe attribute check
    if hasattr(agent, 'pending_action') and agent.pending_action:
        pending_action = agent.pending_action
        action_descriptions = {
            "weather_location": "📍 Waiting for city name for weather information",
            "vaccine_type": "💉 Waiting for vaccine type for reminder",
            "timing": "⏰ Waiting for timing for reminder",
            "feed_parameters": "💰 Waiting for feed calculation parameters"
        }
        st.markdown(f'<div class="pending-action">{action_descriptions.get(pending_action, "Waiting for your input")}</div>', 
                    unsafe_allow_html=True)

# -------------------------
# Conversation History Display
# -------------------------

st.markdown("### Conversation")
chat_box = st.container()

with chat_box:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    if not st.session_state.history:
        st.markdown("<p style='color: #888; text-align: center; padding: 20px;'>No messages yet. Ask me anything about broiler farming!</p>", unsafe_allow_html=True)
    else:
        # Display messages in chronological order (top to bottom)
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
# Input Section at Bottom
# -------------------------

st.markdown('<div class="input-container">', unsafe_allow_html=True)

# Suggestions (if any)
if "suggestions" in st.session_state and st.session_state.suggestions:
    st.markdown("**You might want to ask:**")
    cols = st.columns(2)
    for i, suggestion in enumerate(st.session_state.suggestions[:4]):
        with cols[i % 2]:
            if st.button(suggestion, key=f"sugg_{i}", use_container_width=True):
                st.session_state.query_input = suggestion
                st.session_state.auto_submit = True

# Input form at the bottom
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

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Load FAISS & LLM lazily (only once)
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
            
            # Always create fresh agent instance to ensure new attributes
            st.session_state.react_agent = ReActPoultryAgent(qa_chain)
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
        recent_messages = list(reversed(st.session_state.history))[:3]
        context_text = " ".join([msg["content"] for msg in recent_messages if msg["role"] == "User"])
        entities = extract_key_entities(context_text)
        if entities:
            st.session_state.conversation_context = f"We've been discussing: {', '.join(entities)}"

    st.session_state.history.append(
        {"role": "User", "content": q, "time": datetime.datetime.now().strftime("%H:%M")}
    )

    placeholder = st.empty()
    with st.spinner("Thinking about your question..."):
        answer_text = handle_query(q, qa_chain, st.session_state.conversation_context)
    placeholder.empty()

    st.session_state.history.append(
        {"role": "ChikkaBot", "content": answer_text, "time": datetime.datetime.now().strftime("%H:%M")}
    )
    
    st.session_state.suggestions = generate_suggestions(q, answer_text)
    st.rerun()

# -------------------------
# Footer
# -------------------------

st.write("")
st.caption("💡 Conversation history is temporary and will clear when you refresh the page.")

if st.button("🧹 Clear Conversation"):
    st.session_state.history = []
    st.session_state.conversation_context = ""
    if "suggestions" in st.session_state:
        del st.session_state.suggestions
    if "react_agent" in st.session_state:
        # Reset agent state
        st.session_state.react_agent.pending_action = None
        st.session_state.react_agent.pending_intent = None
        st.session_state.react_agent.pending_parameters = {}
        st.session_state.react_agent.entity_memory = {}
    st.rerun()
