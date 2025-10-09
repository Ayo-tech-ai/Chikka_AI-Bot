🐔 Chikka_AI: RAG-Powered Poultry Assistant

> “Your friendly AI companion for smarter backyard poultry farming.”




---

🌍 Overview

Chikka_AI is a Retrieval-Augmented Generation (RAG) powered conversational assistant designed to help backyard poultry farmers make better, faster, and data-driven decisions.

The assistant provides:

🧠 Health and disease advice

💰 Feed cost calculations

💉 Vaccination reminders

🌦️ Weather insights

💬 Conversational follow-ups and memory


It’s built with a ReAct (Reason + Act) reasoning framework that combines LangChain, FAISS, and Groq LLMs, wrapped in a clean Streamlit interface for real-time interactions.


---

🧩 Features

Feature	Description

🧠 RAG-based Knowledge Retrieval	Uses a FAISS vector database to fetch contextually relevant poultry information before answering.
⚙️ ReAct Reasoning Framework	The agent first reasons about your query, then performs the right action (calculate, remind, answer, etc.).
💰 Feed Cost Calculator	Calculates feed cost automatically when you specify bird count, feed per bird, and price per kg.
💉 Vaccination Reminder Tool	Creates smart vaccination reminders for diseases like Newcastle, Gumboro, and Coccidiosis.
🌦️ Weather Information	Retrieves live weather updates for your specified location (using OpenWeather or similar APIs).
🧾 Memory & Context Awareness	Keeps track of previous topics and entities during conversation for more natural interactions.
🐥 Domain-Focused Expertise	Tailored responses specifically for backyard broiler poultry farming.



---

🧱 System Architecture

┌──────────────────────────────┐
│        User Interface         │
│     (Streamlit Chat UI)       │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│      ReAct Reasoning Layer    │
│  (Intent recognition, memory, │
│   and tool planning logic)    │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│   RAG Pipeline (LangChain)    │
│  - FAISS Vector Store         │
│  - Groq Llama 3.3 LLM         │
│  - Custom Prompt Template     │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│       External Tools          │
│ - Weather API                 │
│ - Feed Cost Calculator        │
│ - Vaccination Reminder        │
└──────────────────────────────┘


---

🧠 Core Technologies

Component	Library / Tool

LLM	Groq Llama 3.3 (via LangChain_Groq)
RAG Engine	LangChain + FAISS Vectorstore
Embeddings	Sentence Transformers (MiniLM-L6-v2)
Framework	Streamlit
Reasoning Pattern	ReAct (Reason + Act)
Tools	Custom Python tools for Weather, Feed Cost, Vaccination
Memory	In-app context tracking with entity memory



---

🚀 Getting Started

1. Clone the Repository

git clone https://github.com/Ayo-tech-ai/Chikka_AI-Bot.git
cd Chikka_AI

2. Install Requirements

pip install -r requirements.txt

3. Add Secrets (Groq API Key)

In Streamlit, open your secrets manager and add:

[GROQ_API_KEY]
your_api_key_here

4. Run the App

streamlit run app.py


---

💬 Example Conversations

User: “Can you help me calculate feed cost for 200 birds eating 0.12kg daily at ₦450 per kg?”
Chikka_AI: “The estimated daily feed cost for 200 birds is ₦10,800. Would you like me to estimate the monthly feed budget too?”


---

User: “Remind me to vaccinate for Newcastle next Monday.”
Chikka_AI: “Got it! I’ve set a reminder for Newcastle vaccination next Monday. Do you want me to add Gumboro as well?”


---

User: “What are the symptoms of Coccidiosis?”
Chikka_AI: “Coccidiosis often causes bloody droppings, ruffled feathers, and weakness. Are you currently seeing these signs in your flock?”


---

💡 Innovation

Unlike regular chatbots, Chikka_AI merges RAG retrieval, reasoning, and tool execution within a single adaptive pipeline.
It doesn’t just answer — it thinks, acts, and follows up like a real farming assistant.


---

🌱 Impact

By empowering smallholder poultry farmers with easy access to expert-level insights, Chikka_AI:

Reduces bird mortality rates 🐣

Optimizes feeding efficiency 💰

Improves vaccination planning 💉

Promotes sustainable backyard farming 🌍




---

🧑🏽‍💻 Author

Ayoola Mujib Ayodele
Data Scientist / AI Engineer
📧 ayodelemujibayoola@gmail.com
📞 08136626696


---

🪪 License

This project is for educational and hackathon demonstration purposes.
© 2025 Chikka_AI by Ayoola Mujib Ayodele.


---
