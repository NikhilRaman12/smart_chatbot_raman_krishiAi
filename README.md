RamanKrishiAI — Smart-Agri Chatbot

An AI-powered multilingual chatbot designed to democratize agricultural expertise for farmers, agronomists, and enterprise stakeholders.
RamanKrishiAI integrates machine learning and natural language processing to deliver real-time insights on sustainable farming practices, crop management, and agronomic recommendations.

Objective

The objective of RamanKrishiAI is to make agricultural knowledge universally accessible through artificial intelligence.
By combining retrieval-augmented generation (RAG) pipelines, semantic embeddings, and multilingual support, the system enables farmers and agronomists to ask domain-specific questions and receive contextually relevant, evidence-based responses in real time.

Live Demo

Coming soon:

Render Deployment: https://raman-krishi-ai.onrender.com

Hugging Face Spaces: https://huggingface.co/spaces/NikhilRaman12/smart_chatbot_raman_krishiAi

Tech Stack
Component	Purpose
FastAPI	Backend framework for API deployment
LangChain	Orchestrates the retrieval-augmented generation (RAG) logic
FAISS	Vector search for semantic similarity matching
Hugging Face Transformers	Generates embeddings using sentence-transformers/all-MiniLM-L6-v2
Gradio	Lightweight user interface for chatbot interaction
Python-dotenv	Environment variable management for API keys and configuration
Multilingual Support

RamanKrishiAI supports user interaction in multiple Indian languages, including:

English

Hindi

Telugu

Tamil

Marathi

Gujarati

Future versions will include translation pipelines and speech-based interfaces to improve accessibility.

Project Structure
smart_insights_dashboard/
├── src/
│   ├── main.py              # FastAPI application entry point
│   ├── embeddings/          # FAISS embedding and vector storage
│   ├── utils/               # Helper modules
│   ├── requirements.txt     # Project dependencies
│   └── runtime.txt          # Python runtime for Render deployment
├── .gitignore
├── README.md
└── data/
    └── faiss_index/
        ├── index.faiss
        └── index.pkl

Local Setup and Testing
# Clone the repository
git clone https://github.com/NikhilRaman12/smart_insights_dashboard.git
cd smart_insights_dashboard

# Create and activate virtual environment
python -m venv venv311
.\venv311\Scripts\Activate.ps1  # For PowerShell

# Install dependencies
pip install -r src/requirements.txt

# Run the application
uvicorn src.main:app --reload


Access the application locally at:

Application: http://127.0.0.1:8000

API Documentation: http://127.0.0.1:8000/docs

Use Cases

Precision agriculture and smart crop management

Sustainable practice recommendations

Knowledge retrieval from curated agronomic datasets

Real-time assistance for farmers and field experts

Multilingual conversational support for inclusive access

Integrations
Tool	Functionality
LangChain + FAISS	Context retrieval and semantic search
Hugging Face	Embedding and transformer-based inference
FastAPI	RESTful API service layer
Gradio	Web-based user interaction
Vision

To bridge the knowledge gap between agricultural research and field application by leveraging AI and language models to support sustainable and data-driven farming.

Roadmap

Add multilingual translation and voice interface

Deploy on Hugging Face Spaces and Render

Integrate advanced retrieval pipelines using LangChain agents

Expand data corpus to include soil, climate, and pest management datasets

Launch browser-based extension for agronomists and researchers

Author

Nikhil Raman
Data Scientist | Data Analyst | AIML Prompt Engineer
Focused on integrating artificial intelligence into agriculture and sustainability through data analysis and prompt engineering.

GitHub: NikhilRaman12

Email: nikhilraman1203@gmail.com