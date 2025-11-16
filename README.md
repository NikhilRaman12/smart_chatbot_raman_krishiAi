RamanKrishiAI — Smart-Agri Chatbot



An AI-powered multilingual chatbot designed to democratize agricultural expertise for farmers, agronomists, and enterprise stakeholders.

RamanKrishiAI integrates machine learning and natural language processing to deliver real-time insights on sustainable farming practices, crop management, and agronomic recommendations.



Objective



The objective of RamanKrishiAI is to make agricultural knowledge universally accessible through artificial intelligence and domain expertise.

While its primary focus is agriculture, the system is capable of answering questions across any domain using retrieval-augmented generation (RAG), semantic embeddings, and multilingual support. This makes it a versatile AI assistant for users from diverse fields seeking contextually relevant, evidence-based responses in real time.



Live Demo



Hugging Face Spaces: https://huggingface.co/spaces/NikhilRaman12/smart\_chatbot\_raman\_krishiAi



Tech Stack

Component	Purpose

FastAPI	Backend framework for API deployment

LangChain	Orchestrates the retrieval-augmented generation (RAG) logic

FAISS	Vector search for semantic similarity matching

Hugging Face Transformers	Generates embeddings using sentence-transformers/all-MiniLM-L6-v2

Gradio	Lightweight user interface for chatbot interaction

Python-dotenv	Environment variable management for API keys and configuration



Multilingual Support



RamanKrishiAI supports user interaction in all languages, including but not limited to:



English



Hindi



Telugu



Tamil



Marathi



Gujarati



Future versions will include translation pipelines and speech-based interfaces to improve accessibility.



Project Structure

smart\_chatbot\_raman\_krishiAi/

├── src/

│   ├── main.py              # FastAPI application entry point

│   ├── embeddings/          # FAISS embedding and vector storage

│   ├── utils/               # Helper modules

│   ├── requirements.txt     # Project dependencies

│   └── runtime.txt          # Python runtime for Render deployment

├── .gitignore

├── README.md

└── data/

&nbsp;   └── faiss\_index/

&nbsp;       ├── index.faiss

&nbsp;       └── index.pkl



Local Setup and Testing

\# Clone the repository

git clone https://github.com/NikhilRaman12/smart\_chatbot\_raman\_krishiAi.git

cd smart\_chatbot\_raman\_krishiAi



\# Create and activate virtual environment

python -m venv venv311

.\\venv311\\Scripts\\Activate.ps1  # For PowerShell



\# Install dependencies

pip install -r src/requirements.txt



\# Run the application

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



General-purpose domain support for education, research, sustainability, and enterprise knowledge systems



Integrations

Tool	Functionality

LangChain + FAISS	Context retrieval and semantic search

Hugging Face	Embedding and transformer-based inference

FastAPI	RESTful API service layer

Gradio	Web-based user interaction



Vision



To bridge the knowledge gap between agricultural research and field application by leveraging AI and language models to support sustainable and data-driven farming.

To extend intelligent assistance across domains through scalable, multilingual, and context-aware AI systems.



Roadmap



Add multilingual translation and voice interface



Deploy on Hugging Face Spaces



Integrate advanced retrieval pipelines using LangChain agents



Expand data corpus to include soil, climate, and pest management datasets



Launch browser-based extension for agronomists and researchers



Author



Nikhil Raman

Data Scientist | Data Analyst | AIML Prompt Engineer

Focused on integrating artificial intelligence into agriculture and sustainability through data analysis and prompt engineering.



GitHub: NikhilRaman12



Email: nikhilraman1203@gmail.com    https://github.com/NikhilRaman12/smart\_chatbot\_raman\_krishiAi

