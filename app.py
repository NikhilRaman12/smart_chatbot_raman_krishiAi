import gradio as gr
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Load FAISS index
INDEX_DIR = "data/faiss_index"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
retriever = FAISS.load_local(
    INDEX_DIR,
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
).as_retriever()

# Define chatbot logic
def raman_krishi_chatbot(query, language):
    results = retriever.invoke(query)
    top_results = [r.page_content for r in results[:3]] if results else ["No results found."]
    response = f"🌐 Language: {language}\n\n" + "\n\n".join(top_results)
    return response

# Define Gradio UI
interface = gr.Interface(
    fn=raman_krishi_chatbot,
    inputs=[
        gr.Textbox(label="🌾 Ask your question about agriculture", placeholder="e.g. What are chemicals used in paddy pest management?"),
        gr.Dropdown(
            choices=["English", "Hindi", "Tamizh", "Telugu", "Marathi", "Gujarati"],
            label="🌐 Select Language",
            value="English"
        )
    ],
    outputs=gr.Textbox(
        label="📘 RamanKrishiAI Response",
        lines=20,
        max_lines=40,
        interactive=False,
        show_copy_button=True,
        container=True
    ),
    title="🤖 RamanKrishiAI Chatbot",
    description="Ask me anything about sustainable agriculture. Powered by FAISS + HuggingFaceEmbeddings.",
    theme="soft",
    allow_flagging="never"
)

# Launch for Hugging Face Spaces
interface.launch()