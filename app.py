import gradio as gr
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import re

# Load FAISS index
INDEX_DIR = "data/faiss_index"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
retriever = FAISS.load_local(
    INDEX_DIR,
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
).as_retriever()

# Load semantic reranker
semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

# Language-specific translation models
translation_models = {
    "Hindi": "Helsinki-NLP/opus-mt-hi-en",
    "Tamizh": "Helsinki-NLP/opus-mt-ta-en",
    "Telugu": "Helsinki-NLP/opus-mt-te-en",
    "Marathi": "Helsinki-NLP/opus-mt-mr-en",
    "Gujarati": "Helsinki-NLP/opus-mt-gu-en"
}

# Translate non-English input
def translate_to_english(text, language):
    if language != "English":
        model_name = translation_models.get(language)
        if model_name:
            try:
                translator = pipeline("translation", model=model_name)
                translated = translator(text)[0]["translation_text"]
                print(f"🔁 Translated from {language}: {translated}")
                return translated
            except Exception as e:
                print(f"⚠️ Translation failed for {language}: {e}")
                return text
    return text

# Detect definition-style queries
def is_definition_query(query):
    return query.lower().startswith(("what is", "define", "explain", "describe"))

# Expanded fallback definitions
def generate_definition(query):
    keyword = query.lower().replace("what is", "").replace("define", "").replace("explain", "").replace("describe", "").strip()
    definitions = {
        "biopesticides": "Biopesticides are natural pest control agents derived from microorganisms, plants, or minerals.",
        "chemical pesticides": "Chemical pesticides are synthetic compounds used to kill or repel pests.",
        "agronomy": "Agronomy is the science of soil management and crop production.",
        "soil science": "Soil science studies soil formation, classification, mapping, and fertility.",
        "crop rotation": "Crop rotation is the practice of growing different crops in succession to improve soil health.",
        "ipm": "Integrated Pest Management (IPM) combines biological, cultural, mechanical, and chemical methods.",
        "natural farming": "Natural farming avoids synthetic inputs and relies on biodiversity and organic matter.",
        "trichoderma": "Trichoderma is a genus of fungi used as a biocontrol agent against soil-borne pathogens.",
        "fertigation": "Fertigation is the application of fertilizers through irrigation systems.",
        "mulching": "Mulching involves covering soil to retain moisture and suppress weeds.",
        "vermicompost": "Vermicompost is organic fertilizer produced by earthworms.",
        "pest management": "Pest management includes monitoring, prevention, and control strategies.",
        "disease management": "Disease management involves identifying symptoms and applying treatments.",
        "seed treatment": "Seed treatment involves coating seeds to protect against pests and diseases.",
        "intercropping": "Intercropping is growing multiple crops together to optimize space and reduce pests.",
        "organic farming": "Organic farming emphasizes natural inputs and ecological balance.",
        "precision agriculture": "Precision agriculture uses sensors and data to optimize farming decisions.",
        "climate-smart agriculture": "Climate-smart agriculture enhances resilience to climate change.",
        "agroforestry": "Agroforestry integrates trees into crop and livestock systems.",
        "rainfall patterns": "Rainfall patterns refer to seasonal distribution and intensity of precipitation.",
        "kharif season": "Kharif season crops are sown with the onset of monsoon and harvested in autumn.",
        "rabi season": "Rabi season crops are sown in winter and harvested in spring.",
        "apiculture": "Apiculture is the practice of beekeeping for honey and pollination.",
        "sericulture": "Sericulture is the cultivation of silkworms for silk production.",
        "allied enterprises": "Allied enterprises include dairy, poultry, fisheries, and other farm-linked activities."
    }
    return definitions.get(keyword, "Definition not found. Please rephrase or ask a more specific question.")

# Semantic reranking
def rerank_results(query, results):
    query_embedding = semantic_model.encode(query, convert_to_tensor=True)
    scored = [(r, util.cos_sim(query_embedding, semantic_model.encode(r.page_content, convert_to_tensor=True)).item()) for r in results]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [r[0] for r in scored[:3]]

# Detect stat-heavy queries
def is_stats_query(query):
    return any(word in query.lower() for word in ["production", "yield", "consumption", "export", "area", "gdp", "statistics", "data"])

# Summarize quantitative stats
def summarize_stats(text):
    stats = re.findall(r"\d+\.?\d*\s*(MMT|kg|%|million|lakh|tons?)", text)
    return "📊 Key Stats:\n" + "\n".join(stats) if stats else text

# Main chatbot logic
def raman_krishi_chatbot(query, language):
    query = translate_to_english(query, language)

    if is_definition_query(query):
        response = generate_definition(query)
    else:
        results = retriever.invoke(query)
        if results:
            reranked = rerank_results(query, results)
            top_results = [r.page_content for r in reranked]
            response = "\n\n".join(top_results)
        else:
            response = generate_definition(query)

    if is_stats_query(query):
        response = summarize_stats(response)

    return f"🌐 Language: {language}\n\n{response}"

# Gradio UI
interface = gr.Interface(
    fn=raman_krishi_chatbot,
    inputs=[
        gr.Textbox(
            label="🌾 Ask your question about agriculture",
            placeholder="e.g. What are chemicals used in paddy pest management?"
        ),
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
    description="Ask me anything about sustainable agriculture. Powered by FAISS + HuggingFaceEmbeddings + Semantic Reranking + Multilingual Translation + Authentic Definitions.",
    theme="soft",
    flagging_mode="never"
)

# Launch for Hugging Face Spaces
interface.launch()