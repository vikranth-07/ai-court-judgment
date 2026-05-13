import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5ForConditionalGeneration
from lime.lime_text import LimeTextExplainer

# Page config
st.set_page_config(page_title="AI Court Judgment Simplifier", layout="wide")

# Custom CSS (background + styling)
st.markdown("""
    <style>
    .main {
        background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
        color: white;
    }
    textarea, .stTextArea textarea {
        background-color: #f0f0f0 !important;
        color: black !important;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border-radius: 10px;
        padding: 10px 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5ForConditionalGeneration

    bert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    bert_model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2
    )

    t5_tokenizer = AutoTokenizer.from_pretrained("t5-small")
    t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")

    return bert_tokenizer, bert_model, t5_tokenizer, t5_model

bert_tokenizer, bert_model, t5_tokenizer, t5_model = load_models()

explainer = LimeTextExplainer(class_names=['Rejected', 'Accepted'])

# Header
st.title("⚖️ AI Court Judgment Simplifier")
st.markdown("### 🔍 Analyze, Summarize & Explain Legal Judgments")

# Input
text = st.text_area("📄 Enter Judgment Text", height=200)

# Predictor for LIME
def predictor(texts):
    inputs = bert_tokenizer(list(texts), return_tensors="pt", padding=True, truncation=True)
    outputs = bert_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    return probs.detach().numpy()

# Button
if st.button("🚀 Analyze Judgment"):
    if text.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        with st.spinner("Processing..."):

            # Classification
            inputs = bert_tokenizer(text, return_tensors="pt", truncation=True)
            outputs = bert_model(**inputs)
            pred = torch.argmax(outputs.logits).item()
            label = "✅ Accepted" if pred == 1 else "❌ Rejected"

            # Summary
            t5_input = "summarize: " + text
            t5_inputs = t5_tokenizer.encode(t5_input, return_tensors="pt", truncation=True)
            summary_ids = t5_model.generate(t5_inputs, max_length=100)
            summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            # LIME
            exp = explainer.explain_instance(text, predictor, num_features=5)

        # Output layout
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📌 Classification")
            st.success(label)

            st.subheader("📝 Summary")
            st.info(summary)

        with col2:
            st.subheader("🔍 Explanation (LIME)")
            for word, score in exp.as_list():
                st.write(f"**{word}** : {round(score, 3)}")

# Footer (YOUR NAME)
st.markdown("---")
st.markdown(
    "<center><h4>👨‍💻Developed by Mandava Sai Vikranth Goud</h4></center>",
    unsafe_allow_html=True
)
