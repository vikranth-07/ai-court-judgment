import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5ForConditionalGeneration
from lime.lime_text import LimeTextExplainer


# LOAD MODELS (FIXED NAMES)

@st.cache_resource
def load_models():
    bert_path = "mandavasaivikranth/legal_bert_model"   
    t5_path = "mandavasaivikranth/t5_summarization_model" 

    bert_tokenizer = AutoTokenizer.from_pretrained(bert_path)
    bert_model = AutoModelForSequenceClassification.from_pretrained(bert_path)

    t5_tokenizer = AutoTokenizer.from_pretrained(t5_path)
    t5_model = T5ForConditionalGeneration.from_pretrained(t5_path)

    bert_model.eval()
    t5_model.eval()

    return bert_tokenizer, bert_model, t5_tokenizer, t5_model

bert_tokenizer, bert_model, t5_tokenizer, t5_model = load_models()

explainer = LimeTextExplainer(class_names=['Rejected', 'Accepted'])


# DEVICE SETUP (IMPORTANT)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)
t5_model.to(device)


# LIME PREDICTOR

def predictor(texts):
    inputs = bert_tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = bert_model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    return probs.cpu().numpy()


# UI

st.title("⚖️ AI Court Judgment Simplifier")

text = st.text_area("Enter Judgment Text")

if st.button("Analyze"):

    if text.strip() == "":
        st.warning("Please enter some text")
        st.stop()


    # CLASSIFICATION

    inputs = bert_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=64
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = bert_model(**inputs)

    pred = torch.argmax(outputs.logits).item()
    label = "Accepted" if pred == 1 else "Rejected"


    # SUMMARIZATION

    t5_input = "summarize: " + text
    t5_inputs = t5_tokenizer.encode(
        t5_input,
        return_tensors="pt",
        truncation=True,
        max_length=256
    ).to(device)

    summary_ids = t5_model.generate(
        t5_inputs,
        max_length=80,
        num_beams=4,
        early_stopping=True
    )

    summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)


    # LIME (OPTIMIZED)

    exp = explainer.explain_instance(
        text,
        predictor,
        num_features=5,
        num_samples=200  
    )

    # OUTPUT

    st.subheader("📌 Classification")
    st.success(label)

    st.subheader("📝 Summary")
    st.write(summary)

    st.subheader("🔍 Explanation")
    for word, score in exp.as_list():
        st.write(f"{word}: {round(score, 3)}")
