import streamlit as st
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "facebook/bart-base"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name).to(DEVICE)
model.load_state_dict(torch.load("best_summary_model.pth", map_location=DEVICE))
model.eval()

st.title("Abstractive Text Summarizer")
input_text = st.text_area("Enter the article text below:")

if st.button("Summarize") and input_text.strip():
    inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    st.subheader("Summary:")
    st.write(summary)
