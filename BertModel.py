import streamlit as st
import openai
import PyPDF2
from transformers import BertTokenizer, BertForQuestionAnswering
import torch

import os

os.environ["OPENAI_API_KEY"] = 'sk-proj-sytG2gNJxunAeXMiXSw9T3BlbkFJd4O2ioEs12RnmSlnTXxq'

pdf_path = "Eng_doc.pdf"


def extract_text_from_pdf(ppath):
    text = ""
    with open(ppath, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text


# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
#model = "gpt-3.5-turbo"
# Example question and passage
st.subheader("Type your text here ... ")
question = st.text_input("Type the text here : ")

if st.button("Search"):


    # Tokenize input question and passage
    passage = extract_text_from_pdf(pdf_path)
    inputs = tokenizer(question, passage, return_tensors='pt', max_length=512, truncation=True)
    # Perform inference
    #start_scores, end_scores = model(**inputs)
    outputs = model(**inputs)

    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    # Convert logits to probabilities
    #start_probs = torch.softmax(start_scores, dim=1, dtype=None)
    #end_probs = torch.softmax(end_scores, dim=1, dtype=None)

    # Find the answer span with the highest probability
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)

    # Decode answer tokens
    answer_tokens = inputs['input_ids'][0][start_index:end_index + 1]
    answer = tokenizer.decode(answer_tokens)

    print("Predicted Answer:", st.write(answer))
