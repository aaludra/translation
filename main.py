import PyPDF2
import streamlit as st
import spacy
import en_core_web_sm


def pdf_to_txt(p_file):
    text = ""
    with open(p_file, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text


def question_to_answer(conv_text, question):
    st.spinner("Searching...")
    doc = nlp(conv_text)
    answers = []
    #        # Process question
    question_doc = nlp(question)
    # Find answer by matching question tokens with document tokens
    answer = ""
    max_similarity = 0
    for sent in doc.sents:
        similarity = question_doc.similarity(sent)
        if similarity > max_similarity:
            max_similarity = similarity
            answer = sent.text
    answers.append(answer)
    return answers


st.title("Query")
st.write()
txt_file = ""
option = st.selectbox(
    "Choose the language...",
    ("English", "Dutch", "German"))
if option == "English":
    user_input = st.text_input(label="Enter the query here...")
    nlp = en_core_web_sm.load()
    st.write()
    st.button("Search")
    st.write()
    st.spinner("Search")
    txt_file = pdf_to_txt("Eng_Doc.pdf")
elif option == "Dutch":
    user_input = st.text_input(label="Voeg hier de vraag in...")
    nlp = spacy.load("model\nl_core_news_sm-3.7.0-py3-none-any.whl")
    st.write()
    st.button("Zoekopdracht")
    txt_file = pdf_to_txt("Dutch_Doc.pdf")
    st.write()
    st.spinner("Zoeken...")
else:
    user_input = st.text_input(label="Geben Sie hier die Abfrage ein...")
    nlp = spacy.load("model\nl_core_news_sm-3.7.0-py3-none-any.whl")
    st.write()
    st.button("Suchen")
    txt_file = pdf_to_txt("German_doc.pdf")
    st.write()
    st.spinner("Suche...")

response = question_to_answer(txt_file, user_input)
st.write(response[0])