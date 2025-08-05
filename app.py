import pickle
import streamlit as st
import numpy as np
from scipy import sparse
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import os

if "history" not in st.session_state:
    st.session_state.history = []

# Ensure necessary NLTK data is available (quiet)
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

ps = PorterStemmer()
#Data preprocessing 
def transformText(text):
    if text is None:
        return ""

    text = str(text).lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)

#loading pickles
model = pickle.load(open('MultinomialNB.pkl', 'rb'))
vectorizer = pickle.load(open('Vectorizer.pkl', 'rb'))

#Title and Description:
st.set_page_config(page_title='Spam Detector', layout='wide')
st.title('SPAAAAAAAM')
st.markdown('Check if any text is Spam or not(ham)')

#Sidebar:
with st.sidebar:
    st.header('How it works')
    st.write("""
    - type or paste a message
    - Click check
    - See predicted label and confidence
    """)

st.subheader('Enter a text to check')
text = st.text_area('Message', height=150, placeholder='write or paste to check for spam')

col1, col2 = st.columns([1,3])

with col1:
    check_btn = st.button('check')
with col2:
    st.write('')


if check_btn:
    if not text or not text.strip():
        st.warning('Please enter some text to check.')
    else:
        # Preprocess, transform and predict
        text_cleaned = transformText(text)
        X = vectorizer.transform([text_cleaned])   # ensure one-row input
        pred = model.predict(X)[0]

        # Display result
        if int(pred) == 1:
            st.error("🚨 **SPAM** 🚨")
        else:
            st.success("✅ **Not Spam (Ham)**")

        # Save to history only once (after a successful check)
        st.session_state.history.insert(0, {'text': text, 'label': 'Spam' if int(pred) == 1 else 'Not Spam'})
        st.session_state.history = st.session_state.history[:5]

# Recent Checks (only displays; does not re-run prediction)
if st.session_state.history:
    st.subheader('Recent Checks')
    for item in st.session_state.history:
        st.write(item["text"])