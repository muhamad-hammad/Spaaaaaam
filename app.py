import pickle
import streamlit as st
import numpy as np

#loading pickles
model = pickle.load(open('MultinomialNB.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))
vectorizer = pickle.load(open('Vectorizer.pkl', 'rb'))