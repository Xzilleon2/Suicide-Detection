%%writefile app.py
import streamlit as st
import pickle as pkl
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
import spacy

# =====================
# CONFIG
# =====================
THRESHOLD = 0.95
MIN_WORDS = 10  # Minimum words for meaningful prediction

# =====================
# CACHED MODEL LOADING
# =====================
@st.cache_resource
def load_resources():
    # Load spaCy model
    nlp_model = spacy.load("en_core_web_sm")
    # Load TF-IDF vectorizer
    tfidf_vect = pkl.load(open("tfidf_vectorizer.pkl", "rb"))
    # Load Naive Bayes model
    nb_model = pkl.load(open("suicide_nb_model.sav", "rb"))
    return nlp_model, tfidf_vect, nb_model

nlp, tfidf, nb_model = load_resources()

# Label encoder
encoder = LabelEncoder()
encoder.fit(['non-suicide', 'suicide'])

# =====================
# TEXT CLEANING + POS FILTER
# =====================
def clean_text(text_series):
    cleaned_text = []
    text_series = text_series.fillna("").astype(str)

    for sent in text_series:
        # Lowercase
        sent = sent.lower()

        # Preserve negations
        sent = sent.replace("don't", "do not").replace("dont", "do not")
        sent = sent.replace("can't", "can not").replace("cant", "can not")
        sent = sent.replace("won't", "will not").replace("wont", "will not")

        # Remove non-alphanumeric chars
        sent = re.sub(r'[^a-z0-9\s]', '', sent)
        sent = re.sub(r'\s+', ' ', sent).strip()

        # POS filtering: keep VERB, ADJ, ADV, NOUN
        doc = nlp(sent)
        tokens = [token.text for token in doc if token.pos_ in ['VERB', 'ADJ', 'ADV', 'NOUN']]
        cleaned_text.append(" ".join(tokens))

    return cleaned_text

# =====================
# PREDICTION FUNCTION
# =====================
def predict_suicide(text):
    # Minimum context check
    if len(text.split()) < MIN_WORDS:
        return None, None

    cleaned_input = clean_text(pd.Series([text]))
    vectorized_input = tfidf.transform(cleaned_input)

    # Probability of suicide class (class = 1)
    suicide_proba = nb_model.predict_proba(vectorized_input)[0][
        list(nb_model.classes_).index(1)
    ]

    # Apply threshold
    final_prediction = 1 if suicide_proba >= THRESHOLD else 0
    return final_prediction, suicide_proba

# =====================
# STREAMLIT APP
# =====================
st.title("Suicidal Risk In Text Prediction App")
st.write(
    "Enter a paragraph below to predict if it indicates suicide tendency. "
    "Please use English."
)

user_input = st.text_area("Input Text:", "")

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text for prediction.")
    else:
        prediction, confidence = predict_suicide(user_input)

        if prediction is None:
            st.warning(f"Please enter a paragraph with at least {MIN_WORDS} words for better prediction.")
        else:
            prediction_label = encoder.inverse_transform([prediction])[0]
            if prediction_label == 'suicide':
                st.error(
                    f"**Text contains SUICIDE tendency**\n\n"
                    f"Confidence: {confidence * 100:.2f}%"
                )
            else:
                st.success(
                    f"✅ **Text contains NON-SUICIDE tendency**\n\n"
                    f"Confidence: {confidence * 100:.2f}%"
                )
