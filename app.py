import streamlit as st
import pickle as pkl
import pandas as pd
import re
import numpy as np
from sklearn.preprocessing import LabelEncoder

# =====================
# CONFIG
# =====================
THRESHOLD = 0.95

# =====================
# LOAD MODELS
# =====================
with open('tfidf_vectorizer.pkl', 'rb') as file:
    tfidf = pkl.load(file)

with open('suicide_nb_model.sav', 'rb') as file:
    nb_model = pkl.load(file)

# Label encoder
encoder = LabelEncoder()
encoder.fit(['non-suicide', 'suicide'])

# =====================
# TEXT CLEANING
# =====================
def clean_text(text_series):
    cleaned_text = []
    text_series = text_series.fillna("").astype(str)

    for sent in text_series:
        sent = sent.lower()
        sent = sent.replace("don't", "do not").replace("dont", "do not")
        sent = sent.replace("can't", "can not").replace("cant", "can not")
        sent = sent.replace("won't", "will not").replace("wont", "will not")
        sent = re.sub(r'[^a-z0-9\s]', '', sent)
        sent = re.sub(r'\s+', ' ', sent).strip()
        cleaned_text.append(sent)

    return cleaned_text

# =====================
# PREDICTION FUNCTION
# =====================
def predict_suicide(text):
    # Minimum context check
    if len(text.split()) < 3:
        return None, None

    cleaned_input = clean_text(pd.Series([text]))
    vectorized_input = tfidf.transform(cleaned_input)

    # Probability of suicide class (class = 1)
    suicide_proba = nb_model.predict_proba(vectorized_input)[0][
        list(nb_model.classes_).index(1)
    ]

    # Apply threshold logic
    if suicide_proba >= THRESHOLD:
        final_prediction = 1  # suicide
    else:
        final_prediction = 0  # non-suicide

    return final_prediction, suicide_proba

# =====================
# STREAMLIT APP
# =====================
st.title("Suicidal Risk In Text Prediction App")
st.write(
    "Enter text below to predict if it indicates suicide tendency "
)

user_input = st.text_area("Input Text:", "")

if st.button("Predict"):
    if user_input:
        prediction, confidence = predict_suicide(user_input)

        if prediction is None:
            st.warning(
                "Please enter a sentence or paragraph with at least 3 words."
            )
        else:
            prediction_label = encoder.inverse_transform([prediction])[0]

            if prediction_label == 'suicide':
                st.error(
                    f"**text contains SUICIDE tendency**\n\n"
                    f"Confidence: {confidence * 100:.2f}%"
                )
            else:
                st.success(
                    f"✅ **text contains NON-SUICIDE tendency**\n\n"
                    f"Confidence: {confidence * 100:.2f}%"
                )
    else:
        st.warning("Please enter some text for prediction.")
