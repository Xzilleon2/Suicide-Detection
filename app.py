import streamlit as st
import pickle as pkl
import pandas as pd
import re
import numpy as np
from sklearn.metrics import f1_score

# Load the saved TFIDF vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as file:
    tfidf = pkl.load(file)

# Load the saved SVM model
with open('suicide_svm_model.sav', 'rb') as file:
    svm = pkl.load(file)

# Load the saved Logistic Regression model
with open('suicide_lr_model.sav', 'rb') as file:
    log_reg = pkl.load(file)

# Clean text function
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

# Hardcoded F1 scores
f1_svm = 0.9029126213592233
f1_lr = 0.9008130081300812

# Prediction function
def predict_best_model(text):
    # Enforce minimum word count
    if len(text.split()) < 3:
        return None, None, None  # Not enough context
    
    cleaned_input = clean_text(pd.Series([text]))
    vectorized_input = tfidf.transform(cleaned_input)

    preds = {}

    # SVM
    pred_svm = svm.predict(vectorized_input)[0]
    decision_svm = svm.decision_function(vectorized_input)
    conf_svm = 1 / (1 + np.exp(-decision_svm[0]))  # pseudo-probability
    preds['SVM'] = (pred_svm, conf_svm * f1_svm)

    # Logistic Regression
    pred_lr = log_reg.predict(vectorized_input)[0]
    conf_lr = log_reg.predict_proba(vectorized_input)[0][log_reg.classes_.tolist().index('suicide')]
    preds['Logistic Regression'] = (pred_lr, conf_lr * f1_lr)

    # Select the model with highest weighted score
    best_model_name = max(preds, key=lambda m: preds[m][1])
    best_prediction = preds[best_model_name][0]

    return best_prediction, best_model_name, preds

# Streamlit App
st.title("Suicidal Risk In Text Prediction App")
st.write("Enter text below to predict if it indicates suicide risk (suicide/non-suicide). Please use English.")

user_input = st.text_area("Input Text:", "")

if st.button("Predict"):
    if user_input:
        prediction, model_name, all_preds = predict_best_model(user_input)

        if prediction is None:
            st.warning("Please enter a sentence or paragraph with at least 3 words for meaningful prediction.")
        else:
            st.subheader("Prediction Result:")
            if prediction == 'suicide':
                st.error(f"The text may indicate **{prediction.upper()}** risk based on **{model_name}**.")
            else:
                st.success(f"The text indicates **{prediction.upper()}** risk based on **{model_name}**.")

            st.subheader("Detailed Model Confidences (Weighted by F1 Score):")
            for m, (pred, prob) in all_preds.items():
                st.write(f"- **{m}**: Predicted '{pred}', Weighted Confidence: {prob:.4f}")

    else:
        st.warning("Please enter some text for prediction.")
