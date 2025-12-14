import streamlit as st
import pickle as pkl
import pandas as pd
import re
from tqdm import tqdm
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

# Re-define the clean_text function (ensure it's identical to training)
def clean_text(text_series):
    cleaned_text = []
    # Ensure all values are strings
    text_series = text_series.fillna("").astype(str)
    # tqdm is not suitable for Streamlit real-time updates, so remove or handle differently
    for sent in text_series:
        sent = sent.lower()
        sent = re.sub(r'[^a-z0-9\s]', '', sent)
        sent = re.sub(r'\s+', ' ', sent).strip()
        cleaned_text.append(sent)
    return cleaned_text

# Assuming f1_svm and f1_lr are globally available or calculated in the app
# For simplicity, we'll hardcode them here based on previous output or re-calculate if needed.
# In a real deployment, these might be saved alongside the models.
# For now, let's use the values obtained from the notebook's last run.
f1_svm = 0.9029126213592233
f1_lr = 0.9008130081300812

# Re-define the predict_best_model function
def predict_best_model(text):
    cleaned_input = clean_text(pd.Series([text]))
    vectorized_input = tfidf.transform(cleaned_input)

    preds = {}

    # SVM
    pred_svm = svm.predict(vectorized_input)[0]
    decision_svm = svm.decision_function(vectorized_input)
    conf_svm = 1 / (1 + np.exp(-decision_svm[0]))  # pseudo-probability
    preds['SVM'] = (pred_svm, conf_svm * f1_svm)  # weighted by F1

    # Logistic Regression
    pred_lr = log_reg.predict(vectorized_input)[0]
    # Using predict_proba for LR confidence
    conf_lr = log_reg.predict_proba(vectorized_input)[0][log_reg.classes_.tolist().index('suicide')] # Probability of 'suicide' class
    preds['Logistic Regression'] = (pred_lr, conf_lr * f1_lr)  # weighted by F1

    # Select the model with highest weighted score
    best_model_name = max(preds, key=lambda m: preds[m][1])
    best_prediction = preds[best_model_name][0]

    return best_prediction, best_model_name, preds

# Streamlit App
st.title("Suicidal Risk In Text Prediction App")
st.write("Enter text below to predict if it indicates suicide risk (suicide/non-suicide). Please use english as language.")

user_input = st.text_area("Input Text:", "")

if st.button("Predict"):
    if user_input:
        prediction, model_name, all_preds = predict_best_model(user_input)

        st.subheader("Prediction Result:")
        if prediction == 'suicide':
            st.error(f"The text may indicate **{prediction.upper()}** risk based on **{model_name}**.")
        else:
            st.success(f"The text indicate **{prediction.upper()}** risk based on **{model_name}**.")

        st.subheader("Detailed Model Confidences (Weighted by F1 Score):")
        for m, (pred, prob) in all_preds.items():
            st.write(f"- **{m}**: Predicted '{pred}', Weighted Confidence: {prob:.4f}")

    else:
        st.warning("Please enter some text for prediction.")
