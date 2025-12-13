# Suicide-Detection
Project Overview

This project focuses on detecting suicidal and non-suicidal text using supervised machine learning techniques. The system analyzes textual input and predicts whether the content indicates suicidal ideation. The goal of the model is to assist in identifying potentially high-risk text that may require further attention or intervention.

The application processes user-provided text, applies text preprocessing and feature extraction, and then classifies the text using trained machine learning models. The project is implemented in Python and deployed using a Streamlit web interface for easy interaction.

Machine Learning Models Used

This project uses supervised learning models trained on labeled text data:

Support Vector Machine (SVM)
The SVM model is used for its effectiveness in high-dimensional text data and its ability to find optimal decision boundaries between suicidal and non-suicidal classes.

Logistic Regression
Logistic Regression is employed as a baseline linear classifier that provides interpretable results and performs well on text classification tasks when combined with TF-IDF features.

Both models are trained using TF-IDF (Term Frequency–Inverse Document Frequency) vectorization to convert textual data into numerical feature representations.

What the Model Does

Accepts raw text input (paragraphs or comments)

Cleans and preprocesses the text (lowercasing, removing special characters, handling missing values)

Converts text into numerical features using TF-IDF

Predicts whether the text is suicidal or non-suicidal based on learned patterns

Selects the most confident prediction among the trained models based on performance metrics

Model Limitations

While the model performs well on keyword-based and surface-level patterns, it has several limitations:

The model does not understand deeper contextual or semantic meaning of text.

Predictions are largely influenced by the presence or absence of suicidal keywords and phrases rather than true emotional intent.

The model may struggle with sarcasm, metaphors, implicit expressions, or complex emotional language.

It should not be used as a standalone diagnostic tool, but rather as a preliminary screening or research-based system.

Future improvements could include the use of contextual embeddings or transformer-based models to better capture semantic meaning and emotional nuance.
