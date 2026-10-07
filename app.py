import streamlit as st
import pickle
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# Initialize
ps = PorterStemmer()
stop_words = set(stopwords.words("english"))


# Text preprocessing
def transform_text(text):
    text = text.lower()

    # Tokenization
    text = nltk.word_tokenize(text)

    # Remove punctuation and non-alphanumeric tokens
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    # Remove stopwords
    text = y[:]
    y.clear()

    for i in text:
        if i not in stop_words:
            y.append(i)

    # Stemming
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# Load trained model and vectorizer
with open("vectorizer.pkl", "rb") as file:
    tfidf = pickle.load(file)

with open("model.pkl", "rb") as file:
    model = pickle.load(file)


# Streamlit UI
st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")


if st.button("Predict"):

    if input_sms.strip() == "":
        st.warning("Please enter a message.")

    else:

        # Preprocess
        transformed_sms = transform_text(input_sms)

        # Vectorize
        vector_input = tfidf.transform([transformed_sms])

        # Predict
        result = model.predict(vector_input)[0]

        # Display result
        if result == 1:
            st.error("Spam")
        else:
            st.success("Not Spam")