import streamlit as st
import joblib

# Load model
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("📰 Fake News Detector")

st.write("Enter a news article or headline below.")

news = st.text_area("News Text")

if st.button("Check News"):

    if news.strip():

        news_vector = vectorizer.transform([news])

        prediction = model.predict(news_vector)[0]
        probability = model.predict_proba(news_vector)[0]

        confidence = max(probability) * 100

        if confidence < 70:
            st.warning("⚠️ Low Confidence Prediction")

        if prediction == 1:
            st.success("✅ Real News")
            st.write(f"Confidence: {confidence:.2f}%")
        else:
            st.error("❌ Fake News")
            st.write(f"Confidence: {confidence:.2f}%")

    else:
        st.warning("Please enter some text.")