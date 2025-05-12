import streamlit as st
import google.generativeai as genai
import speech_recognition as sr
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ✅ Set Page Config
st.set_page_config(page_title="Prompt Enhancer", layout="wide")

# 🔐 Gemini API Keys
API_KEYS = [
    "AIzaSyAYIIwHicFzIv2gYRUvk2pfEsnqVje9TfA",
    "AIzaSyAznPx4tiDhm5hnt1w1qQSoNjxEQgV4KUQ",
    "AIzaSyDmbj626LuMQwAJcmaJZwzdYfOdR_U96KI",
]

#  Load and cache Gemini Model (only once)
@st.cache_resource
def load_gemini_model():
    for key in API_KEYS:
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel('gemini-2.0-flash')
            model.generate_content("Test")  # Dummy check
            return model
        except Exception:
            continue
    return None

gemini_model = load_gemini_model()
if gemini_model is None:
    st.error("❌ All Gemini API keys are exhausted!")

#  Load and cache MT5 Model (only once)
@st.cache_resource
def load_mt5_model():
    tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")
    model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
    return tokenizer, model

mt5_tokenizer, mt5_model = load_mt5_model()

# 🚀 Generate AI Response using Gemini
def generate_ai_response(prompt):
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error: {e}"

# 🎙️ Speech Recognition
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("🎙 Listening... Speak now!")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "❌ Could not understand the audio."
    except sr.RequestError:
        return "❌ Could not request results. Check internet connection."

# 🌐 Sentiment Analysis
def analyze_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    return "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"

# 🌐 Word Cloud Generator
def generate_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)

# 🌐 UI: Title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Prompt Enhancer</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'><i>ADAPTIVE AI FOR REAL-TIME USER PROMPT CORRECTION IN CONVERSATIONAL SYSTEMS</i></h4>", unsafe_allow_html=True)

# 🧭 Tabs: Chat & Speech
tab1, tab2 = st.tabs(["💬 AI Chat", "🎤 Speech-to-Text & AI Reply"])

# 💬 Tab 1 - Text Chat
with tab1:
    st.subheader("💬 Chat with AI")
    user_prompt = st.text_area("Enter your prompt:")
    
    if st.button("Get AI Response", use_container_width=True):
        if user_prompt:
            ai_response = generate_ai_response(user_prompt)

            st.subheader("Sentiment of Your Input:")
            st.write(f"Sentiment: {analyze_sentiment(user_prompt)}")

            st.subheader("✨ AI Response:")
            st.write(ai_response)

            st.subheader("🧠 Word Cloud for Input:")
            generate_word_cloud(user_prompt)
        else:
            st.warning("⚠ Please enter a prompt before generating a response.")

# 🎤 Tab 2 - Speech-to-Text
with tab2:
    st.subheader("🎙 Speak and Get AI Response")
    if st.button("Speak & Get Response", use_container_width=True):
        speech_output = speech_to_text()
        if speech_output:
            st.subheader("🗣 Recognized Speech:")
            st.write(speech_output)

            ai_speech_response = generate_ai_response(speech_output)

            st.subheader("Sentiment of Your Speech:")
            st.write(f"Sentiment: {analyze_sentiment(speech_output)}")

            st.subheader("✨ AI Response:")
            st.write(ai_speech_response)
        else:
            st.warning("⚠ Please speak clearly for speech recognition.")
