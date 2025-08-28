import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt 
from deep_translator import GoogleTranslator
from fpdf import FPDF
import base64
import tempfile

st.set_page_config(page_title="MedEdu Chatbot", layout="centered")
st.title("Medical Education Analysis Chatbot – PoC on 28-07-2025")

# Load data
@st.cache_data
def load_data():
    df_colleges = pd.read_csv("data/colleges.csv", encoding="ISO-8859-1")
    df_consolidated = pd.read_csv("data/Consolidated.csv", encoding="ISO-8859-1")
    df_medical = pd.read_csv("data/Medical data.csv", encoding="ISO-8859-1")
    return {"Colleges": df_colleges, "Consolidated": df_consolidated, "MedicalData": df_medical}

datasets = load_data()
df_consolidated = datasets["Consolidated"]

# Chat state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# UI
lang = st.selectbox("Language", ["English", "Hindi", "Marathi"])
query = st.text_input("Ask your question about the data")

# Helper: Translate to English
def translate_to_english(text, lang_code):
    try:
        return GoogleTranslator(source=lang_code, target="en").translate(text)
    except:
        return text

# Helper: Forecast seats over time
def forecast_seats_trend(course, district):
    df = df_consolidated.copy()
    df = df[(df["Course"].str.lower() == course.lower()) & (df["District"].str.lower() == district.lower())]
    if df.empty:
        return "No relevant data found."

    df_grouped = df.groupby("Year").agg({"Capacity": "sum"}).sort_index()
    df_grouped.index = pd.to_datetime(df_grouped.index.str[:4])
    
    # Fit model
    try:
        model = ExponentialSmoothing(df_grouped["Capacity"], trend="add", seasonal=None).fit()
        future_years = pd.date_range(start="2025", periods=3, freq="Y")
        forecast = model.forecast(len(future_years))
        full_series = pd.concat([df_grouped["Capacity"], forecast])
    except:
        return "Not enough data for forecasting."

    # Plot
    fig, ax = plt.subplots()
    full_series.plot(ax=ax, marker="o")
    ax.set_title(f"Seat Capacity Trend & Forecast for {course.upper()} in {district.title()}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Seats")
    st.pyplot(fig)
    return None

# Answer Query
def answer_query(query):
    q = query.lower()

    if "trend" in q or "forecast" in q:
        # Extract course and district
        if "anm" in q:
            course = "ANM"
        elif "gnm" in q:
            course = "GNM"
        else:
            return "Please specify a course (ANM or GNM)."

        for district in df_consolidated["District"].unique():
            if district.lower() in q:
                return forecast_seats_trend(course, district)
        return "Please specify a valid district."

    # Keyword Matching
    matched_df = df_consolidated.copy()
    for course in ["ANM", "GNM"]:
        if course.lower() in q:
            matched_df = matched_df[matched_df["Course"].str.lower() == course.lower()]
            break

    for district in df_consolidated["District"].str.lower().unique():
        if district in q:
            matched_df = matched_df[matched_df["District"].str.lower() == district]

    for division in df_consolidated["Division"].str.lower().unique():
        if division in q:
            matched_df = matched_df[matched_df["Division"].str.lower() == division]

    for year in df_consolidated["Year"].unique():
        if year in q:
            matched_df = matched_df[matched_df["Year"] == year]

    if matched_df.empty:
        return "No relevant data found."

    total_capacity = matched_df["Capacity"].sum()
    total_enrolled = matched_df["Enrolled"].sum()
    gap = total_capacity - total_enrolled
    percent_gap = (gap / total_capacity * 100) if total_capacity else 0

    response = f"""
Total Capacity: {total_capacity}  
Total Enrolled: {total_enrolled}  
Unfilled Seats: {gap}  
Percentage Unfilled: {percent_gap:.2f}%  
Institutes Count: {matched_df.shape[0]}  
"""
    return response

# Translate
if lang != "English" and query:
    query = translate_to_english(query, lang.lower())

# Process query
if query:
    with st.spinner("Processing..."):
        answer = answer_query(query)
        if answer:
            st.success("Answer:")
            if isinstance(answer, str):
                st.markdown(answer)
            st.session_state.chat_history.append((query, answer if isinstance(answer, str) else "[Graph]"))

# History
if st.sidebar.checkbox("Show Chat History"):
    for i, (q, a) in enumerate(st.session_state.chat_history[::-1], 1):
        st.sidebar.markdown(f"**Q{i}:** {q}")
        st.sidebar.markdown(f"**A{i}:** {a[:200] if isinstance(a, str) else '[Graph]'}")

# Export to PDF
if st.sidebar.button("Export as PDF"):
    if st.session_state.chat_history:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        for i, (q, a) in enumerate(st.session_state.chat_history, 1):
            pdf.multi_cell(0, 10, f"Q{i}: {q}\nA{i}: {a if isinstance(a, str) else '[Graph]'}\n")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            pdf.output(tmpfile.name)
            with open(tmpfile.name, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
                st.sidebar.markdown(f'<a href="data:application/octet-stream;base64,{b64}" download="chat_history.pdf">Download PDF</a>', unsafe_allow_html=True)
    else:
        st.sidebar.warning("No chat to export.")

# Export to CSV
if st.sidebar.button("Export as CSV"):
    if st.session_state.chat_history:
        df_chat = pd.DataFrame(st.session_state.chat_history, columns=["Question", "Answer"])
        csv = df_chat.to_csv(index=False).encode("utf-8")
        st.sidebar.download_button("Download CSV", csv, "chat_history.csv", "text/csv")
    else:
        st.sidebar.warning("No chat to export.")
