import streamlit as st
import pandas as pd
from langchain.llms import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access the API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI
llm = OpenAI(temperature=0.5, openai_api_key=OPENAI_API_KEY)

# Streamlit UI
st.title("AI-Powered Inventory Optimization")
st.header("Inventory Dashboard")

# Session state to hold uploaded data
if 'data' not in st.session_state:
    st.session_state.data = None

uploaded_file = st.file_uploader("Upload CSV with inventory data", type="csv")
if uploaded_file is not None:
    try:
        st.session_state.data = pd.read_csv(uploaded_file)
        st.write("Data loaded:")
        st.write(st.session_state.data)
    except Exception as e:
        st.error(f"Error loading file: {e}")


# Forecasting and Recommendations
st.header("Forecasting and Recommendations")
if st.button("Get Recommendations"):
    if st.session_state.data is None:
        st.warning("Please upload inventory data before getting recommendations.")
    else:
        prompt = "Based on the following inventory data, provide restocking recommendations:\n" + st.session_state.data.to_string(index=False)
        recommendations = llm(prompt)
        st.write(recommendations)
