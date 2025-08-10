import streamlit as st
import os
import sys
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
from langchain.schema import Document

# Import our modules
from utils.data_fetcher import fetch_financial_data
from utils.highlights_table import highlight_better

st.write("Python version on Streamlit Cloud:", sys.version)

# --- PAGE CONFIG ---
st.set_page_config(page_title="Financial Multi-Modal RAG Dashboard", page_icon="💰", layout="wide")
st.title("💰 Financial Multi-Modal RAG Dashboard")

# --- API KEY ---
st.sidebar.header("⚙️ Settings")
OPENAI_API_KEY = st.sidebar.text_input("🔑 Enter OpenAI API Key:", type="password")
if not OPENAI_API_KEY:
    st.sidebar.warning("Please enter your OpenAI API Key.")
    st.stop()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# --- SESSION STATE ---
if "qa_history" not in st.session_state:
    st.session_state["qa_history"] = []
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None

# --- TABS ---
tab1 = st.tabs(["📄 Documents"])

# -------- TAB 1: DOCUMENT UPLOAD --------
with tab1:
    uploaded_file = st.file_uploader("Upload a financial PDF", type=["pdf"])
    if uploaded_file and st.button("Process Document"):
        with st.spinner("Processing document..."):
            st.session_state["vectorstore"] = process_pdf(uploaded_file, st.session_state["vectorstore"])
            st.success("✅ Document processed!")


