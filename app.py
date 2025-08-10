import streamlit as st
import os
import sys
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
from langchain.schema import Document

# Import our modules
from utils.data_fetcher import fetch_financial_data
from utils.rag_engine import process_pdf, add_docs_to_rag
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
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📄 Documents", "🖼️ Charts"])

# -------- TAB 1: DOCUMENT UPLOAD --------
with tab1:
    uploaded_file = st.file_uploader("Upload a financial PDF", type=["pdf"])
    if uploaded_file and st.button("Process Document"):
        with st.spinner("Processing document..."):
            st.session_state["vectorstore"] = process_pdf(uploaded_file, st.session_state["vectorstore"])
            st.success("✅ Document processed!")

# -------- TAB 2: CHART UPLOAD --------
with tab2:
    uploaded_images = st.file_uploader("Upload financial charts", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    if uploaded_images and st.button("Process Charts"):
        with st.spinner("Generating chart captions..."):
            image_captions = []
            for img in uploaded_images:
                img_obj = Image.open(img).convert("RGB")
                caption = generate_chart_caption(img_obj, blip_processor, blip_model)
                image_captions.append((img.name, caption))
            chart_docs = [Document(page_content=cap, metadata={"source": name}) for name, cap in image_captions]
            st.session_state["vectorstore"] = add_docs_to_rag(chart_docs, st.session_state["vectorstore"])
            st.session_state["chart_captions"] = image_captions
            st.success("✅ Charts processed!")


