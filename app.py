import os
# Suppress Hugging Face & Tokenizer warnings before loading libraries
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import pandas as pd
import numpy as np
from src.retriever import PaperRetriever
from src.llm_service import LLMService
from src.pipeline import RelayPipeline
from src.load_tester import LoadTester

st.set_page_config(page_title="The Relay - Research Assistant", layout="wide")

st.title("⚡ The Relay: arXiv Assistant & Load Testing Suite")
st.markdown("Architected for answering queries on *Attention*, *Moshi*, and *Dynamic Chunking* under strict rate limits.")

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")
mode = st.sidebar.radio("LLM Engine Mode", ["Stand-in Model (Simulated)", "Groq API (Real Free Tier)"])
groq_key = st.sidebar.text_input("Groq API Key", type="password") if mode == "Groq API (Real Free Tier)" else None

st.sidebar.subheader("Retrieval Settings")
piece_size = st.sidebar.select_slider("Piece Size (Tokens)", options=[250, 500, 1000], value=500)
pieces_fetched = st.sidebar.slider("Pieces Fetched", 1, 5, 3)
rerank = st.sidebar.checkbox("Enable Re-ranking", value=False)

# Initialize Components
@st.cache_resource
def init_retriever(psize):
    retriever = PaperRetriever()
    retriever.load_and_chunk_papers(piece_size=psize)
    return retriever

retriever = init_retriever(piece_size)
llm_service = LLMService(mode="groq" if "Groq" in mode else "standin", api_key=groq_key)
pipeline = RelayPipeline(retriever=retriever, llm=llm_service)

tab1, tab2, tab3 = st.tabs(["💬 Interactive Assistant", "📊 Benchmark Load Tester", "📋 Report Tables"])

with tab1:
    st.subheader("Ask Questions About arXiv Papers")
    query = st.text_input("Enter engineering question:", value="How does multi-head attention work in Attention Is All You Need?")
    
    if st.button("Submit Question", type="primary"):
        with st.spinner("Executing 4-Step Pipeline..."):
            ret_setting = {"piece_size": piece_size, "pieces_fetched": pieces_fetched, "rerank": rerank}
            res = pipeline.run(query, retrieval_setting=ret_setting)
            
            st.markdown(f"### Answer Grade: `{res['answer_grade']}`")
            st.write(res["content"])
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Time to 1st Token", f"{res['first_token_ms']} ms")
            col2.metric("Total Latency", f"{res['total_ms']} ms")
            col3.metric("Tokens In/Out", f"{res['tokens_in']} / {res['tokens_out']}")
            col4.metric("Retries Count", res["retries"])

with tab2:
    st.subheader("Run Traffic Load Shapes")
    scenario = st.selectbox("Select Scenario", ["Steady Traffic", "Sudden Burst", "Sustained Overload"])
    num_reqs = st.number_input("Number of Requests", min_value=5, max_value=100, value=10)
    
    if st.button("Run Benchmark Scenario"):
        tester = LoadTester(pipeline)
        delay = 1.0 if scenario == "Steady Traffic" else (0.05 if scenario == "Sudden Burst" else 0.2)
        
        with st.spinner(f"Running {scenario}..."):
            df_res = tester.run_scenario(scenario, num_reqs, delay, source="groq" if "Groq" in mode else "stand-in")
            st.success(f"Completed {len(df_res)} requests. Raw data logged to `data/results.csv`.")
            st.dataframe(df_res)

with tab3:
    st.subheader("Assignment Report Tables View")
    csv_path = "data/results.csv"
    if os.path.exists(csv_path):
        df_raw = pd.read_csv(csv_path)
        st.markdown("### Raw Results (`results.csv`) Summary")
        st.dataframe(df_raw.tail(15))
        
        st.markdown("### Table 2a Preview: Latency Percentiles")
        if not df_raw.empty:
            summary = df_raw.groupby("scenario").agg(
                requests=("request_id", "count"),
                p50_ttft=("first_token_ms", lambda x: float(np.percentile(x, 50))),
                p95_ttft=("first_token_ms", lambda x: float(np.percentile(x, 95))),
                p50_total=("total_ms", lambda x: float(np.percentile(x, 50))),
                p95_total=("total_ms", lambda x: float(np.percentile(x, 95)))
            ).reset_index()
            st.table(summary)
    else:
        st.info("No run logs found yet. Run a benchmark scenario to generate `results.csv`.")