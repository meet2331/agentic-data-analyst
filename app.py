import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
   
st.set_page_config(page_title="Agentic Data Analyst", layout="wide")
st.title("Agentic Data Analyst 🤖")
st.write("Upload any CSV. Ask a business question. Get charts + summary in 60s.")
   
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
question = st.text_input("What business question should I answer?",  "What factors most affect the target column?")
if st.button("Run Agent", type="primary") and uploaded_file:
   with st.spinner("Agent is thinking... Loading → Cleaning → Analyzing → Plotting..."):
    # This is where you'd call Gemini API. For free demo, we simulate.
    # In real version, you'd use Gemini API or LangChain + local model.
           
    df = pd.read_csv(uploaded_file)
    st.subheader("1. Data Preview")
    st.dataframe(df.head())       
    st.subheader("2. Auto-Generated Insights")
    st.write("- Dataset shape:", df.shape)
    st.write("- Missing values found:", df.isnull().sum().sum())
    st.write("- The agent would now run EDA, create 3 charts, and write summary.")
           
    # Example chart so it looks real
    fig, ax = plt.subplots()
    df.hist(ax=ax)
    st.pyplot(fig)
           
    st.success("Done! In the full agent version, this is where Gemini would loop, fix errors, and save files. Business impact: 2hr → 60s.")
   
st.caption("Built to demonstrate agentic workflow design: Goal → Plan → Execute → Recover")