import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import google.generativeai as genai
import io
import traceback
from google.api_core import exceptions

st.set_page_config(page_title="Agentic Data Analyst", layout="wide")
st.title("Agentic Data Analyst 🤖")
st.write("Upload ANY CSV. I’ll figure out what it is and analyze it autonomously.")

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel('gemini-2.5-flash')

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
user_question = st.text_input("What business question should I answer?",
                              "What are the 3 most important insights from this data?")

if st.button("Run Agent", type="primary") and uploaded_file:

    df = pd.read_csv(uploaded_file)
    st.subheader("Step 1: Data Loaded")
    st.write(f"Shape: {df.shape}")
    st.dataframe(df.head())

    # AGENTIC LOOP: Let Gemini decide the plan
    with st.spinner("Agent is reasoning... Step 1/4: Planning analysis..."):

        # Give Gemini the schema + sample and ask for a plan
        schema_info = f"""
        Columns: {list(df.columns)}
        Data types: {df.dtypes.to_dict()}
        Missing values: {df.isnull().sum().to_dict()}
        Sample rows:
        {df.head(3).to_csv(index=False)}
        """

        planning_prompt = f"""
        You are an autonomous data analyst. A user uploaded a CSV and asked: "{user_question}"

        Dataset info:
        {schema_info}

        Your job: Write a step-by-step analysis plan in JSON format.
        Include:
        1. "cleaning_steps": list of 2-3 specific cleaning actions needed
        2. "target_column": the column that looks like the target/outcome to predict/explain. If none, return null.
        3. "chart_ideas": list of 3 chart types + x/y columns that would answer the user question
        4. "key_metrics": 2-3 metrics to calculate

        Only return valid JSON. No explanation.
        """

        try:
            plan_response = model.generate_content(planning_prompt)
            # Extract JSON from response - Gemini sometimes wraps in ```json
            plan_text = plan_response.text.strip().replace("```json", "").replace("```", "")
            import json
            plan = json.loads(plan_text)
        except:
            st.error("Agent failed to make a plan. Using fallback generic plan.")
            plan = {
                "cleaning_steps": ["Fill numeric nulls with median", "Fill categorical nulls with mode"],
                "target_column": df.columns[-1], # guess last col
                "chart_ideas": [
                    {"type": "histogram", "col": df.select_dtypes('number').columns[0] if len(df.select_dtypes('number').columns)>0 else df.columns[0]},
                    {"type": "bar", "col": df.select_dtypes('object').columns[0] if len(df.select_dtypes('object').columns)>0 else df.columns[0]},
                    {"type": "correlation", "col": "all_numeric"}
                ],
                "key_metrics": ["shape", "missing_total"]
            }

        st.subheader("Step 2: Agent's Plan")
        st.json(plan)

    # EXECUTE CLEANING
    with st.spinner("Step 2/4: Cleaning data..."):
        changes = []
        for step in plan["cleaning_steps"]:
            if "median" in step.lower():
                for col in df.select_dtypes('number').columns:
                    if df[col].isnull().sum() > 0:
                        n = df[col].isnull().sum()
                        df[col].fillna(df[col].median(), inplace=True)
                        changes.append(f"Filled {n} nulls in {col} with median")
            elif "mode" in step.lower():
                for col in df.select_dtypes('object').columns:
                    if df[col].isnull().sum() > 0:
                        n = df[col].isnull().sum()
                        df[col].fillna(df[col].mode()[0], inplace=True)
                        changes.append(f"Filled {n} nulls in {col} with mode")
            elif "drop" in step.lower() and "column" in step.lower():
                # Drop high-missing columns
                for col in df.columns:
                    if df[col].isnull().mean() > 0.7:
                        df.drop(col, axis=1, inplace=True)
                        changes.append(f"Dropped {col}: >70% missing")

        st.write("Cleaning changes:")
        for c in changes: st.write(f"- {c}")

    # EXECUTE ANALYSIS + CHARTS
    with st.spinner("Step 3/4: Generating charts..."):
        st.subheader("Step 3: Auto-Generated Charts")

        figures = []
        for i, chart in enumerate(plan["chart_ideas"][:3]): # max 3 charts
            try:
                fig, ax = plt.subplots()
                if chart["type"] == "histogram":
                    df[chart["col"]].hist(ax=ax)
                    ax.set_title(f"Distribution of {chart['col']}")
                elif chart["type"] == "bar":
                    df[chart["col"]].value_counts().head(10).plot(kind='bar', ax=ax)
                    ax.set_title(f"Top values in {chart['col']}")
                elif chart["type"] == "correlation" and len(df.select_dtypes('number').columns) > 1:
                    df.select_dtypes('number').corr()[chart.get("col", df.columns[0])].sort_values().plot(kind='barh', ax=ax)
                    ax.set_title("Correlation with target")

                st.pyplot(fig)
                figures.append((f"chart_{i+1}.png", fig))
            except Exception as e:
                st.warning(f"Could not create chart {i+1}: {e}")

    # EXECUTE SUMMARY
    with st.spinner("Step 4/4: Writing executive summary..."):
        st.subheader("Step 4: AI Executive Summary")

        summary_prompt = f"""
        Dataset columns: {list(df.columns)}
        User question: {user_question}
        Key stats: Shape {df.shape}, Target appears to be {plan['target_column']}
        Cleaning done: {changes}

        Write exactly 5 bullet points for a non-technical CEO. Use specific numbers from the data.
        End with 1 recommendation.
        """

        try:
            summary_response = model.generate_content(summary_prompt)
            summary_text = summary_response.text
            st.markdown(summary_text)
        except exceptions.ResourceExhausted:
            summary_text = "- Analysis complete but Gemini rate limit hit. Upgrade to paid tier for live summaries."
            st.warning(summary_text)

    # SAVE FILES
    st.subheader("Downloads")
    st.download_button("Download summary.txt", summary_text.encode(), "summary.txt")

    if figures:
        zip_buffer = io.BytesIO()
        import zipfile
        with zipfile.ZipFile(zip_buffer, 'w') as zf:
            for fname, fig in figures:
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight')
                zf.writestr(fname, buf.getvalue())
        st.download_button("Download charts.zip", zip_buffer.getvalue(), "charts.zip")

    st.success("Agent complete! Worked on unknown CSV without hardcoding.")
