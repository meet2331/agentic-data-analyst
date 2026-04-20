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
    chart_ideas = plan.get("chart_ideas", [])

    # Fallback: if Gemini gave us garbage, make our own charts
    if not chart_ideas or not isinstance(chart_ideas, list):
        st.info("Agent plan unclear. Using fallback: auto-charts for first numeric + categorical columns.")
        numeric_cols = df.select_dtypes('number').columns.tolist()
        cat_cols = df.select_dtypes('object').columns.tolist()
        chart_ideas = []
        if numeric_cols: chart_ideas.append({"type": "histogram", "col": numeric_cols[0]})
        if cat_cols: chart_ideas.append({"type": "bar", "col": cat_cols[0]})
        if len(numeric_cols) > 1: chart_ideas.append({"type": "correlation", "col": numeric_cols[0]})

    for i, chart in enumerate(chart_ideas[:3]):
        try:
            fig, ax = plt.subplots()
            chart_type = chart.get("type", "histogram") # Default to histogram if missing
            col = chart.get("col", df.columns[0]) # Default to first column if missing

            if col not in df.columns:
                st.warning(f"Skipping chart {i+1}: column '{col}' not found")
                continue

            if chart_type == "histogram" and pd.api.types.is_numeric_dtype(df[col]):
                df[col].hist(ax=ax)
                ax.set_title(f"Distribution of {col}")
            elif chart_type == "bar":
                df[col].value_counts().head(10).plot(kind='bar', ax=ax)
                ax.set_title(f"Top values in {col}")
            elif chart_type == "correlation" and len(df.select_dtypes('number').columns) > 1:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df.select_dtypes('number').corr()[col].sort_values().plot(kind='barh', ax=ax)
                    ax.set_title(f"Correlation with {col}")
                else:
                    df[col].value_counts().head(10).plot(kind='bar', ax=ax)
                    ax.set_title(f"Top values in {col}")
            else: # Final fallback
                df[col].value_counts().head(10).plot(kind='bar', ax=ax)
                ax.set_title(f"Values in {col}")

            st.pyplot(fig)
            figures.append((f"chart_{i+1}_{col}.png", fig))

        except Exception as e:
            st.warning(f"Could not create chart {i+1} for '{chart.get('col', 'unknown')}': {str(e)}")

    # EXECUTE SUMMARY
with st.spinner("Step 4/4: Writing executive summary..."):
    st.subheader("Step 4: AI Executive Summary")
    try:
        summary_response = model.generate_content(summary_prompt)
        summary_text = summary_response.text
        st.markdown(summary_text)
    except exceptions.ResourceExhausted:
        summary_text = f"""
        - Gemini API rate limit hit: 15 requests/min on free tier. Wait 60s and try again.
        - Your data shape: {df.shape}. Agent successfully loaded and cleaned it.
        - Fallback insight: First numeric column '{df.select_dtypes('number').columns[0] if len(df.select_dtypes('number').columns)>0 else 'N/A'}' has mean {df.select_dtypes('number').iloc[:,0].mean():.2f} if numeric exists.
        - Fallback insight: First categorical column has {df.select_dtypes('object').iloc[:,0].nunique() if len(df.select_dtypes('object').columns)>0 else 'N/A'} unique values.
        - Recommendation: For production demos, use paid Gemini tier or add caching to avoid rate limits.
        """
        st.warning("Rate limit hit. Showing fallback summary:")
        st.markdown(summary_text)
    except Exception as e:
        summary_text = f"Agent error during summary: {str(e)}"
        st.error(summary_text)

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
