import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import google.generativeai as genai
import io
import json
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

    numeric_cols = df.select_dtypes('number').columns.tolist()
    cat_cols = df.select_dtypes('object').columns.tolist()
    all_cols = df.columns.tolist()

    # AGENTIC LOOP: Let Gemini decide the plan
    with st.spinner("Agent is reasoning... Step 1/4: Planning analysis..."):

        schema_info = f"""
        Columns: {all_cols}
        Numeric columns: {numeric_cols}
        Categorical columns: {cat_cols}
        Missing values: {df.isnull().sum().to_dict()}
        Sample rows:
        {df.head(3).to_csv(index=False)}
        """

        planning_prompt = f"""
        You are an autonomous data analyst. User asked: "{user_question}"

        Dataset info:
        {schema_info}

        Return ONLY valid JSON. No markdown, no explanation. Keys required:
        1. "cleaning_steps": list of strings, like ["Fill numeric nulls with median"]
        2. "target_column": string, pick the column most likely to be the outcome/target from {all_cols}. If unsure, use "{numeric_cols[-1] if numeric_cols else all_cols[-1]}"
        3. "chart_ideas": list of exactly 3 dicts. Each dict MUST have "type" and "col". "type" is "histogram" or "bar". "col" MUST be from {all_cols}. Never invent columns.
        4. "key_metrics": list of strings

        Example: {{"cleaning_steps":["..."], "target_column":"Survived", "chart_ideas":[{{"type":"bar","col":"Sex"}}], "key_metrics":["shape"]}}
        """

        try:
            plan_response = model.generate_content(planning_prompt)
            plan_text = plan_response.text.strip().replace("```json", "").replace("```", "")
            plan = json.loads(plan_text)
            # Validate chart ideas
            for chart in plan.get("chart_ideas", []):
                if chart.get("col") not in all_cols:
                    raise ValueError(f"Invalid column {chart.get('col')}")
        except Exception as e:
            st.warning(f"Agent planning failed: {e}. Using fallback plan.")
            plan = {
                "cleaning_steps": ["Fill numeric nulls with median", "Fill categorical nulls with mode"],
                "target_column": numeric_cols[-1] if numeric_cols else all_cols[-1],
                "chart_ideas": [
                    {"type": "histogram", "col": numeric_cols[0]} if numeric_cols else {"type": "bar", "col": all_cols[0]},
                    {"type": "bar", "col": cat_cols[0]} if cat_cols else {"type": "bar", "col": all_cols[0]},
                    {"type": "bar", "col": all_cols[1]} if len(all_cols) > 1 else {"type": "bar", "col": all_cols[0]}
                ],
                "key_metrics": ["shape", "missing_total"]
            }

        st.subheader("Step 2: Agent's Plan")
        st.json(plan)

    # EXECUTE CLEANING
    with st.spinner("Step 2/4: Cleaning data..."):
        changes = []
        df_clean = df.copy()
        for step in plan["cleaning_steps"]:
            if "median" in step.lower():
                for col in numeric_cols:
                    if df_clean[col].isnull().sum() > 0:
                        n = df_clean[col].isnull().sum()
                        df_clean[col].fillna(df_clean[col].median(), inplace=True)
                        changes.append(f"Filled {n} nulls in {col} with median {df_clean[col].median():.1f}")
            elif "mode" in step.lower():
                for col in cat_cols:
                    if df_clean[col].isnull().sum() > 0:
                        n = df_clean[col].isnull().sum()
                        mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else "Unknown"
                        df_clean[col].fillna(mode_val, inplace=True)
                        changes.append(f"Filled {n} nulls in {col} with mode '{mode_val}'")
            elif "drop" in step.lower():
                for col in all_cols:
                    if df_clean[col].isnull().mean() > 0.7:
                        df_clean.drop(col, axis=1, inplace=True)
                        changes.append(f"Dropped {col}: >70% missing")

        st.write("Cleaning changes:")
        if changes:
            for c in changes: st.write(f"- {c}")
        else:
            st.write("- No cleaning needed")

    # EXECUTE ANALYSIS + CHARTS
    with st.spinner("Step 3/4: Generating charts..."):
        st.subheader("Step 3: Auto-Generated Charts")
        figures = []
        for i, chart in enumerate(plan["chart_ideas"][:3]):
            try:
                fig, ax = plt.subplots()
                chart_type = chart.get("type", "bar")
                col = chart.get("col", all_cols[0])

                if col not in df_clean.columns:
                    st.warning(f"Skipping chart {i+1}: column '{col}' not found after cleaning")
                    continue

                if chart_type == "histogram" and pd.api.types.is_numeric_dtype(df_clean[col]):
                    df_clean[col].hist(ax=ax)
                    ax.set_title(f"Distribution of {col}")
                else: # Default to bar for safety
                    df_clean[col].value_counts().head(10).plot(kind='bar', ax=ax)
                    ax.set_title(f"Top 10 values in {col}")

                plt.tight_layout()
                st.pyplot(fig)
                figures.append((f"chart_{i+1}_{col}.png", fig))

            except Exception as e:
                st.warning(f"Could not create chart {i+1} for '{col}': {str(e)}")

    # EXECUTE SUMMARY - DEFINE PROMPT FIRST SO IT ALWAYS EXISTS
    with st.spinner("Step 4/4: Writing executive summary..."):
        st.subheader("Step 4: AI Executive Summary")

        summary_prompt = f"""
        You are an analyst reporting to a non-technical CEO.
        User question: {user_question}
        Dataset columns: {all_cols}
        Shape after cleaning: {df_clean.shape}
        Cleaning done: {changes}
        Target column appears to be: {plan['target_column']}

        Write exactly 5 bullet points. Each 1 sentence. Use specific numbers. No jargon.
        End with 1 recommendation starting with "Recommendation:".
        """

        try:
            summary_response = model.generate_content(summary_prompt)
            summary_text = summary_response.text
            st.markdown(summary_text)
        except exceptions.ResourceExhausted:
            summary_text = f"""
- Gemini API rate limit hit: 15 requests/min on free tier. Wait 60s and retry.
- Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns. Cleaned to {df_clean.shape}.
- Potential target column: {plan['target_column']}.
- Most complete numeric column: {numeric_cols[0] if numeric_cols else 'None'} with mean {df_clean[numeric_cols[0]].mean():.2f if numeric_cols else 'N/A'}.
- Recommendation: For production, add caching or upgrade to paid Gemini tier to avoid rate limits during demos.
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
                fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                zf.writestr(fname, buf.getvalue())
        st.download_button("Download charts.zip", zip_buffer.getvalue(), "charts.zip")

    st.success("Agent complete! Worked on unknown CSV without hardcoding.")
