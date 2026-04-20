import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import google.generativeai as genai
import io
from datetime import datetime
from google.api_core import exceptions

st.set_page_config(page_title="Universal CSV Analyst", layout="wide")
st.title("Universal CSV Analyst 🤖")
st.write("Upload ANY CSV. I clean it, chart it, and summarize it — no hardcoding.")

# Configure Gemini - uses your Streamlit secret
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel('gemini-1.5-flash') # 15 RPM free tier

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
question = st.text_input("What should I analyze?", "What are the key patterns in this data?")

if st.button("Run Analysis", type="primary") and uploaded_file:
    df = pd.read_csv(uploaded_file)
    df_clean = df.copy()

    # ===== STEP 1: AUTOMATIC CLEANING =====
    st.subheader("Step 1: Automatic Cleaning")
    changes = []

    for col in df_clean.columns:
        # Auto-detect dates
        if 'date' in col.lower() or 'time' in col.lower():
            try:
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                changes.append(f"Converted '{col}' to datetime")
                continue
            except:
                pass

        # Numeric columns
        if pd.api.types.is_numeric_dtype(df_clean[col]):
            nulls = df_clean[col].isnull().sum()
            if nulls > 0:
                median = df_clean[col].median()
                df_clean[col].fillna(median, inplace=True)
                changes.append(f"Filled {nulls} nulls in numeric '{col}' with median {median:.2f}")
        # Text/categorical columns
        else:
            nulls = df_clean[col].isnull().sum()
            if nulls > 0:
                mode = df_clean[col].mode()
                fill_val = mode[0] if not mode.empty else 'Unknown'
                df_clean[col].fillna(fill_val, inplace=True)
                changes.append(f"Filled {nulls} nulls in text '{col}' with '{fill_val}'")

    # Drop columns with >70% missing
    for col in df.columns:
        if df[col].isnull().mean() > 0.7:
            df_clean.drop(col, axis=1, inplace=True)
            changes.append(f"Dropped '{col}' (>70% missing)")

    if changes:
        for c in changes[:6]:
            st.write(f"- {c}")
    else:
        st.write("- No cleaning needed")
    st.write(f"Shape: {df.shape} → {df_clean.shape}")

    # ===== STEP 2: AUTO CHARTS =====
    st.subheader("Step 2: Auto Charts")
    figures = []

    numeric_cols = df_clean.select_dtypes(include=['number']).columns.tolist()
    datetime_cols = df_clean.select_dtypes(include=['datetime']).columns.tolist()
    cat_cols = [c for c in df_clean.select_dtypes('object').columns if df_clean[c].nunique() < 50]
    text_cols = [c for c in df_clean.select_dtypes('object').columns if df_clean[c].nunique() >= 50]

    # Chart 1: Priority - numeric > datetime > categorical
    if numeric_cols:
        col = numeric_cols[0]
        fig1, ax1 = plt.subplots()
        df_clean[col].hist(ax=ax1, bins=20, edgecolor='black')
        ax1.set_title(f'Distribution of {col}')
        ax1.set_xlabel(col)
        ax1.set_ylabel('Count')
        plt.tight_layout()
        st.pyplot(fig1)
        figures.append((f"hist_{col}.png", fig1))
    elif datetime_cols:
        col = datetime_cols[0]
        fig1, ax1 = plt.subplots()
        df_clean.set_index(col).resample('ME').size().plot(ax=ax1)
        ax1.set_title(f'Records Over Time ({col})')
        ax1.set_ylabel('Count per month')
        plt.tight_layout()
        st.pyplot(fig1)
        figures.append((f"time_{col}.png", fig1))
    elif cat_cols:
        col = cat_cols[0]
        fig1, ax1 = plt.subplots()
        df_clean[col].value_counts().head(10).plot(kind='bar', ax=ax1)
        ax1.set_title(f'Top 10 {col}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig1)
        figures.append((f"bar_{col}.png", fig1))

    # Chart 2: First categorical
    if cat_cols:
        col = cat_cols[0]
        fig2, ax2 = plt.subplots()
        df_clean[col].value_counts().head(10).plot(kind='bar', ax=ax2, color='skyblue', edgecolor='black')
        ax2.set_title(f'Breakdown by {col}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig2)
        figures.append((f"cat_{col}.png", fig2))

    # Chart 3: Correlation or second categorical
    if len(numeric_cols) >= 2:
        fig3, ax3 = plt.subplots()
        corr_series = df_clean[numeric_cols].corr().iloc[0].sort_values().drop(numeric_cols[0])
        corr_series.plot(kind='barh', ax=ax3, color='coral')
        ax3.set_title(f'Correlation with {numeric_cols[0]}')
        plt.tight_layout()
        st.pyplot(fig3)
        figures.append(("correlation.png", fig3))
    elif len(cat_cols) >= 2:
        col = cat_cols[1]
        fig3, ax3 = plt.subplots()
        df_clean[col].value_counts().head(10).plot(kind='bar', ax=ax3, color='orange', edgecolor='black')
        ax3.set_title(f'Breakdown by {col}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig3)
        figures.append((f"cat2_{col}.png", fig3))

    if text_cols:
        st.info(f"Skipped high-cardinality text columns: {', '.join(text_cols[:3])}")

    # ===== STEP 3: AI SUMMARY =====
    st.subheader("Step 3: AI Summary")

    stats_parts = []
    stats_parts.append(f"{len(df_clean)} rows, {len(df_clean.columns)} columns")
    if numeric_cols:
        stats_parts.append(f"'{numeric_cols[0]}' avg {df_clean[numeric_cols[0]].mean():.1f}")
    if cat_cols:
        top_val = df_clean[cat_cols[0]].mode()[0]
        top_count = df_clean[cat_cols[0]].value_counts().iloc[0]
        stats_parts.append(f"top '{cat_cols[0]}' = '{top_val}' ({top_count} times)")
    if datetime_cols:
        start = df_clean[datetime_cols[0]].min().date()
        end = df_clean[datetime_cols[0]].max().date()
        stats_parts.append(f"date range {start} to {end}")

    stats = ". ".join(stats_parts) + "."
    prompt = f"Question: {question}. Data: {stats} Write 5 short bullet points for a manager. Use numbers. End with 'Recommendation:'"

    try:
        response = model.generate_content(prompt)
        summary = response.text
    except exceptions.ResourceExhausted:
        summary = f"""- Analysis complete (Gemini rate limit hit - 15 req/min free tier).
- {stats_parts[0]}
- {stats_parts[1] if len(stats_parts) > 1 else ''}
- {stats_parts[2] if len(stats_parts) > 2 else ''}
- Recommendation: Re-run in 60 seconds for full AI summary."""
        st.warning("Using statistical fallback - wait 60s between runs for AI.")
    except Exception as e:
        summary = f"Error: {str(e)}"
        st.error(summary)

    st.markdown(summary)

    # ===== DOWNLOADS =====
    st.subheader("Downloads")
    report = f"# Analysis Report\nFile: {uploaded_file.name}\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n## Summary\n{summary}\n\n## Cleaning Log\n" + "\n".join([f"- {c}" for c in changes])

    st.download_button(
        "Download report.md",
        report.encode(),
        file_name=f"report_{uploaded_file.name.split('.')[0]}.md"
    )

    if figures:
        zip_buf = io.BytesIO()
        import zipfile
        with zipfile.ZipFile(zip_buf, 'w') as zf:
            for name, fig in figures:
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                zf.writestr(name, buf.getvalue())
        st.download_button(
            "Download charts.zip",
            zip_buf.getvalue(),
            file_name="charts.zip"
        )

    st.success("✅ Analysis complete — works on any CSV!")
