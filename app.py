import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
import io
import numpy as np
from datetime import datetime
from google.api_core import exceptions

st.set_page_config(page_title="Universal CSV Analyst", layout="wide")
st.title("Universal CSV Analyst 🤖")
st.write("Upload ANY CSV. I clean it, chart it, and summarize it — no hardcoding.")

# Configure Gemini - uses your Streamlit secret
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel('gemini-2.0-flash') # 15 RPM free tier
sns.set_style("whitegrid")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
question = st.text_input("What should I analyze?", "What are the key patterns in this data?")

# ===== HELPER FUNCTIONS =====
def smart_label(col_name, series):
    """Guess what the numbers represent from name + data"""
    col_lower = col_name.lower()
    if any(k in col_lower for k in ['revenue', 'sales', 'price', 'cost', 'amount', 'income', 'spend']):
        return 'Amount ($)'
    elif any(k in col_lower for k in ['age', 'year', 'tenure', 'duration', 'days']):
        return col_name.title() + ' (Units)'
    elif any(k in col_lower for k in ['count', 'num', 'qty', 'click', 'view', 'visit', 'impression']):
        return f'Number of {col_name}'
    elif series.nunique() == 2 and set(series.unique()).issubset({0, 1, True, False}):
        return f'{col_name} (0=No, 1=Yes)'
    elif pd.api.types.is_integer_dtype(series) and 'id' not in col_lower:
        return f'Count of {col_name}'
    else:
        return col_name

def pick_best_numeric(df, question):
    """Pick most interesting numeric column based on question + variance"""
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if not numeric_cols:
        return None

    # Priority 1: Mentioned in question
    for col in numeric_cols:
        if col.lower() in question.lower():
            return col

    # Priority 2: Skip ID-like columns, pick highest coefficient of variation
    candidates = [c for c in numeric_cols if df[c].nunique() > 10 and not c.lower().endswith('id')]
    if candidates:
        cv = df[candidates].std() / df[candidates].mean().replace(0, np.nan)
        return cv.idxmax()

    return numeric_cols[0] if numeric_cols else None

def pick_best_categorical(df):
    """Pick categorical with 2-20 unique values, prefer 3-10"""
    cat_cols = df.select_dtypes('object').columns
    candidates = [(c, df[c].nunique()) for c in cat_cols if 2 <= df[c].nunique() <= 50]
    if not candidates:
        return None
    # Prefer 3-10 categories as most readable
    candidates.sort(key=lambda x: abs(x[1] - 6))
    return candidates[0][0]

# ===== MAIN APP =====
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
                if df_clean[col].notna().sum() > len(df_clean) * 0.5: # Only if >50% parsed
                    changes.append(f"Converted '{col}' to datetime")
                    continue
                else:
                    df_clean[col] = df[col] # Revert if mostly NaT
            except:
                pass

        # Numeric columns
        if pd.api.types.is_numeric_dtype(df_clean[col]):
            nulls = df_clean[col].isnull().sum()
            if nulls > 0:
                median = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median)
                changes.append(f"Filled {nulls} nulls in numeric '{col}' with median {median:.2f}")
        # Text/categorical columns
        else:
            nulls = df_clean[col].isnull().sum()
            if nulls > 0:
                mode = df_clean[col].mode()
                fill_val = mode[0] if not mode.empty else 'Unknown'
                df_clean[col] = df_clean[col].fillna(fill_val)
                changes.append(f"Filled {nulls} nulls in text '{col}' with '{fill_val}'")

    # Drop columns with >70% missing
    cols_to_drop = []
    for col in df.columns:
        if df[col].isnull().mean() > 0.7:
            cols_to_drop.append(col)
            changes.append(f"Dropped '{col}' (>70% missing)")
    if cols_to_drop:
        df_clean = df_clean.drop(columns=cols_to_drop)

    if changes:
        for c in changes[:8]:
            st.write(f"- {c}")
        if len(changes) > 8:
            st.write(f"-...and {len(changes)-8} more")
    else:
        st.write("- No cleaning needed")
    st.write(f"Shape: {df.shape} → {df_clean.shape}")

    # ===== STEP 2: AUTO CHARTS =====
    st.subheader("Step 2: Auto Charts")
    figures = []

    numeric_cols = df_clean.select_dtypes(include=['number']).columns.tolist()
    datetime_cols = df_clean.select_dtypes(include=['datetime']).columns.tolist()
    cat_cols = [c for c in df_clean.select_dtypes('object').columns if 2 <= df_clean[c].nunique() < 50]
    text_cols = [c for c in df_clean.select_dtypes('object').columns if df_clean[c].nunique() >= 50]

    # Chart 1: Smart Distribution
    best_num = pick_best_numeric(df_clean, question)
    if best_num:
        series = df_clean[best_num].dropna()
        fig1, ax1 = plt.subplots(figsize=(8, 5))

        if series.nunique() == 2: # Binary data
            counts = series.value_counts().sort_index()
            counts.plot(kind='bar', ax=ax1, color='#1f77b4', edgecolor='black')
            labels = ['No/0', 'Yes/1'] if set(series.unique()).issubset({0,1}) else [str(i) for i in counts.index]
            ax1.set_xticklabels(labels, rotation=0)
            pos_count = counts.get(1, counts.get(True, 0))
            ax1.set_title(f'{best_num}: {pos_count} Positive ({pos_count/len(series):.0%} of total)')
        else: # Continuous
            # Cap outliers at 99th percentile for readability
            plot_series = series.clip(upper=series.quantile(0.99))
            plot_series.hist(ax=ax1, bins=min(30, series.nunique()), edgecolor='black', color='#1f77b4')
            mean_val, median_val = series.mean(), series.median()
            ax1.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
            ax1.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.1f}')
            ax1.legend()
            skew = series.skew()
            skew_text = "Right-skewed" if skew > 1 else "Left-skewed" if skew < -1 else "Normal"
            ax1.set_title(f'Distribution of {best_num} | {skew_text}')

        ax1.set_xlabel(smart_label(best_num, series))
        ax1.set_ylabel('Frequency')
        ax1.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig1)
        st.caption(f"Insight: 75% of values are ≤ {series.quantile(0.75):.1f}. Range: {series.min():.1f} to {series.max():.1f}")
        figures.append((f"dist_{best_num}.png", fig1))

    # Chart 2: Top Categories with Labels
    best_cat = pick_best_categorical(df_clean)
    if best_cat:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        counts = df_clean[best_cat].value_counts().head(10)
        total = len(df_clean)

        counts.plot(kind='bar', ax=ax2, color='skyblue', edgecolor='black')
        ax2.set_title(f'Top {len(counts)} Categories in {best_cat}')
        ax2.set_xlabel(best_cat)
        ax2.set_ylabel(f'Number of Records (Total: {total})')

        # Add value + percentage on each bar
        for i, v in enumerate(counts):
            ax2.text(i, v + total*0.01, f'{v}\n({v/total:.0%})',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig2)
        st.caption(f"Insight: '{counts.index[0]}' dominates with {counts.iloc[0]/total:.0%} of all records. Top 3 = {counts.iloc[:3].sum()/total:.0%}")
        figures.append((f"cat_{best_cat}.png", fig2))

    # Chart 3: Correlation Analysis
    if len(numeric_cols) >= 2:
        fig3, ax3 = plt.subplots(figsize=(9, 6))
        corr_matrix = df_clean[numeric_cols].corr()

        # If we have a target mentioned in question, do bar chart
        target = None
        for col in numeric_cols:
            if col.lower() in question.lower():
                target = col
                break

        if target and len(numeric_cols) > 2:
            corr_series = corr_matrix[target].drop(target).sort_values()
            colors = ['#d62728' if x < 0 else '#2ca02c' for x in corr_series]
            corr_series.plot(kind='barh', ax=ax3, color=colors, edgecolor='black')
            ax3.axvline(0, color='black', linewidth=1.5)
            ax3.axvline(0.3, color='green', linestyle=':', linewidth=1, label='Strong positive')
            ax3.axvline(-0.3, color='red', linestyle=':', linewidth=1, label='Strong negative')
            ax3.set_title(f'What Correlates with {target}?')
            ax3.set_xlabel('Correlation Coefficient (-1 to +1)')
            ax3.legend()
            st.caption("Insight: Green bars increase with target, red bars decrease. |corr| > 0.3 is noteworthy.")
        else: # Show heatmap
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) # Hide upper triangle
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax3,
                       fmt='.2f', mask=mask, square=True, linewidths=0.5)
            ax3.set_title('Correlation Between All Numeric Variables')
            st.caption("Insight: Red = positive correlation, Blue = negative. Darker = stronger relationship.")

        plt.tight_layout()
        st.pyplot(fig3)
        figures.append(("correlation.png", fig3))

    # Chart 4: Time Series if available
    if datetime_cols:
        date_col = datetime_cols[0]
        fig4, ax4 = plt.subplots(figsize=(10, 4))
        ts = df_clean.set_index(date_col).resample('ME').size()
        ts.plot(ax=ax4, marker='o', color='#2ca02c')
        ax4.set_title(f'Records Over Time by {date_col}')
        ax4.set_xlabel('Month')
        ax4.set_ylabel('Records per Month')
        ax4.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig4)
        peak_month = ts.idxmax().strftime('%b %Y')
        st.caption(f"Insight: Peak activity in {peak_month} with {ts.max()} records. Avg: {ts.mean():.0f}/month")
        figures.append((f"time_{date_col}.png", fig4))

    if text_cols:
        st.info(f"Skipped {len(text_cols)} high-cardinality text columns: {', '.join(text_cols[:3])}")

    # ===== STEP 3: AI SUMMARY =====
    st.subheader("Step 3: AI Summary")

    stats_parts = []
    stats_parts.append(f"{len(df_clean)} rows, {len(df_clean.columns)} columns")
    if best_num:
        stats_parts.append(f"'{best_num}' avg {df_clean[best_num].mean():.1f}, median {df_clean[best_num].median():.1f}")
    if best_cat:
        top_val = df_clean[best_cat].mode()[0]
        top_count = df_clean[best_cat].value_counts().iloc[0]
        stats_parts.append(f"top '{best_cat}' = '{top_val}' ({top_count} times, {top_count/len(df_clean):.0%})")
    if datetime_cols:
        start = df_clean[datetime_cols[0]].min().date()
        end = df_clean[datetime_cols[0]].max().date()
        stats_parts.append(f"date range {start} to {end}")

    stats = ". ".join(stats_parts) + "."
    prompt = f"""Question: {question}
Data summary: {stats}
Numeric columns: {numeric_cols[:5]}
Categorical columns: {cat_cols[:5]}

Write 5 short bullet points for a business manager. Use specific numbers from the data. Start each with an action verb. End with 'Recommendation:' and one concrete next step."""

    try:
        response = model.generate_content(prompt)
        summary = response.text
    except exceptions.ResourceExhausted:
        summary = f"""- Dataset contains {len(df_clean)} records across {len(df_clean.columns)} fields.
- {stats_parts[1] if len(stats_parts) > 1 else 'Analysis complete.'}
- {stats_parts[2] if len(stats_parts) > 2 else ''}
- Gemini rate limit hit (15 req/min free tier). Statistical summary shown.
- Recommendation: Re-run in 60 seconds for full AI insights, or explore charts above."""
        st.warning("Using statistical fallback - wait 60s between runs for AI.")
    except Exception as e:
        summary = f"Error generating summary: {str(e)}"
        st.error(summary)

    st.markdown(summary)

    # ===== DOWNLOADS =====
    st.subheader("Downloads")
    report = f"""# Analysis Report
File: {uploaded_file.name}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Summary
{summary}

## Cleaning Log
""" + "\n".join([f"- {c}" for c in changes]) + f"""

## Columns Analyzed
Numeric: {', '.join(numeric_cols)}
Categorical: {', '.join(cat_cols)}
Datetime: {', '.join(datetime_cols)}
"""

    st.download_button(
        "Download report.md",
        report.encode(),
        file_name=f"report_{uploaded_file.name.split('.')[0]}.md",
        mime="text/markdown"
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
            file_name="charts.zip",
            mime="application/zip"
        )

    st.success("✅ Analysis complete — works on any CSV!")

elif uploaded_file is None:
    st.info("👆 Upload a CSV file to begin analysis")
