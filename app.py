import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
import io
import zipfile
from datetime import datetime
from google.api_core import exceptions

# ======================= PAGE CONFIG =======================
st.set_page_config(page_title="Universal CSV Analyst", layout="wide")
st.title("Universal CSV Analyst 🤖")
st.write("Upload ANY CSV. I clean it, chart it, and summarize it — with clear, annotated charts.")

# Configure Gemini
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel('gemini-2.0-flash')
sns.set_style("whitegrid")

# ======================= SIDEBAR OPTIONS =======================
st.sidebar.header("Options")
cap_outliers_toggle = st.sidebar.toggle("Cap outliers (IQR)", value=True, help="Clips extreme values at Q1-1.5*IQR and Q3+1.5*IQR")
show_dict = st.sidebar.toggle("Show data dictionary", value=True)

# ======================= HELPER FUNCTIONS =======================
def infer_unit(col_name: str) -> str:
    col_lower = col_name.lower()
    if 'age' in col_lower: return 'years'
    if any(k in col_lower for k in ['revenue','sales','price','cost','income','amount']): return 'USD ($)'
    if any(k in col_lower for k in ['click','visit','impression','count','qty']): return 'count'
    if any(k in col_lower for k in ['rate','ratio','pct','percent']): return '%'
    if any(k in col_lower for k in ['time','duration','days']): return 'seconds'
    return ''

def format_axis_label(col: str, data: pd.Series) -> str:
    unit = infer_unit(col)
    max_val = data.max()
    if pd.isna(max_val): return col
    if max_val > 1e6: base = f"{col} (millions)"
    elif max_val > 1e3: base = f"{col} (thousands)"
    else: base = col
    return f"{base} ({unit})" if unit else base

def cap_outliers_iqr(series: pd.Series) -> pd.Series:
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    return series.clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)

def pick_best_numeric(df, question):
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if not numeric_cols: return None
    for col in numeric_cols:
        if col.lower() in question.lower(): return col
    candidates = [c for c in numeric_cols if df[c].nunique() > 10 and not c.lower().endswith('id')]
    if candidates:
        cv = df[candidates].std() / df[candidates].mean().replace(0, np.nan)
        return cv.idxmax()
    return numeric_cols[0]

def pick_best_categorical(df):
    cat_cols = [c for c in df.select_dtypes('object').columns if 2 <= df[c].nunique() <= 50]
    if not cat_cols: return None
    cat_cols.sort(key=lambda x: abs(df[x].nunique() - 6)) # prefer ~6 categories
    return cat_cols[0]

# ======================= MAIN APP =======================
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
question = st.text_input("What should I analyze?", "What are the key patterns in this data?")

if st.button("Run Analysis", type="primary") and uploaded_file:
    df = pd.read_csv(uploaded_file)
    df_clean = df.copy()

    # ---------- STEP 1: CLEANING ----------
    st.subheader("Step 1: Automatic Cleaning")
    changes = []

    for col in df_clean.columns:
        # Boolean detection - improved
        unique_vals = df_clean[col].dropna().astype(str).str.lower().unique()
        if set(unique_vals) <= {'0','1','true','false','yes','no','y','n'}:
            mapping = {'true':1,'false':0,'yes':1,'no':0,'y':1,'n':0,'1':1,'0':0}
            df_clean[col] = df_clean[col].astype(str).str.lower().map(mapping)
            changes.append(f"Converted '{col}' to boolean (0/1)")
            continue
        # Date detection
        if 'date' in col.lower() or 'time' in col.lower():
            try:
                parsed = pd.to_datetime(df_clean[col], errors='coerce')
                if parsed.notna().sum() > len(df_clean)*0.5:
                    df_clean[col] = parsed
                    changes.append(f"Converted '{col}' to datetime")
                    continue
            except: pass

    numeric_cols = df_clean.select_dtypes(include=['number']).columns.tolist()
    for col in numeric_cols:
        if cap_outliers_toggle:
            before = df_clean[col].copy()
            df_clean[col] = cap_outliers_iqr(df_clean[col])
            if not before.equals(df_clean[col]):
                changes.append(f"Capped outliers in '{col}' using IQR")
        nulls = df_clean[col].isnull().sum()
        if nulls > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            changes.append(f"Filled {nulls} nulls in '{col}' with median")

    # Categorical cleaning
    cat_cols = [c for c in df_clean.select_dtypes('object').columns if df_clean[c].nunique() < 50]
    for col in cat_cols:
        df_clean[col] = df_clean[col].astype(str).str.strip().str.title()
        nulls = df_clean[col].isnull().sum()
        if nulls > 0:
            fill = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
            df_clean[col] = df_clean[col].fillna(fill)
            changes.append(f"Filled {nulls} nulls in '{col}' with '{fill}'")

    # Drop >70% missing
    for col in df.columns:
        if df[col].isnull().mean() > 0.7:
            df_clean.drop(columns=[col], inplace=True, errors='ignore')
            changes.append(f"Dropped '{col}' (>70% missing)")

    for c in changes[:8]: st.write(f"- {c}")
    st.write(f"Shape: {df.shape} → {df_clean.shape}")

    # ---------- STEP 2: DATA DICTIONARY ----------
    if show_dict:
        st.subheader("Data Dictionary")
        data_dict = pd.DataFrame({
            'Column': df_clean.columns,
            'Type': df_clean.dtypes.astype(str),
            'Inferred Unit': [infer_unit(c) for c in df_clean.columns],
            'Example': [df_clean[c].iloc[0] if len(df_clean) > 0 else '' for c in df_clean.columns],
            'Missing %': (df_clean.isnull().mean()*100).round(1)
        })
        st.dataframe(data_dict, use_container_width=True)

    # ---------- STEP 3: CHARTS ----------
    st.subheader("Step 2: Auto Charts")
    figures = []
    numeric_cols = df_clean.select_dtypes(include=['number']).columns.tolist()
    datetime_cols = df_clean.select_dtypes(include=['datetime']).columns.tolist()
    cat_cols = [c for c in df_clean.select_dtypes('object').columns if 2 <= df_clean[c].nunique() <= 50]

    best_num = pick_best_numeric(df_clean, question)
    best_cat = pick_best_categorical(df_clean)

    # Chart 1: Distribution
    if best_num:
        series = df_clean[best_num].dropna()
        fig1, ax1 = plt.subplots(figsize=(8,5))
        if series.nunique() == 2:
            counts = series.value_counts().sort_index()
            counts.plot(kind='bar', ax=ax1, color='#1f77b4', edgecolor='black')
            ax1.set_xticklabels(['No','Yes'], rotation=0)
            ax1.set_title(f'{best_num}: {counts.get(1,0)} Yes ({counts.get(1,0)/len(series):.0%})', fontweight='bold')
        else:
            series.hist(ax=ax1, bins=min(30, series.nunique()), edgecolor='black', color='steelblue', alpha=0.8)
            ax1.axvline(series.mean(), color='red', linestyle='--', label=f"Mean: {series.mean():.1f}")
            ax1.axvline(series.median(), color='green', linestyle='-.', label=f"Median: {series.median():.1f}")
            ax1.legend()
            ax1.set_title(f'Distribution of {best_num}', fontweight='bold')
        ax1.set_xlabel(format_axis_label(best_num, series))
        ax1.set_ylabel('Frequency')
        ax1.text(0.95,0.95, f"Skew: {series.skew():.2f} | Std: {series.std():.1f}", transform=ax1.transAxes,
                 ha='right', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)
        plt.tight_layout()
        st.pyplot(fig1)
        st.caption(f"Insight: 75% of values ≤ {series.quantile(0.75):.1f}")
        figures.append((f"dist_{best_num}.png", fig1))

    # Chart 2: Bar
    if best_cat:
        counts = df_clean[best_cat].value_counts().head(10)
        fig2, ax2 = plt.subplots(figsize=(8,5))
        counts.plot(kind='bar', ax=ax2, color='skyblue', edgecolor='black')
        ax2.set_title(f'Top {len(counts)} {best_cat}', fontweight='bold')
        ax2.set_xlabel(best_cat)
        ax2.set_ylabel(f'Number of Records (Total: {len(df_clean):,})')
        for i,v in enumerate(counts):
            ax2.text(i, v*1.01, f'{v}\n({v/len(df_clean):.0%})', ha='center', va='bottom', fontsize=9)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig2)
        st.caption(f"Insight: '{counts.index[0]}' dominates with {counts.iloc[0]/len(df_clean):.0%}")
        figures.append((f"bar_{best_cat}.png", fig2))

    # Chart 3: Correlation
    if len(numeric_cols) >= 2:
        top_vars = df_clean[numeric_cols].var().nlargest(min(10, len(numeric_cols))).index.tolist()
        corr = df_clean[top_vars].corr()
        fig3, ax3 = plt.subplots(figsize=(9,7))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax3, square=True)
        ax3.set_title('Correlation Matrix (Top Variables)', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig3)
        figures.append(("correlation.png", fig3))

    # Chart 4: Time series - FIXED resample
    if datetime_cols and best_num:
        date_col = datetime_cols[0]
        df_temp = df_clean.set_index(date_col).resample('1D').mean(numeric_only=True).reset_index()
        fig4, ax4 = plt.subplots(figsize=(10,4))
        ax4.plot(df_temp[date_col], df_temp[best_num], alpha=0.5, label='Daily')
        rolling = df_temp[best_num].rolling(7, center=True).mean()
        ax4.plot(df_temp[date_col], rolling, color='red', linewidth=2, label='7-day trend')
        ax4.set_title(f'{best_num} over time', fontweight='bold')
        ax4.set_ylabel(format_axis_label(best_num, df_temp[best_num]))
        ax4.legend()
        max_idx = df_temp[best_num].idxmax()
        ax4.annotate(f"Peak: {df_temp.loc[max_idx, best_num]:.0f}",
                     xy=(df_temp.loc[max_idx, date_col], df_temp.loc[max_idx, best_num]),
                     xytext=(0,15), textcoords='offset points', ha='center',
                     arrowprops=dict(arrowstyle='->'))
        plt.tight_layout()
        st.pyplot(fig4)
        figures.append((f"timeseries.png", fig4))

    # ---------- STEP 4: AI SUMMARY WITH SAMPLE ROWS ----------
    st.subheader("Step 3: AI Summary")
    stats_parts = [f"{len(df_clean)} rows, {len(df_clean.columns)} cols"]
    if best_num: stats_parts.append(f"'{best_num}' mean {df_clean[best_num].mean():.1f}")
    if best_cat:
        top = df_clean[best_cat].value_counts().iloc[0]
        stats_parts.append(f"top {best_cat} = {df_clean[best_cat].mode()[0]} ({top/len(df_clean):.0%})")
    stats = ". ".join(stats_parts) + "."

    sample_text = df_clean.head(3).to_csv(index=False) # CSV is cleaner for LLM

    prompt = f"""Question: {question}
Data: {stats}
Sample rows:
{sample_text}

Write 5 bullet points for a manager. Use numbers. End with 'Recommendation:' and one concrete action."""

    try:
        summary = model.generate_content(prompt).text
    except exceptions.ResourceExhausted:
        summary = f"- {stats_parts[0]}\n- Gemini limit hit. Wait 60s.\n- Recommendation: Review charts above."
        st.warning("Using fallback - rate limit")
    except Exception as e:
        summary = f"Error: {e}"
    st.markdown(summary)

    # ---------- DOWNLOADS ----------
    st.subheader("Downloads")
    report = f"# Report\nFile: {uploaded_file.name}\nDate: {datetime.now():%Y-%m-%d %H:%M}\n\n## Summary\n{summary}\n\n## Cleaning\n" + "\n".join(f"- {c}" for c in changes)
    st.download_button("Download report.md", report.encode(), f"report_{uploaded_file.name.split('.')[0]}.md")
    if figures:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf,'w') as zf:
            for name,fig in figures:
                b = io.BytesIO(); fig.savefig(b, format='png', dpi=150, bbox_inches='tight'); zf.writestr(name, b.getvalue())
        st.download_button("Download charts.zip", buf.getvalue(), "charts.zip")
    st.success("✅ Analysis complete!")
