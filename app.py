import streamlit as st
import joblib
import re
import pandas as pd
import matplotlib.pyplot as plt
from langdetect import detect

st.set_page_config(
    page_title="Sentimen Saham",
    page_icon="📈",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;700;900&display=swap');

*, html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #080810;
    color: #e8e8f0;
}

.hero {
    padding: 2.5rem 0 1rem 0;
    margin-bottom: 0.5rem;
}

.hero-badge {
    display: inline-block;
    background: #ffffff0f;
    border: 1px solid #ffffff18;
    border-radius: 100px;
    padding: 0.3rem 1rem;
    font-size: 0.72rem;
    font-family: 'DM Mono', monospace;
    color: #888;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

.hero-title {
    font-size: 3.2rem;
    font-weight: 900;
    line-height: 1.1;
    letter-spacing: -2px;
    color: #fff;
    margin-bottom: 0.5rem;
}

.hero-title span {
    background: linear-gradient(90deg, #00e5a0, #00c8ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-sub {
    font-size: 0.9rem;
    color: #555;
    font-family: 'DM Mono', monospace;
    margin-bottom: 2rem;
}

.card {
    background: #0f0f1a;
    border: 1px solid #1a1a2e;
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 1.5rem;
}

.result-wrapper {
    display: flex;
    align-items: center;
    gap: 1.2rem;
    padding: 1.5rem 2rem;
    border-radius: 14px;
    margin-top: 1.2rem;
}

.result-positive { background: #00e5a008; border: 1px solid #00e5a030; }
.result-negative { background: #ff294408; border: 1px solid #ff294430; }
.result-neutral  { background: #00c8ff08; border: 1px solid #00c8ff30; }

.result-dot { width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }
.result-dot-positive { background: #00e5a0; }
.result-dot-negative { background: #ff2944; }
.result-dot-neutral  { background: #00c8ff; }

.result-label {
    font-size: 1.6rem;
    font-weight: 900;
    letter-spacing: 3px;
    text-transform: uppercase;
}

.result-label-positive { color: #00e5a0; }
.result-label-negative { color: #ff2944; }
.result-label-neutral  { color: #00c8ff; }

.clean-text-box {
    background: #080810;
    border: 1px solid #1a1a2e;
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #444;
    margin-top: 0.8rem;
    line-height: 1.6;
}

.clean-text-box span { color: #666; }

.stat-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1rem;
}

.stat-box {
    flex: 1;
    background: #080810;
    border: 1px solid #1a1a2e;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}

.stat-num { font-size: 1.6rem; font-weight: 900; color: #fff; }
.stat-label {
    font-size: 0.72rem;
    color: #444;
    font-family: 'DM Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.2rem;
}

.divider { border: none; border-top: 1px solid #1a1a2e; margin: 1.5rem 0; }

.footer {
    text-align: center;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #2a2a3a;
    padding: 2rem 0 1rem 0;
}

div[data-testid="stTextArea"] textarea {
    background: #080810 !important;
    color: #e8e8f0 !important;
    border: 1px solid #1a1a2e !important;
    border-radius: 12px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.88rem !important;
    line-height: 1.7 !important;
    padding: 1rem !important;
}

div[data-testid="stTextArea"] textarea:focus {
    border-color: #00e5a040 !important;
    box-shadow: 0 0 0 3px #00e5a010 !important;
}

.stButton > button {
    background: linear-gradient(135deg, #00e5a0, #00c8ff) !important;
    color: #080810 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 2rem !important;
    width: 100% !important;
    letter-spacing: 0.5px !important;
}

.stButton > button:hover { opacity: 0.85 !important; }

.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #1a1a2e !important;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    color: #444 !important;
    font-size: 0.88rem !important;
}

.stTabs [aria-selected="true"] {
    color: #00e5a0 !important;
    border-bottom: 2px solid #00e5a0 !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load("sentiment_pipe.joblib")

pipe = load_model()

def clean_tweet(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = text.replace("[username]", " ")
    text = text.replace("[url]", " ")
    text = text.replace("[hashtag]", " ")
    text = re.sub(r"@\w+", " ", text)
    text = text.replace("#", " ")
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

st.markdown("""
<div class="hero">
    <div class="hero-badge">📈 NLP Pipeline</div>
    <div class="hero-title">Sentimen<br><span>Saham</span></div>
    <div class="hero-sub">Analisis sentimen teks saham · Bahasa Indonesia</div>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["  Prediksi  ", "  Analisis Batch  "])

with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Masukkan teks untuk dianalisis**")
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    user_input = st.text_area(
        label="",
        placeholder="Contoh: Saham BBCA naik terus, bagus banget performanya!",
        height=140,
        label_visibility="collapsed"
    )

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    if st.button("Analisis Sentimen"):
        if user_input.strip() == "":
            st.warning("⚠️ Teks tidak boleh kosong.")
        else:
            try:
                lang = detect(user_input)
                if lang != "id":
                    st.error("❌ Harap masukkan teks dalam **Bahasa Indonesia**.")
                else:
                    cleaned = clean_tweet(user_input)
                    result  = pipe.predict([cleaned])[0]

                    css_map   = {"Positive": "result-positive",       "Negative": "result-negative",       "Neutral": "result-neutral"}
                    dot_map   = {"Positive": "result-dot-positive",   "Negative": "result-dot-negative",   "Neutral": "result-dot-neutral"}
                    label_map = {"Positive": "result-label-positive", "Negative": "result-label-negative", "Neutral": "result-label-neutral"}
                    text_map  = {"Positive": "↑ Positif",             "Negative": "↓ Negatif",             "Neutral": "→ Netral"}

                    st.markdown(f"""
                    <div class="result-wrapper {css_map[result]}">
                        <div class="result-dot {dot_map[result]}"></div>
                        <div class="result-label {label_map[result]}">{text_map[result]}</div>
                    </div>
                    <div class="clean-text-box">teks bersih → <span>{cleaned}</span></div>
                    """, unsafe_allow_html=True)
            except Exception:
                st.error("❌ Tidak dapat mendeteksi bahasa. Coba lagi.")

    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Upload file CSV** — harus memiliki kolom `Sentence`")
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["csv"], label_visibility="collapsed")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if "Sentence" not in df.columns:
            st.error("❌ Kolom `Sentence` tidak ditemukan.")
        else:
            df["text_clean"] = df["Sentence"].apply(clean_tweet)
            df["Predicted"]  = pipe.predict(df["text_clean"])
            counts = df["Predicted"].value_counts()

            pos = counts.get("Positive", 0)
            neg = counts.get("Negative", 0)
            neu = counts.get("Neutral",  0)

            st.markdown(f"""
            <div class="stat-row">
                <div class="stat-box">
                    <div class="stat-num" style="color:#00e5a0">{pos}</div>
                    <div class="stat-label">Positive</div>
                </div>
                <div class="stat-box">
                    <div class="stat-num" style="color:#ff2944">{neg}</div>
                    <div class="stat-label">Negative</div>
                </div>
                <div class="stat-box">
                    <div class="stat-num" style="color:#00c8ff">{neu}</div>
                    <div class="stat-label">Neutral</div>
                </div>
                <div class="stat-box">
                    <div class="stat-num">{len(df)}</div>
                    <div class="stat-label">Total</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(5, 4), facecolor="#080810")
            colors = ["#00e5a0", "#ff2944", "#00c8ff"]
            _, texts, autotexts = ax.pie(
                counts.values,
                labels=counts.index,
                autopct="%1.1f%%",
                colors=colors[:len(counts)],
                startangle=140,
                wedgeprops={"linewidth": 2, "edgecolor": "#080810"},
                textprops={"color": "#888", "fontsize": 11}
            )
            for at in autotexts:
                at.set_color("#080810")
                at.set_fontweight("bold")
            ax.set_facecolor("#080810")
            fig.patch.set_facecolor("#080810")
            st.pyplot(fig)

            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown("**Hasil Prediksi**")
            st.dataframe(
                df[["Sentence", "Predicted"]].rename(columns={"Sentence": "Teks", "Predicted": "Sentimen"}),
                use_container_width=True,
                height=280
            )

            csv_out = df[["Sentence", "Predicted"]].to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇ Download Hasil CSV",
                data=csv_out,
                file_name="hasil_sentimen.csv",
                mime="text/csv"
            )

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    TF-IDF + LinearSVC · Dataset IDSMSA · Streamlit Cloud
</div>
""", unsafe_allow_html=True)
