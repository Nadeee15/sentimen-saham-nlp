import streamlit as st
import joblib
import re
import pandas as pd
import matplotlib.pyplot as plt
from langdetect import detect

st.set_page_config(
    page_title="Analisis Sentimen Saham",
    page_icon="📈",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

.stApp {
    background: #0a0a0f;
    color: #f0f0f0;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
}

.main-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00ff87, #60efff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}

.subtitle {
    color: #888;
    font-size: 0.95rem;
    font-family: 'Space Mono', monospace;
    margin-top: 0.2rem;
    margin-bottom: 2rem;
}

.result-box {
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-top: 1.5rem;
    text-align: center;
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2rem;
    letter-spacing: 2px;
}

.result-positive {
    background: linear-gradient(135deg, #00ff8720, #00ff8740);
    border: 1px solid #00ff87;
    color: #00ff87;
}

.result-negative {
    background: linear-gradient(135deg, #ff006020, #ff006040);
    border: 1px solid #ff0060;
    color: #ff0060;
}

.result-neutral {
    background: linear-gradient(135deg, #60efff20, #60efff40);
    border: 1px solid #60efff;
    color: #60efff;
}

.info-box {
    background: #111118;
    border: 1px solid #222;
    border-radius: 10px;
    padding: 1rem 1.5rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: #666;
    margin-top: 1rem;
}

div[data-testid="stTextArea"] textarea {
    background: #111118 !important;
    color: #f0f0f0 !important;
    border: 1px solid #333 !important;
    border-radius: 10px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.9rem !important;
}

div[data-testid="stTextArea"] textarea:focus {
    border-color: #00ff87 !important;
    box-shadow: 0 0 0 2px #00ff8730 !important;
}

.stButton > button {
    background: linear-gradient(135deg, #00ff87, #60efff) !important;
    color: #0a0a0f !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 2rem !important;
    width: 100% !important;
    letter-spacing: 1px !important;
    transition: opacity 0.2s !important;
}

.stButton > button:hover {
    opacity: 0.85 !important;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Syne', sans-serif !important;
    color: #888 !important;
}

.stTabs [aria-selected="true"] {
    color: #00ff87 !important;
}

hr {
    border-color: #222 !important;
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

st.markdown('<div class="main-title">📈 Sentimen Saham</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">NLP Pipeline · Analisis Sentimen Teks Bahasa Indonesia</div>', unsafe_allow_html=True)
st.markdown("---")

tab1, tab2 = st.tabs(["🔍 Prediksi", "📊 Analisis Batch"])

with tab1:
    st.markdown("#### Masukkan teks untuk dianalisis")
    user_input = st.text_area(
        label="",
        placeholder="Contoh: Saham BBCA naik terus, bagus banget performanya!",
        height=130,
        label_visibility="collapsed"
    )

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
                    result = pipe.predict([cleaned])[0]

                    emoji_map = {"Positive": "🟢", "Negative": "🔴", "Neutral": "🔵"}
                    css_map = {"Positive": "result-positive", "Negative": "result-negative", "Neutral": "result-neutral"}

                    st.markdown(
                        f'<div class="result-box {css_map[result]}">{emoji_map[result]} {result.upper()}</div>',
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f'<div class="info-box">Teks bersih: <span style="color:#aaa">{cleaned}</span></div>',
                        unsafe_allow_html=True
                    )
            except Exception:
                st.error("❌ Tidak dapat mendeteksi bahasa. Coba lagi.")

with tab2:
    st.markdown("#### Upload file CSV")
    st.caption("File harus memiliki kolom `Sentence`")

    uploaded_file = st.file_uploader("", type=["csv"], label_visibility="collapsed")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if "Sentence" not in df.columns:
            st.error("❌ Kolom `Sentence` tidak ditemukan di file CSV.")
        else:
            df["text_clean"] = df["Sentence"].apply(clean_tweet)
            df["Predicted"] = pipe.predict(df["text_clean"])

            counts = df["Predicted"].value_counts()

            fig, ax = plt.subplots(figsize=(5, 4), facecolor="#0a0a0f")
            colors = ["#00ff87", "#ff0060", "#60efff"]
            wedges, texts, autotexts = ax.pie(
                counts.values,
                labels=counts.index,
                autopct="%1.1f%%",
                colors=colors[:len(counts)],
                startangle=140,
                textprops={"color": "#f0f0f0", "fontsize": 12}
            )
            for at in autotexts:
                at.set_color("#0a0a0f")
                at.set_fontweight("bold")
            ax.set_facecolor("#0a0a0f")
            st.pyplot(fig)

            st.markdown("#### Hasil Prediksi")
            st.dataframe(
                df[["Sentence", "Predicted"]].rename(columns={"Sentence": "Teks", "Predicted": "Sentimen"}),
                use_container_width=True,
                height=300
            )

            csv_out = df[["Sentence", "Predicted"]].to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Hasil CSV",
                data=csv_out,
                file_name="hasil_sentimen.csv",
                mime="text/csv"
            )

st.markdown("---")
st.markdown(
    '<div style="text-align:center; font-family: Space Mono, monospace; font-size:0.75rem; color:#444;">'
    'Model: TF-IDF + LinearSVC · Dataset: IDSMSA · Built with Streamlit'
    '</div>',
    unsafe_allow_html=True
)
