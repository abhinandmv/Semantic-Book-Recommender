import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import streamlit as st

load_dotenv()

st.set_page_config(
    page_title="Semantic Book Recommender",
    page_icon="📚",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Lato:wght@300;400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Lato', sans-serif;
    background-color: #0f0e0c;
    color: #e8e0d0;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem; max-width: 1400px; }

.hero {
    text-align: center;
    padding: 3.5rem 0 2.5rem;
    border-bottom: 1px solid #2a2820;
    margin-bottom: 2.5rem;
}
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.4rem, 5vw, 4rem);
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #f5ead8;
    margin: 0 0 0.4rem;
}
.hero p {
    font-size: 1rem;
    color: #7a7060;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 0;
}

.stTextInput > div > div > input {
    background: #1a1914 !important;
    border: 1px solid #2e2c26 !important;
    border-radius: 6px !important;
    color: #e8e0d0 !important;
    font-family: 'Lato', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 0.65rem 1rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #c9a84c !important;
    box-shadow: 0 0 0 2px rgba(201,168,76,0.15) !important;
}
.stTextInput label, .stSelectbox label {
    color: #7a7060 !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    font-weight: 700 !important;
}
.stSelectbox > div > div {
    background: #1a1914 !important;
    border: 1px solid #2e2c26 !important;
    border-radius: 6px !important;
    color: #e8e0d0 !important;
}

.stButton > button {
    background: #1a1914 !important;
    color: #c9a84c !important;
    border: 1px solid #2e2c26 !important;
    border-radius: 0 0 6px 6px !important;
    font-family: 'Lato', sans-serif !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 0.3rem 0.5rem !important;
    width: 100% !important;
    transition: background 0.18s, color 0.18s;
}
.stButton > button:hover {
    background: #c9a84c !important;
    color: #0f0e0c !important;
}

.section-label {
    font-size: 0.7rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #7a7060;
    margin-bottom: 1.2rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #2a2820;
}

.book-card {
    border-radius: 6px 6px 0 0;
    overflow: hidden;
    background: #1a1914;
    border: 1px solid #2a2820;
    border-bottom: none;
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s;
}
.book-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 14px 36px rgba(0,0,0,0.55);
    border-color: #c9a84c;
}
.book-card img { width: 100%; aspect-ratio: 2/3; object-fit: cover; display: block; }
.card-title {
    padding: 0.5rem 0.55rem 0.4rem;
    font-family: 'Playfair Display', serif;
    font-size: 0.75rem;
    line-height: 1.35;
    color: #f0e6d0;
    font-weight: 700;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
    min-height: 2.4em;
    background: #1a1914;
    border-left: 1px solid #2a2820;
    border-right: 1px solid #2a2820;
}

/* Style the native st.dialog */
div[data-testid="stDialog"] > div > div {
    background: #1a1914 !important;
    border: 1px solid #3a3628 !important;
    border-radius: 12px !important;
    padding: 0 !important;
    max-width: 680px !important;
}
div[data-testid="stDialog"] button[aria-label="Close"] {
    color: #7a7060 !important;
    background: transparent !important;
}
div[data-testid="stDialog"] button[aria-label="Close"]:hover {
    color: #f5ead8 !important;
}

/* Close button inside dialog */
div[data-testid="stDialog"] .stButton > button {
    background: #c9a84c !important;
    color: #0f0e0c !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 700 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.08em !important;
    padding: 0.5rem 1.5rem !important;
    width: auto !important;
}
div[data-testid="stDialog"] .stButton > button:hover {
    background: #e0bc5a !important;
    color: #0f0e0c !important;
}
</style>
""", unsafe_allow_html=True)


# ── Data & model ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_data_and_model():
    books = pd.read_csv("books_with_emotions.csv")
    books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
    books["large_thumbnail"] = np.where(
        books["large_thumbnail"].isna(), "cover-not-found.jpg", books["large_thumbnail"]
    )
    raw_documents = TextLoader("tagged_description.txt").load()
    text_splitter = CharacterTextSplitter(chunk_size=1, chunk_overlap=0, separator="\n")
    documents = text_splitter.split_documents(raw_documents)
    local_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db_books = Chroma.from_documents(
        documents, embedding=local_embeddings, persist_directory="./chroma_db"
    )
    return books, db_books

books, db_books = load_data_and_model()


# ── Core logic (unchanged) ────────────────────────────────────────────────────
def retrieve_semantic_recommendations(
        query: str, category: str = None, tone: str = None,
        initial_top_k: int = 50, final_top_k: int = 16,
) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":        book_recs.sort_values(by="joy",      ascending=False, inplace=True)
    elif tone == "Surprising": book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":      book_recs.sort_values(by="anger",    ascending=False, inplace=True)
    elif tone == "Suspenseful":book_recs.sort_values(by="fear",     ascending=False, inplace=True)
    elif tone == "Sad":        book_recs.sort_values(by="sadness",  ascending=False, inplace=True)
    return book_recs


def recommend_books(query: str, category: str, tone: str):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []
    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_description = " ".join(description.split()[:30]) + "..."
        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]
        results.append({
            "thumbnail": row["large_thumbnail"],
            "title": row["title"],
            "authors": authors_str,
            "description": description,
            "truncated": truncated_description,
        })
    return results


# ── Native Streamlit dialog (real modal with working close) ───────────────────
@st.dialog("Book Details", width="large")
def show_book_dialog(book: dict):
    col_img, col_text = st.columns([1, 2.5])
    with col_img:
        st.image(book["thumbnail"], use_container_width=True)
    with col_text:
        st.markdown(f"""
        <div style="font-family:'Playfair Display',serif; font-size:1.4rem; font-weight:700;
                    color:#f5ead8; line-height:1.3; margin-bottom:0.3rem;">
            {book['title']}
        </div>
        <div style="font-size:0.85rem; color:#c9a84c; font-style:italic; margin-bottom:1rem;">
            {book['authors']}
        </div>
        <hr style="border:none; border-top:1px solid #2a2820; margin-bottom:1rem;">
        <div style="font-size:0.88rem; line-height:1.75; color:#a09080;">
            {book['description']}
        </div>
        """, unsafe_allow_html=True)
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    if st.button("✕  Close", key="dialog_close"):
        st.rerun()


# ── Session state ─────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = []
if "selected" not in st.session_state:
    st.session_state.selected = None

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>Semantic Book Recommendations</h1>
</div>
""", unsafe_allow_html=True)

# ── Search controls ───────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns([4, 2, 2, 1.2])
with c1:
    user_query = st.text_input("What kind of book?", placeholder="e.g., A story about forgiveness and redemption")
with c2:
    category = st.selectbox("Category", categories)
with c3:
    tone = st.selectbox("Emotional Tone", tones)
with c4:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    search = st.button("Search", use_container_width=True)

if search:
    if user_query.strip():
        with st.spinner("Searching the shelves…"):
            st.session_state.results = recommend_books(user_query, category, tone)
            st.session_state.selected = None
    else:
        st.warning("Please describe the kind of book you're looking for.")

# ── Trigger dialog when a book is selected ────────────────────────────────────
if st.session_state.selected:
    show_book_dialog(st.session_state.selected)
    st.session_state.selected = None

# ── Results grid ──────────────────────────────────────────────────────────────
if st.session_state.results:
    st.markdown(
        f"<div class='section-label'>{len(st.session_state.results)} Recommendations</div>",
        unsafe_allow_html=True,
    )

    COLS = 8
    results = st.session_state.results
    rows = [results[i:i + COLS] for i in range(0, len(results), COLS)]

    for row in rows:
        cols = st.columns(COLS)
        for col, book in zip(cols, row):
            with col:
                safe_title = book['title'].replace('"', '&quot;').replace("'", "&#39;")
                st.markdown(f"""
                <div class="book-card">
                    <img src="{book['thumbnail']}" alt="{safe_title}"
                         onerror="this.src='cover-not-found.jpg'"/>
                </div>
                <div class="card-title">{safe_title}</div>
                """, unsafe_allow_html=True)
                if st.button("View Details", key=f"btn_{book['title'][:25]}_{id(book)}", use_container_width=True):
                    st.session_state.selected = book
                    st.rerun()