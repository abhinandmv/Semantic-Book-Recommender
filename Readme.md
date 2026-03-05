# 📚Semantic Book Recommender

A semantic book recommendation app that lets you find books by describing what you're looking for in natural language. Filter by category and emotional tone to narrow down results.

Built with Streamlit, LangChain, ChromaDB, and HuggingFace sentence embeddings.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Features

- **Semantic search** — describe a book in plain English and get relevant matches
- **Category filter** — narrow results by genre or subject
- **Emotional tone filter** — sort by joy, surprise, anger, fear, or sadness scores
- **Book detail modal** — click any result to see the full description

---

## How it works

1. Book descriptions from `books_with_emotions.csv` are embedded using the `all-MiniLM-L6-v2` sentence transformer model
2. Embeddings are stored in a ChromaDB vector store built from `tagged_description.txt`
3. When you search, your query is embedded and matched against the vector store
4. Results are re-ranked by category and emotional tone based on your filters

---

## Project Structure

```
├── app.py                        # Main Streamlit app
├── books_with_emotions.csv       # Book dataset with emotion scores
├── tagged_description.txt        # ISBN-tagged descriptions for embedding
├── requirements.txt              # Python dependencies
└── .gitignore
```

---

## Running Locally

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

> On first run, the HuggingFace embedding model (~90 MB) will be downloaded and the ChromaDB vector store will be built. This takes 2–3 minutes. Subsequent runs are fast.

---

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app**
4. Select your repo, branch (`main`), and set the main file to `app.py`
5. Click **Deploy**

---

## Tech Stack

- [Streamlit](https://streamlit.io) — UI framework
- [LangChain](https://langchain.com) — document loading and text splitting
- [ChromaDB](https://www.trychroma.com) — vector store
- [HuggingFace Sentence Transformers](https://www.sbert.net) — `all-MiniLM-L6-v2` embeddings
- [Pandas](https://pandas.pydata.org) — data handling