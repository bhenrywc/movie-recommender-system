
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR / "src"))

from recommender import HybridRecommender  # noqa: E402

st.set_page_config(page_title="Movie Recommender", layout="wide")

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
MODEL_PATH = ROOT_DIR / "models" / "hybrid_recommender.pkl"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"


@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return HybridRecommender.load(str(MODEL_PATH))

    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    movies = pd.read_csv(PROCESSED_DIR / "movies_clean.csv")
    genre = pd.read_csv(PROCESSED_DIR / "genre_encoded.csv")
    model = HybridRecommender(train_df=train, movies_df=movies, genre_df=genre).fit()
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_PATH))
    return model


def extract_year(title: str) -> Optional[str]:
    if not isinstance(title, str):
        return None
    if title.endswith(")") and "(" in title:
        return title[-5:-1]
    return None


@st.cache_data(show_spinner=False)
def fetch_poster(title: str):
    if not TMDB_API_KEY or not title:
        return None

    year = extract_year(title)
    clean_title = title.rsplit("(", 1)[0].strip()

    try:
        params = {"api_key": TMDB_API_KEY, "query": clean_title}
        if year and year.isdigit():
            params["year"] = year
        response = requests.get("https://api.themoviedb.org/3/search/movie", params=params, timeout=10)
        response.raise_for_status()
        results = response.json().get("results", [])
        if not results:
            return None
        poster_path = results[0].get("poster_path")
        overview = results[0].get("overview", "")
        release_date = results[0].get("release_date", "")
        vote_average = results[0].get("vote_average", None)
        if not poster_path:
            return {"overview": overview, "release_date": release_date, "vote_average": vote_average, "poster_url": None}
        return {
            "poster_url": f"https://image.tmdb.org/t/p/w500{poster_path}",
            "overview": overview,
            "release_date": release_date,
            "vote_average": vote_average,
        }
    except Exception:
        return None


def movie_card(row: pd.Series):
    details = fetch_poster(row["title"])
    poster_url = details.get("poster_url") if details else None
    overview = details.get("overview", "No overview found.") if details else "Add TMDB_API_KEY to show posters and plot summaries."
    release_date = details.get("release_date", "Unknown") if details else "Unknown"
    vote_average = details.get("vote_average", "N/A") if details else "N/A"

    st.markdown(
        f"""
        <div class="movie-card">
            <div class="movie-card-inner">
                <div class="poster-wrapper">
                    {"<img src='" + poster_url + "' class='poster' />" if poster_url else "<div class='poster placeholder'>No poster</div>"}
                </div>
                <div class="movie-meta">
                    <div class="movie-title">{row['title']}</div>
                    <div class="movie-sub">{row.get('genres', 'Unknown genres')}</div>
                    <div class="score-row">
                        <span>Hybrid: {row.get('hybrid_score', row.get('mf_score', row.get('weighted_score', 0))):.3f}</span>
                    </div>
                    <div class="hover-card">
                        <strong>Why this was recommended</strong><br/>
                        {row.get('reason', 'Popular fallback recommendation.')}<br/><br/>
                        <strong>Release date:</strong> {release_date}<br/>
                        <strong>TMDb rating:</strong> {vote_average}<br/><br/>
                        <strong>Overview:</strong> {overview}
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <style>
    .movie-card {
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 18px;
        padding: 12px;
        margin-bottom: 16px;
        background: rgba(255,255,255,0.03);
        position: relative;
        min-height: 180px;
    }
    .movie-card-inner {
        display: flex;
        gap: 14px;
        align-items: flex-start;
    }
    .poster-wrapper {
        flex: 0 0 110px;
    }
    .poster {
        width: 110px;
        border-radius: 12px;
    }
    .placeholder {
        width: 110px;
        height: 165px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 12px;
        background: rgba(255,255,255,0.08);
        font-size: 14px;
    }
    .movie-meta {
        position: relative;
        flex: 1;
    }
    .movie-title {
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .movie-sub {
        font-size: 0.9rem;
        opacity: 0.85;
        margin-bottom: 8px;
    }
    .score-row {
        font-size: 0.9rem;
        font-weight: 600;
    }
    .hover-card {
        display: none;
        position: absolute;
        top: 0;
        left: 0;
        width: min(420px, 95%);
        padding: 14px;
        border-radius: 14px;
        background: #111827;
        border: 1px solid rgba(255,255,255,0.12);
        box-shadow: 0 16px 36px rgba(0,0,0,0.35);
        z-index: 1000;
    }
    .movie-card:hover .hover-card {
        display: block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎬 Movie Recommender System")
st.caption("Matrix factorization + collaborative filtering + genre/profile reranking + popularity fallback")

model = load_model()

tab1, tab2, tab3 = st.tabs(["Recommend for User", "Find Similar Movies", "Top Popular"])

with tab1:
    user_id = st.number_input("User ID", min_value=1, step=1, value=1)
    top_n = st.slider("How many recommendations?", 5, 20, 10)

    if st.button("Get hybrid recommendations"):
        recs = model.recommend_hybrid(int(user_id), top_n=top_n)
        st.subheader("Recommendations")
        for _, row in recs.iterrows():
            movie_card(row)

with tab2:
    title_query = st.text_input("Search a movie title", value="Toy Story")
    if st.button("Find similar movies"):
        matches = model.find_movies_by_title(title_query, limit=10)
        if matches.empty:
            st.warning("No matches found.")
        else:
            picked = matches.iloc[0]
            st.write(f"Using: **{picked['title']}**")
            similar = model.get_similar_movies(int(picked["movieId"]), n_neighbors=10)
            for _, row in similar.iterrows():
                movie_card(row)

with tab3:
    st.subheader("Popularity fallback")
    popular = model.popularity_df.head(10)
    for _, row in popular.iterrows():
        movie_card(row)

st.divider()
st.markdown(
    """
    **Deployment notes**
    - Local run: `streamlit run app/app.py`
    - For posters, set `TMDB_API_KEY` in your environment or Streamlit secrets.
    """
)
