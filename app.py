keep a fallback when no poster is found
Merged app.py

Replace your current file with this:

import os
import sys
from typing import Optional, Dict, Any

import pandas as pd
import requests
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from collaborative_filtering import HybridRecommender, load_processed_artifacts


# =========================
# PATH / DATA LOADING
# =========================
BASE_DIR = os.environ.get("MOVIE_RECOMMENDER_BASE_DIR", PROJECT_ROOT)

train, val, test, movies_clean, genre_encoded = load_processed_artifacts(BASE_DIR)

model = HybridRecommender(
    train_df=train,
    movies_df=movies_clean,
    genre_df=genre_encoded,
    neighbor_k=15,
    knn_neighbors=30,
    min_similarity=0.10,
    min_rating_for_profile=4.0,
    genre_weight=0.35,
    popularity_weight=0.15,
    min_candidate_rating_count=3,
    confidence_scaling=True,
).fit()

app = FastAPI(title="Movie Recommender Demo")


# =========================
# TMDB CONFIG
# =========================
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "")
TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"


# =========================
# HELPERS
# =========================
def clean_movie_title_for_search(title: str) -> str:
    """
    Clean titles like 'Toy Story (1995)' -> 'Toy Story'
    """
    if not isinstance(title, str):
        return ""
    if title.endswith(")") and "(" in title:
        return title.rsplit("(", 1)[0].strip()
    return title.strip()


def get_tmdb_movie(title: str) -> Optional[Dict[str, Any]]:
    """
    Search TMDb by title and return the first result.
    Safe fallback to None if API key is missing or request fails.
    """
    if not TMDB_API_KEY:
        return None

    search_title = clean_movie_title_for_search(title)
    if not search_title:
        return None

    params = {
        "api_key": TMDB_API_KEY,
        "query": search_title,
        "include_adult": False,
        "language": "en-US",
        "page": 1,
    }

    try:
        response = requests.get(TMDB_SEARCH_URL, params=params, timeout=8)
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])
        if not results:
            return None
        return results[0]
    except Exception:
        return None


def get_poster_url(title: str) -> Optional[str]:
    movie = get_tmdb_movie(title)
    if not movie:
        return None

    poster_path = movie.get("poster_path")
    if not poster_path:
        return None

    return f"{TMDB_IMAGE_BASE_URL}{poster_path}"


def html_escape(value: Any) -> str:
    """
    Minimal HTML escaping for safe rendering.
    """
    if value is None:
        return ""
    text = str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def build_movie_card(row: pd.Series, user_id: int) -> str:
    title = row.get("title", "Unknown Title")
    movie_id = row.get("movieId", "")
    genres = row.get("genres", "Unknown")
    poster_url = get_poster_url(title)

    hybrid_score = row.get("hybrid_score", row.get("score", "N/A"))
    cf_score = row.get("cf_score", "N/A")
    content_score = row.get("content_score", "N/A")
    popularity_score = row.get("popularity_score", "N/A")

    try:
        explanation_text = model.explain_recommendation_text(
            user_id=user_id,
            recommended_movie_id=int(movie_id)
        )
    except Exception:
        explanation_text = "Recommended based on your taste profile and similar user behavior."

    title_html = html_escape(title)
    genres_html = html_escape(genres)
    explanation_html = html_escape(explanation_text)

    if isinstance(hybrid_score, (int, float)):
        hybrid_score_html = f"{hybrid_score:.3f}"
    else:
        hybrid_score_html = html_escape(hybrid_score)

    if isinstance(cf_score, (int, float)):
        cf_score_html = f"{cf_score:.3f}"
    else:
        cf_score_html = html_escape(cf_score)

    if isinstance(content_score, (int, float)):
        content_score_html = f"{content_score:.3f}"
    else:
        content_score_html = html_escape(content_score)

    if isinstance(popularity_score, (int, float)):
        popularity_score_html = f"{popularity_score:.3f}"
    else:
        popularity_score_html = html_escape(popularity_score)

    explain_link = f"/explain?user_id={user_id}&movie_id={movie_id}"

    if poster_url:
        media_block = f"""
            <img class="movie-poster" src="{poster_url}" alt="{title_html} poster" loading="lazy">
        """
    else:
        media_block = f"""
            <div class="fallback-poster">
                <div class="fallback-title">{title_html}</div>
            </div>
        """

    return f"""
        <div class="movie-card">
            {media_block}
            <div class="movie-overlay">
                <div class="movie-meta">
                    <div class="movie-title">{title_html}</div>
                    <div class="movie-genres">{genres_html}</div>
                    <div class="movie-score">Hybrid score: {hybrid_score_html}</div>

                    <div class="score-row">
                        <span class="pill">CF: {cf_score_html}</span>
                        <span class="pill">Content: {content_score_html}</span>
                        <span class="pill">Popularity: {popularity_score_html}</span>
                    </div>

                    <div class="movie-reason">{explanation_html}</div>

                    <div class="card-actions">
                        <a class="action-link" href="{explain_link}" target="_blank">View explanation</a>
                    </div>
                </div>
            </div>
        </div>
    """


def build_demo_page(user_id: int, top_n: int, recs: pd.DataFrame) -> str:
    cards_html = "".join(build_movie_card(row, user_id) for _, row in recs.iterrows())

    tmdb_note = (
        "<p class='note ok'>Poster images enabled with TMDB_API_KEY.</p>"
        if TMDB_API_KEY
        else "<p class='note warn'>Poster images are off. Set the <code>TMDB_API_KEY</code> environment variable to enable posters.</p>"
    )

    return f"""
    <html>
        <head>
            <title>Movie Recommender Demo</title>
            <meta name="viewport" content="width=device-width, initial-scale=1.0" />
            <style>
                * {{
                    box-sizing: border-box;
                }}

                body {{
                    margin: 0;
                    font-family: Arial, sans-serif;
                    background: #0f1115;
                    color: #f5f7fa;
                }}

                .page {{
                    max-width: 1400px;
                    margin: 0 auto;
                    padding: 32px 24px 48px 24px;
                }}

                .header {{
                    margin-bottom: 24px;
                }}

                .title {{
                    font-size: 2.2rem;
                    font-weight: 800;
                    margin-bottom: 8px;
                }}

                .subtitle {{
                    color: #b8c0cc;
                    margin-bottom: 10px;
                }}

                .controls {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 12px;
                    margin: 18px 0 10px 0;
                }}

                .control-pill {{
                    background: #1b2230;
                    border: 1px solid #2b3548;
                    border-radius: 999px;
                    padding: 10px 14px;
                    color: #d9e2ef;
                    font-size: 0.95rem;
                }}

                .note {{
                    margin-top: 8px;
                    font-size: 0.95rem;
                }}

                .ok {{
                    color: #8fe388;
                }}

                .warn {{
                    color: #ffd166;
                }}

                .grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
                    gap: 22px;
                    margin-top: 26px;
                }}

                .movie-card {{
                    position: relative;
                    border-radius: 18px;
                    overflow: hidden;
                    background: #161b22;
                    min-height: 380px;
                    box-shadow: 0 10px 24px rgba(0, 0, 0, 0.35);
                    transition: transform 0.22s ease, box-shadow 0.22s ease;
                }}

                .movie-card:hover {{
                    transform: translateY(-6px) scale(1.02);
                    box-shadow: 0 18px 38px rgba(0, 0, 0, 0.45);
                }}

                .movie-poster {{
                    width: 100%;
                    height: 100%;
                    min-height: 380px;
                    object-fit: cover;
                    display: block;
                    background: #111;
                }}

                .fallback-poster {{
                    min-height: 380px;
                    height: 100%;
                    display: flex;
                    align-items: flex-end;
                    justify-content: flex-start;
                    padding: 18px;
                    background: linear-gradient(135deg, #1f2430, #343d4f);
                }}

                .fallback-title {{
                    font-size: 1.1rem;
                    font-weight: 700;
                    color: white;
                }}

                .movie-overlay {{
                    position: absolute;
                    inset: 0;
                    display: flex;
                    align-items: flex-end;
                    padding: 16px;
                    background: linear-gradient(
                        to top,
                        rgba(0,0,0,0.96) 8%,
                        rgba(0,0,0,0.76) 45%,
                        rgba(0,0,0,0.10) 100%
                    );
                    opacity: 0;
                    transition: opacity 0.22s ease;
                }}

                .movie-card:hover .movie-overlay {{
                    opacity: 1;
                }}

                .movie-meta {{
                    width: 100%;
                }}

                .movie-title {{
                    font-size: 1.05rem;
                    font-weight: 800;
                    margin-bottom: 6px;
                    line-height: 1.2;
                }}

                .movie-genres {{
                    font-size: 0.86rem;
                    color: #d2d8e2;
                    margin-bottom: 8px;
                }}

                .movie-score {{
                    font-size: 0.93rem;
                    color: #ffd166;
                    font-weight: 700;
                    margin-bottom: 8px;
                }}

                .score-row {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 6px;
                    margin-bottom: 10px;
                }}

                .pill {{
                    display: inline-block;
                    border-radius: 999px;
                    padding: 5px 9px;
                    background: rgba(255,255,255,0.10);
                    font-size: 0.78rem;
                }}

                .movie-reason {{
                    font-size: 0.84rem;
                    color: #edf1f7;
                    line-height: 1.35;
                    margin-bottom: 12px;
                    max-height: 7.2em;
                    overflow: hidden;
                }}

                .card-actions {{
                    margin-top: auto;
                }}

                .action-link {{
                    display: inline-block;
                    text-decoration: none;
                    color: white;
                    background: #2563eb;
                    padding: 8px 12px;
                    border-radius: 10px;
                    font-size: 0.85rem;
                    font-weight: 700;
                }}

                .action-link:hover {{
                    background: #1d4ed8;
                }}

                .footer-links {{
                    margin-top: 28px;
                    display: flex;
                    flex-wrap: wrap;
                    gap: 12px;
                }}

                .footer-links a {{
                    color: #9ec1ff;
                    text-decoration: none;
                }}

                .footer-links a:hover {{
                    text-decoration: underline;
                }}

                @media (max-width: 768px) {{
                    .title {{
                        font-size: 1.7rem;
                    }}

                    .grid {{
                        grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
                        gap: 16px;
                    }}

                    .movie-card,
                    .movie-poster,
                    .fallback-poster {{
                        min-height: 300px;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="page">
                <div class="header">
                    <div class="title">🎬 Movie Recommender Demo</div>
                    <div class="subtitle">Hover over a poster card to see why it was recommended.</div>

                    <div class="controls">
                        <div class="control-pill"><strong>User ID:</strong> {user_id}</div>
                        <div class="control-pill"><strong>Top N:</strong> {top_n}</div>
                        <div class="control-pill"><strong>Total Results:</strong> {len(recs)}</div>
                    </div>

                    {tmdb_note}
                </div>

                <div class="grid">
                    {cards_html}
                </div>

                <div class="footer-links">
                    <a href="/recommend?user_id={user_id}&top_n={top_n}" target="_blank">JSON recommendations</a>
                    <a href="/health" target="_blank">Health check</a>
                </div>
            </div>
        </body>
    </html>
    """


# =========================
# ROUTES
# =========================
@app.get("/")
def home():
    return {
        "message": "Movie recommender API is running.",
        "routes": {
            "health": "/health",
            "recommend": "/recommend?user_id=1&top_n=10",
            "similar_movies": "/similar-movies?movie_title=Toy%20Story&top_n=10",
            "explain": "/explain?user_id=1&movie_id=1252",
            "demo": "/demo?user_id=1&top_n=10",
        },
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "train_rows": int(train.shape[0]),
        "val_rows": int(val.shape[0]),
        "test_rows": int(test.shape[0]),
        "movie_rows": int(movies_clean.shape[0]),
        "tmdb_enabled": bool(TMDB_API_KEY),
    }


@app.get("/recommend")
def recommend(user_id: int = Query(...), top_n: int = Query(10, ge=1, le=50)):
    recs = model.recommend(user_id=user_id, top_n=top_n)
    return {
        "user_id": user_id,
        "top_n": top_n,
        "recommendations": recs.to_dict(orient="records"),
    }


@app.get("/similar-movies")
def similar_movies(movie_title: str = Query(...), top_n: int = Query(10, ge=1, le=50)):
    matches = movies_clean[
        movies_clean["title"].str.contains(movie_title, case=False, na=False)
    ][["movieId", "title", "genres"]].drop_duplicates()

    if matches.empty:
        return {"message": f"No movies found for title search: {movie_title}"}

    movie_id = int(matches.iloc[0]["movieId"])
    neighbors = model.get_similar_movies(movie_id=movie_id, n_neighbors=top_n)

    return {
        "query": movie_title,
        "matched_movie": matches.iloc[0].to_dict(),
        "similar_movies": neighbors.head(top_n).to_dict(orient="records"),
    }


@app.get("/explain")
def explain(user_id: int = Query(...), movie_id: int = Query(...)):
    explanation_df = model.explain_recommendation(user_id=user_id, recommended_movie_id=movie_id)
    explanation_text = model.explain_recommendation_text(user_id=user_id, recommended_movie_id=movie_id)

    return {
        "user_id": user_id,
        "movie_id": movie_id,
        "explanation_text": explanation_text,
        "details": explanation_df.to_dict(orient="records"),
    }


@app.get("/demo", response_class=HTMLResponse)
def demo(user_id: int = Query(1), top_n: int = Query(10, ge=1, le=20)):
    recs = model.recommend(user_id=user_id, top_n=top_n).copy()

    if recs.empty:
        return HTMLResponse("<h2>No recommendations found.</h2>")

    html = build_demo_page(user_id=user_id, top_n=top_n, recs=recs)
    return HTMLResponse(content=html)


# Run locally with:
# uvicorn app:app --reload --host 0.0.0.0 --port 8000
