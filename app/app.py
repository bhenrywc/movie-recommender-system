import os
import sys
import pandas as pd
import streamlit as st

APP_DIR = os.path.dirname(__file__)
BASE_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))
SRC_DIR = os.path.join(BASE_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from config import (
    MODEL_PATH,
    USER_RATINGS_PATH,
    RECOMMENDER_PARAMS,
    DEFAULT_TOP_N,
)
from data_loader import load_processed_data
from recommender import HybridRecommender
from user_profiles import UserProfileStore
from search_engine import MovieSearchEngine
from chatbot import MovieChatbot
from metrics import compare_models
from poster_utils import get_poster_url


st.set_page_config(
    page_title="Movie Recommender System 3.0",
    page_icon="🎬",
    layout="wide",
)


@st.cache_data
def cached_data():
    return load_processed_data()


@st.cache_resource
def cached_model(train, movies, genres):
    if os.path.exists(MODEL_PATH):
        try:
            return HybridRecommender.load(MODEL_PATH)
        except Exception:
            pass

    model = HybridRecommender(
        train_df=train,
        movies_df=movies,
        genre_df=genres,
        **RECOMMENDER_PARAMS,
    ).fit()

    model.save(MODEL_PATH)
    return model


def show_movie_table(df, score_col=None, limit=20):
    if df is None or df.empty:
        st.info("No movies found.")
        return

    show_cols = ["movieId", "title", "genres"]
    optional = ["rating_count", "weighted_score", "similarity", "mf_score", "cf_score", "hybrid_score", "final_score"]

    for col in optional:
        if col in df.columns and col not in show_cols:
            show_cols.append(col)

    if score_col and score_col in df.columns and score_col not in show_cols:
        show_cols.append(score_col)

    st.dataframe(df[show_cols].head(limit), use_container_width=True)


def show_movie_cards(df, score_col=None):
    if df is None or df.empty:
        st.info("No movies found.")
        return

    for _, row in df.iterrows():
        with st.container(border=True):
            cols = st.columns([1, 5, 2])

            poster_url = get_poster_url(row)

            with cols[0]:
                if poster_url:
                    st.image(poster_url, width=95)
                else:
                    st.write("🎬")

            with cols[1]:
                st.subheader(row.get("title", "Unknown Title"))
                st.caption(row.get("genres", ""))
                st.write(f"Movie ID: `{row.get('movieId')}`")

            with cols[2]:
                if score_col and score_col in row:
                    st.metric(score_col, round(float(row[score_col]), 4))
                if "rating_count" in row:
                    st.write(f"Ratings: **{int(row['rating_count'])}**")


def main():
    st.title("🎬 Movie Recommender System 3.0")
    st.caption("Hybrid ML recommender + profiles + chatbot + search + metrics dashboard")

    try:
        train, val, test, movies, genres = cached_data()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    model = cached_model(train, movies, genres)
    search_engine = MovieSearchEngine(movies)
    chatbot = MovieChatbot(model=model, movies_df=movies)
    profile_store = UserProfileStore(USER_RATINGS_PATH)

    with st.sidebar:
        st.header("👤 Profile")
        users = profile_store.get_users()

        default_user = users[0] if users else "brandon"
        username = st.text_input("Username", value=default_user)

        st.divider()
        st.header("⚙️ Settings")
        top_n = st.slider("Number of recommendations", 5, 25, DEFAULT_TOP_N, 1)
        display_mode = st.radio("Display", ["Cards", "Table"], horizontal=True)

        st.divider()
        st.header("📊 Dataset")
        st.write(f"Movies: **{movies.shape[0]:,}**")
        st.write(f"Train ratings: **{train.shape[0]:,}**")
        st.write(f"Users: **{train['userId'].nunique():,}**")

    tabs = st.tabs(
        [
            "🏠 Home",
            "🔎 Search",
            "🎞 Similar",
            "⭐ Profile",
            "🎯 My Recs",
            "💬 Chatbox",
            "📊 Metrics",
            "🧠 Interview Notes",
        ]
    )

    with tabs[0]:
        st.header("Popular Movies")
        recs = model._popularity_fallback(top_n=top_n)

        if display_mode == "Cards":
            show_movie_cards(recs, score_col="weighted_score")
        else:
            show_movie_table(recs, score_col="weighted_score")

    with tabs[1]:
        st.header("Advanced Search")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            title_query = st.text_input("Title contains", "")

        with col2:
            genre = st.selectbox("Genre", ["All"] + search_engine.all_genres())

        with col3:
            min_year = st.text_input("Min year", "")

        with col4:
            max_year = st.text_input("Max year", "")

        results = search_engine.search(
            title_query=title_query,
            genre=genre,
            min_year=min_year,
            max_year=max_year,
            limit=100,
        )

        st.write(f"Results: **{len(results):,}**")

        if display_mode == "Cards":
            show_movie_cards(results.head(30))
        else:
            show_movie_table(results, limit=100)

    with tabs[2]:
        st.header("Similar Movie Finder")

        query = st.text_input("Search a movie", "Toy Story")
        matches = search_engine.title_matches(query, limit=20)

        if matches.empty:
            st.info("No movie found.")
        else:
            selected = st.selectbox("Choose a movie", matches["title"].tolist())
            movie_id = int(matches.loc[matches["title"] == selected, "movieId"].iloc[0])

            similar = model.get_similar_movies(movie_id, n_neighbors=top_n)

            if display_mode == "Cards":
                show_movie_cards(similar, score_col="similarity")
            else:
                show_movie_table(similar, score_col="similarity")

    with tabs[3]:
        st.header("Create Your Movie Profile")

        st.write(f"Current profile: **{username}**")

        query = st.text_input("Search movie to rate", "Matrix")
        matches = search_engine.title_matches(query, limit=25)

        if not matches.empty:
            selected = st.selectbox("Movie", matches["title"].tolist())
            movie_id = int(matches.loc[matches["title"] == selected, "movieId"].iloc[0])
            rating = st.slider("Your rating", 0.5, 5.0, 4.0, 0.5)

            col_a, col_b = st.columns(2)

            with col_a:
                if st.button("Save rating"):
                    profile_store.add_or_update_rating(username, movie_id, selected, rating)
                    st.success(f"Saved {selected}: {rating}")

            with col_b:
                if st.button("Delete this rating"):
                    profile_store.delete_rating(username, movie_id)
                    st.warning(f"Deleted rating for {selected}")

        user_ratings = profile_store.get_user_ratings(username)

        st.subheader("Your Ratings")
        st.dataframe(user_ratings, use_container_width=True)

        if not user_ratings.empty:
            avg_rating = user_ratings["rating"].mean()
            st.metric("Your average rating", round(avg_rating, 2))

            if st.button("Clear this profile"):
                profile_store.clear_user(username)
                st.warning("Profile cleared. Refresh the app if the table does not update immediately.")

    with tabs[4]:
        st.header("Recommendations from Your Ratings")

        user_ratings = profile_store.get_user_ratings(username)

        if user_ratings.empty:
            st.info("Rate a few movies first. Showing popularity fallback for now.")
            recs = model._popularity_fallback(top_n=top_n)
            score_col = "weighted_score"
        else:
            recs = model.recommend_from_custom_ratings(
                user_ratings[["movieId", "rating"]],
                top_n=top_n,
            )
            score_col = "final_score"

        if display_mode == "Cards":
            show_movie_cards(recs, score_col=score_col)
        else:
            show_movie_table(recs, score_col=score_col)

    with tabs[5]:
        st.header("Movie Chatbox")
        st.write("Try: `Recommend funny action movies from the 90s`, `best horror movies`, or `romantic comedy`.")

        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []

        prompt = st.chat_input("Ask for movie recommendations...")

        if prompt:
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            response, recs = chatbot.recommend(prompt, top_n=top_n)
            st.session_state.chat_messages.append(
                {"role": "assistant", "content": response, "results": recs}
            )

        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if "results" in msg:
                    show_movie_table(msg["results"], score_col="weighted_score", limit=top_n)

    with tabs[6]:
        st.header("Model Evaluation Dashboard")

        st.write("This compares Popularity, Item-Item CF, Matrix Factorization, and the Upgraded Hybrid recommender.")

        sample_users = st.slider("Sample users", 25, 200, 100, 25)
        min_rating = st.slider("Relevant rating threshold", 3.0, 5.0, 4.0, 0.5)

        if st.button("Run evaluation"):
            with st.spinner("Evaluating models..."):
                results = compare_models(
                    model=model,
                    val_df=val,
                    test_df=test,
                    top_n=top_n,
                    min_eval_rating=min_rating,
                    sample_users=sample_users,
                )

            st.dataframe(results, use_container_width=True)

            chart_df = results.pivot(index="model", columns="split", values="hit_rate@k")
            st.bar_chart(chart_df)

    with tabs[7]:
        st.header("Interview Talking Points")

        st.markdown(
            """
### Project Summary

This project is a production-style movie recommender system that combines:

- Matrix factorization using TruncatedSVD
- Item-item collaborative filtering
- Genre-based user taste profiles
- Popularity fallback for cold-start users
- Custom app user profiles
- Search and chatbot-style discovery
- Evaluation dashboard with hit rate, precision, recall, and coverage

### Strong Interview Explanation

> I built a hybrid recommender system that combines collaborative filtering, matrix factorization, content-based genre signals, and popularity fallback. I also added a user profile layer so new users can rate movies and immediately receive personalized recommendations.

### Why Hybrid?

A pure collaborative filtering model struggles with cold-start users and sparse ratings. A pure popularity model is too generic. My hybrid approach balances personalization, similarity, genre preference, and popularity.

### Metrics to Mention

- Hit Rate@K: whether at least one relevant item appears in the top K
- Precision@K: quality of the recommendation list
- Recall@K: how many relevant items were recovered
- Coverage@K: how much of the catalog the model exposes

### Future Improvements

- Add TMDB poster metadata
- Add movie overviews and cast/director features
- Add review sentiment from IMDb or TMDB reviews
- Add persistent database storage
- Deploy with Docker and Render
"""
        )


if __name__ == "__main__":
    main()
