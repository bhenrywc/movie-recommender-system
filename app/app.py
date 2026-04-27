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
    optional = [
        "rating_count",
        "mean_rating",
        "weighted_score",
        "similarity",
        "mf_score",
        "cf_score",
        "hybrid_score",
        "final_score",
        "trend_score",
    ]

    for col in optional:
        if col in df.columns and col not in show_cols:
            show_cols.append(col)

    if score_col and score_col in df.columns and score_col not in show_cols:
        show_cols.append(score_col)

    existing_cols = [col for col in show_cols if col in df.columns]
    st.dataframe(df[existing_cols].head(limit), use_container_width=True)


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
                if "mean_rating" in row:
                    st.write(f"Avg Rating: **{round(float(row['mean_rating']), 2)}**")


def build_movie_stats(train, movies):
    stats = train.groupby("movieId").agg(
        rating_count=("rating", "count"),
        mean_rating=("rating", "mean"),
    ).reset_index()

    min_count = stats["rating_count"].min()
    max_count = stats["rating_count"].max()

    if max_count > min_count:
        stats["count_norm"] = (stats["rating_count"] - min_count) / (max_count - min_count)
    else:
        stats["count_norm"] = 0.0

    stats["trend_score"] = 0.65 * stats["count_norm"] + 0.35 * (stats["mean_rating"] / 5.0)

    if movies is not None and "movieId" in movies.columns:
        movie_cols = ["movieId"]
        if "title" in movies.columns:
            movie_cols.append("title")
        if "genres" in movies.columns:
            movie_cols.append("genres")

        stats = stats.merge(
            movies[movie_cols].drop_duplicates(subset=["movieId"]),
            on="movieId",
            how="left",
        )

    return stats.sort_values("trend_score", ascending=False).reset_index(drop=True)


def extract_year_from_title(title):
    if not isinstance(title, str):
        return None
    if "(" in title and ")" in title:
        possible = title.strip().split("(")[-1].replace(")", "")
        if possible.isdigit() and len(possible) == 4:
            return int(possible)
    return None


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
    movie_stats = build_movie_stats(train, movies)

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
            "📈 Analytics Dashboard",
            "🔥 Trending Movies",
            "🤖 AI Assistant",
            "👤 User Insights",
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
        st.header("📈 Analytics Dashboard")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Movies", f"{movies.shape[0]:,}")
        col2.metric("Total Ratings", f"{train.shape[0]:,}")
        col3.metric("Total Users", f"{train['userId'].nunique():,}")
        col4.metric("Avg Rating", round(train["rating"].mean(), 2))

        st.subheader("Rating Distribution")
        rating_counts = train["rating"].value_counts().sort_index()
        st.bar_chart(rating_counts)

        st.subheader("Top Genres")
        if "genres" in movies.columns:
            genre_counts = (
                movies["genres"]
                .fillna("")
                .str.split("|")
                .explode()
                .value_counts()
                .head(15)
            )
            st.bar_chart(genre_counts)
        else:
            st.info("No genres column found in movies data.")

        st.subheader("Most Rated Movies")
        show_movie_table(movie_stats.sort_values("rating_count", ascending=False), limit=15)

    with tabs[8]:
        st.header("🔥 Trending Movies")

        st.write("Trending score combines rating volume and average rating.")

        min_ratings_filter = st.slider("Minimum ratings", 1, int(movie_stats["rating_count"].max()), 25, 1)

        trending = movie_stats[movie_stats["rating_count"] >= min_ratings_filter].sort_values(
            "trend_score", ascending=False
        )

        if display_mode == "Cards":
            show_movie_cards(trending.head(top_n), score_col="trend_score")
        else:
            show_movie_table(trending, score_col="trend_score", limit=top_n)

    with tabs[9]:
        st.header("🤖 AI Assistant")

        st.write("Ask the assistant for movie ideas, genre suggestions, or profile-based recommendations.")

        assistant_prompt = st.text_area(
            "What do you want to watch?",
            value="Recommend something exciting but not too scary.",
            height=100,
        )

        if st.button("Ask AI Assistant"):
            response, ai_recs = chatbot.recommend(assistant_prompt, top_n=top_n)

            st.subheader("Assistant Response")
            st.write(response)

            st.subheader("Recommended Movies")
            if display_mode == "Cards":
                show_movie_cards(ai_recs, score_col="weighted_score")
            else:
                show_movie_table(ai_recs, score_col="weighted_score", limit=top_n)

    with tabs[10]:
        st.header("👤 User Insights")

        user_ratings = profile_store.get_user_ratings(username)

        if user_ratings.empty:
            st.info("Rate a few movies first to unlock personal insights.")
        else:
            st.subheader(f"Insights for {username}")

            col1, col2, col3 = st.columns(3)
            col1.metric("Movies Rated", len(user_ratings))
            col2.metric("Average Rating", round(user_ratings["rating"].mean(), 2))
            col3.metric("Highest Rating", round(user_ratings["rating"].max(), 2))

            enriched_ratings = user_ratings.merge(
                movies[["movieId", "genres"]] if "genres" in movies.columns else movies[["movieId"]],
                on="movieId",
                how="left",
            )

            if "genres" in enriched_ratings.columns:
                st.subheader("Your Favorite Genres")
                genre_pref = (
                    enriched_ratings.assign(genres=enriched_ratings["genres"].fillna(""))
                    .assign(genre=lambda df: df["genres"].str.split("|"))
                    .explode("genre")
                    .groupby("genre")["rating"]
                    .mean()
                    .sort_values(ascending=False)
                    .head(10)
                )

                genre_pref = genre_pref[genre_pref.index != ""]
                st.bar_chart(genre_pref)

            st.subheader("Your Highest Rated Movies")
            top_user_movies = user_ratings.sort_values("rating", ascending=False).head(10)
            st.dataframe(top_user_movies, use_container_width=True)

            if "title" in movies.columns:
                years = movies[["movieId", "title"]].copy()
                years["year"] = years["title"].apply(extract_year_from_title)
                rated_years = user_ratings.merge(years, on="movieId", how="left")

                if rated_years["year"].notna().any():
                    st.subheader("Your Movie Years")
                    year_counts = rated_years["year"].dropna().astype(int).value_counts().sort_index()
                    st.line_chart(year_counts)


if __name__ == "__main__":
    main()
