import os
import sys
import re
import pandas as pd
import streamlit as st

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(BASE_DIR, "src")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
USER_DATA_DIR = os.path.join(BASE_DIR, "data", "user")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from recommender import HybridRecommender
from user_profiles import UserProfileStore


st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide",
)


def clean_year(title):
    match = re.search(r"\((\d{4})\)", str(title))
    return int(match.group(1)) if match else None


@st.cache_data
def load_processed_data():
    train = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    movies_clean = pd.read_csv(os.path.join(PROCESSED_DIR, "movies_clean.csv"))
    genre_encoded = pd.read_csv(os.path.join(PROCESSED_DIR, "genre_encoded.csv"))

    if "year" not in movies_clean.columns:
        movies_clean["year"] = movies_clean["title"].apply(clean_year)

    return train, movies_clean, genre_encoded


@st.cache_resource
def load_or_train_model():
    train, movies_clean, genre_encoded = load_processed_data()
    model_path = os.path.join(MODELS_DIR, "hybrid_recommender.pkl")

    if os.path.exists(model_path):
        try:
            return HybridRecommender.load(model_path)
        except Exception:
            pass

    model = HybridRecommender(
        train_df=train,
        movies_df=movies_clean,
        genre_df=genre_encoded,
        n_components=50,
        neighbor_k=15,
        min_similarity=0.10,
        min_rating_for_profile=4.0,
        cf_weight=0.30,
        mf_weight=0.45,
        genre_weight=0.15,
        popularity_weight=0.10,
    ).fit()

    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(model_path)
    return model


def poster_url_from_tmdb_id(tmdb_id):
    """
    Optional upgrade:
    If your links.csv or movies_clean.csv has tmdbId and you later add poster_path,
    replace this function with real TMDB image URLs.
    """
    return None


def display_movie_cards(df, score_col=None):
    if df is None or df.empty:
        st.info("No movies found.")
        return

    for _, row in df.iterrows():
        with st.container(border=True):
            cols = st.columns([4, 2, 2])

            with cols[0]:
                st.subheader(row.get("title", "Unknown Title"))
                st.caption(row.get("genres", ""))

            with cols[1]:
                st.write(f"**Movie ID:** {row.get('movieId')}")
                if "rating_count" in row:
                    st.write(f"**Ratings:** {int(row.get('rating_count', 0))}")

            with cols[2]:
                if score_col and score_col in row:
                    st.metric(score_col, round(float(row[score_col]), 4))


def simple_chat_recommendation(query, movies_df, model):
    query_lower = query.lower()

    genre_keywords = [
        "action", "adventure", "animation", "children", "comedy", "crime", "documentary",
        "drama", "fantasy", "film-noir", "horror", "musical", "mystery", "romance",
        "sci-fi", "thriller", "war", "western"
    ]

    matched_genres = [g for g in genre_keywords if g in query_lower]

    year_match = re.search(r"(19\d{2}|20\d{2})", query_lower)
    year = int(year_match.group(1)) if year_match else None

    recs = model.popularity_df.copy()

    if matched_genres:
        pattern = "|".join([re.escape(g) for g in matched_genres])
        recs = recs[recs["genres"].str.lower().str.contains(pattern, na=False)]

    if year and "title" in recs.columns:
        recs["year"] = recs["title"].apply(clean_year)
        recs = recs[recs["year"] == year]

    if "top" in query_lower or "best" in query_lower or "recommend" in query_lower:
        recs = recs.sort_values("weighted_score", ascending=False)

    return recs.head(10)


def main():
    st.title("🎬 Movie Recommender System 2.0")
    st.caption("Hybrid recommendations + user profiles + ratings + search + chatbot-style discovery")

    train, movies_clean, genre_encoded = load_processed_data()
    model = load_or_train_model()

    profile_store = UserProfileStore(os.path.join(USER_DATA_DIR, "user_ratings.csv"))

    with st.sidebar:
        st.header("👤 User Profile")
        existing_users = profile_store.get_users()
        username = st.text_input("Username", value=existing_users[0] if existing_users else "brandon")

        st.divider()
        st.write("Dataset")
        st.write(f"Movies: {movies_clean.shape[0]:,}")
        st.write(f"Training ratings: {train.shape[0]:,}")

    tab_home, tab_search, tab_similar, tab_profile, tab_recs, tab_chat = st.tabs([
        "🏠 Home",
        "🔎 Search Movies",
        "🎞 Similar Movies",
        "⭐ Profile & Ratings",
        "🎯 My Recommendations",
        "💬 Movie Chatbox",
    ])

    with tab_home:
        st.header("Popular Movies")
        display_movie_cards(model._popularity_fallback(top_n=10), score_col="weighted_score")

    with tab_search:
        st.header("Search Movies")

        col1, col2, col3 = st.columns(3)
        with col1:
            title_query = st.text_input("Search by title", "")
        with col2:
            all_genres = sorted(set("|".join(movies_clean["genres"].dropna()).split("|")))
            selected_genre = st.selectbox("Filter by genre", ["All"] + all_genres)
        with col3:
            year_filter = st.text_input("Filter by year", "")

        results = movies_clean.copy()

        if title_query:
            results = results[results["title"].str.contains(title_query, case=False, na=False)]

        if selected_genre != "All":
            results = results[results["genres"].str.contains(selected_genre, case=False, na=False)]

        if year_filter.strip():
            results["year_tmp"] = results["title"].apply(clean_year)
            try:
                results = results[results["year_tmp"] == int(year_filter)]
            except ValueError:
                st.warning("Year must be a number, like 1995.")

        st.write(f"Results: {len(results):,}")
        display_movie_cards(results.head(30))

    with tab_similar:
        st.header("Find Similar Movies")

        movie_title = st.text_input("Type a movie title", "Toy Story")
        matches = model.find_movies_by_title(movie_title)

        if not matches.empty:
            selected_title = st.selectbox("Choose movie", matches["title"].tolist())
            selected_movie_id = int(matches.loc[matches["title"] == selected_title, "movieId"].iloc[0])

            similar = model.get_similar_movies(selected_movie_id, n_neighbors=10)
            display_movie_cards(similar, score_col="similarity")
        else:
            st.info("No matching movie found.")

    with tab_profile:
        st.header("Create Your Profile & Rate Movies")

        st.write(f"Current profile: **{username}**")

        rate_query = st.text_input("Search movie to rate", "Matrix")
        rate_matches = model.find_movies_by_title(rate_query)

        if not rate_matches.empty:
            movie_choice = st.selectbox("Choose movie to rate", rate_matches["title"].tolist())
            movie_id = int(rate_matches.loc[rate_matches["title"] == movie_choice, "movieId"].iloc[0])
            rating = st.slider("Your rating", 0.5, 5.0, 4.0, 0.5)

            if st.button("Save Rating"):
                profile_store.add_or_update_rating(username, movie_id, movie_choice, rating)
                st.success(f"Saved rating: {movie_choice} = {rating}")

        st.subheader("Your Ratings")
        user_ratings = profile_store.get_user_ratings(username)
        st.dataframe(user_ratings, use_container_width=True)

    with tab_recs:
        st.header("Recommendations Based on Your Profile")

        user_ratings = profile_store.get_user_ratings(username)

        if user_ratings.empty:
            st.info("Rate at least a few movies first. Until then, here are popular recommendations.")
            display_movie_cards(model._popularity_fallback(top_n=10), score_col="weighted_score")
        else:
            recs = model.recommend_from_custom_ratings(
                user_ratings[["movieId", "rating"]],
                top_n=10,
            )
            display_movie_cards(recs, score_col="final_score")

    with tab_chat:
        st.header("Movie Chatbox")
        st.write("Ask something like: `Recommend me funny action movies from 1995` or `best horror movies`.")

        query = st.chat_input("Ask for movie recommendations...")

        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []

        if query:
            st.session_state.chat_messages.append({"role": "user", "content": query})
            recs = simple_chat_recommendation(query, movies_clean, model)
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": "Here are recommendations that match your request:",
                "results": recs,
            })

        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if "results" in message:
                    display_movie_cards(message["results"], score_col="weighted_score")


if __name__ == "__main__":
    main()
