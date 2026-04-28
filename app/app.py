import os
import sys
import math
import random
import pandas as pd
import streamlit as st

# ----------------------------------------------------------
# PATH SETUP
# ----------------------------------------------------------
APP_DIR = os.path.dirname(__file__)
BASE_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))
SRC_DIR = os.path.join(BASE_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# ----------------------------------------------------------
# IMPORT PROJECT MODULES
# ----------------------------------------------------------
from config import (
    MODEL_PATH,
    USER_RATINGS_PATH,
    RECOMMENDER_PARAMS,
    DEFAULT_TOP_N
)

from data_loader import load_processed_data
from recommender import HybridRecommender
from user_profiles import UserProfileStore
from search_engine import MovieSearchEngine
from chatbot import MovieChatbot
from metrics import compare_models
from poster_utils import get_poster_url

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------
st.set_page_config(
    page_title="Movie Recommender 5.0",
    page_icon="🎬",
    layout="wide"
)

# ----------------------------------------------------------
# CACHE LOADERS
# ----------------------------------------------------------
@st.cache_data
def cached_data():
    return load_processed_data()

@st.cache_resource
def cached_model(train, movies, genres):

    if os.path.exists(MODEL_PATH):
        try:
            return HybridRecommender.load(MODEL_PATH)
        except:
            pass

    model = HybridRecommender(
        train_df=train,
        movies_df=movies,
        genre_df=genres,
        **RECOMMENDER_PARAMS
    ).fit()

    model.save(MODEL_PATH)
    return model

# ----------------------------------------------------------
# MOCK LIVE FEATURES
# Replace with APIs later
# ----------------------------------------------------------
def get_streaming(title):

    platforms = [
        "Netflix",
        "Prime Video",
        "Max",
        "Hulu",
        "Disney+",
        "Peacock",
        "Paramount+"
    ]

    random.seed(abs(hash(title)) % 100000)
    n = random.randint(1, 3)
    return random.sample(platforms, n)

def get_social(title):

    random.seed(abs(hash(title)) % 100000)

    return {
        "X": random.randint(1000, 40000),
        "Reddit": random.randint(500, 12000),
        "Facebook": random.randint(1000, 25000),
        "Instagram": random.randint(2000, 55000)
    }

def get_sentiment(title):

    random.seed(abs(hash(title)) % 100000)
    return random.randint(74, 97)

def get_youtube(title):

    return "https://www.youtube.com/watch?v=YoHD9XEInc0"

def hype_score(title):

    buzz = get_social(title)
    mentions = sum(buzz.values())
    sentiment = get_sentiment(title)

    score = (
        (mentions / 130000) * 0.60 +
        (sentiment / 100) * 0.40
    ) * 100

    return round(score, 2)

# ----------------------------------------------------------
# DISPLAY HELPERS
# ----------------------------------------------------------
def show_movie_table(df, limit=20):

    if df is None or df.empty:
        st.info("No results found.")
        return

    st.dataframe(df.head(limit), use_container_width=True)

def show_movie_cards(df, score_col=None):

    if df is None or df.empty:
        st.info("No movies found.")
        return

    for _, row in df.head(25).iterrows():

        title = row.get("title", "Unknown")
        genres = row.get("genres", "")
        movie_id = row.get("movieId", "")

        with st.container(border=True):

            c1, c2, c3 = st.columns([1, 5, 2])

            with c1:
                poster = get_poster_url(row)
                if poster:
                    st.image(poster, width=90)
                else:
                    st.write("🎬")

            with c2:
                st.subheader(title)
                st.caption(genres)
                st.write(f"Movie ID: `{movie_id}`")

                stream = ", ".join(get_streaming(title))
                st.caption(f"📺 {stream}")

                buzz = get_social(title)
                total = sum(buzz.values())
                st.caption(f"🔥 {total:,} social mentions")

            with c3:

                if score_col and score_col in row:
                    try:
                        st.metric(score_col, round(float(row[score_col]), 4))
                    except:
                        pass

                st.metric("Hype", hype_score(title))

# ----------------------------------------------------------
# MAIN APP
# ----------------------------------------------------------
def main():

    st.title("🎬 Movie Recommender 5.0")
    st.caption("Hybrid AI + Social Buzz + Streaming + Analytics")

    # ------------------------------------------
    # LOAD DATA
    # ------------------------------------------
    try:
        train, val, test, movies, genres = cached_data()
    except Exception as e:
        st.error(str(e))
        st.stop()

    model = cached_model(train, movies, genres)

    search_engine = MovieSearchEngine(movies)
    chatbot = MovieChatbot(model=model, movies_df=movies)
    profile_store = UserProfileStore(USER_RATINGS_PATH)

    # ------------------------------------------
    # SIDEBAR
    # ------------------------------------------
    with st.sidebar:

        st.header("👤 Profile")

        users = profile_store.get_users()
        default_user = users[0] if users else "brandon"

        username = st.text_input("Username", default_user)

        st.divider()

        top_n = st.slider("Recommendations", 5, 25, DEFAULT_TOP_N)

        display = st.radio(
            "Display Mode",
            ["Cards", "Table"],
            horizontal=True
        )

        st.divider()

        st.header("📊 Dataset")
        st.write(f"Movies: **{movies.shape[0]:,}**")
        st.write(f"Ratings: **{train.shape[0]:,}**")
        st.write(f"Users: **{train['userId'].nunique():,}**")

    # ------------------------------------------
    # TABS
    # ------------------------------------------
    tabs = st.tabs([
        "🏠 Home",
        "🔎 Search",
        "🎞 Similar",
        "🎯 My Recs",
        "⭐ Rate Movies",
        "📺 Streaming",
        "📱 Social Buzz",
        "▶️ YouTube",
        "📊 Metrics",
        "🤖 AI Assistant",
        "👤 Insights"
    ])

    # ------------------------------------------
    # HOME
    # ------------------------------------------
    with tabs[0]:

        st.header("🔥 Trending Right Now")

        trending = model.popularity_df.copy()
        trending["hype"] = trending["title"].apply(hype_score)

        trending = trending.sort_values(
            ["hype", "weighted_score"],
            ascending=False
        )

        if display == "Cards":
            show_movie_cards(trending, "hype")
        else:
            show_movie_table(trending)

    # ------------------------------------------
    # SEARCH
    # ------------------------------------------
    with tabs[1]:

        st.header("🔎 Search Movies")

        q = st.text_input("Title contains")

        results = search_engine.search(
            title_query=q,
            genre="All",
            min_year="",
            max_year="",
            limit=100
        )

        if display == "Cards":
            show_movie_cards(results)
        else:
            show_movie_table(results)

    # ------------------------------------------
    # SIMILAR
    # ------------------------------------------
    with tabs[2]:

        st.header("🎞 Similar Movie Finder")

        q = st.text_input("Search movie", "Toy Story")

        matches = search_engine.title_matches(q, limit=20)

        if not matches.empty:

            chosen = st.selectbox(
                "Choose Movie",
                matches["title"].tolist()
            )

            movie_id = int(
                matches.loc[
                    matches["title"] == chosen,
                    "movieId"
                ].iloc[0]
            )

            similar = model.get_similar_movies(
                movie_id,
                n_neighbors=top_n
            )

            if display == "Cards":
                show_movie_cards(similar, "similarity")
            else:
                show_movie_table(similar)

    # ------------------------------------------
    # RECOMMENDATIONS
    # ------------------------------------------
    with tabs[3]:

        st.header("🎯 Personalized Recommendations")

        try:
            recs = model.recommend_hybrid(
                user_id=1,
                top_n=top_n
            )
        except:
            recs = model._popularity_fallback(top_n=top_n)

        if display == "Cards":
            show_movie_cards(recs, "final_score")
        else:
            show_movie_table(recs)

    # ------------------------------------------
    # RATE MOVIES
    # ------------------------------------------
    with tabs[4]:

        st.header("⭐ Build Your Profile")

        q = st.text_input("Search movie to rate", "Matrix")

        matches = search_engine.title_matches(q, limit=20)

        if not matches.empty:

            selected = st.selectbox(
                "Choose title",
                matches["title"].tolist()
            )

            rating = st.slider(
                "Your Rating",
                1.0, 5.0, 4.0, 0.5
            )

            if st.button("Save Rating"):

                movie_id = int(
                    matches.loc[
                        matches["title"] == selected,
                        "movieId"
                    ].iloc[0]
                )

                profile_store.add_rating(
                    username=username,
                    movie_id=movie_id,
                    rating=rating
                )

                st.success("Rating saved.")

    # ------------------------------------------
    # STREAMING
    # ------------------------------------------
    with tabs[5]:

        st.header("📺 Where To Watch")

        movie = st.selectbox(
            "Select Movie",
            movies["title"].dropna().unique()
        )

        for p in get_streaming(movie):
            st.success(p)

    # ------------------------------------------
    # SOCIAL BUZZ
    # ------------------------------------------
    with tabs[6]:

        st.header("📱 Social Buzz")

        movie = st.selectbox(
            "Movie",
            movies["title"].dropna().unique(),
            key="buzz"
        )

        buzz = get_social(movie)

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("X", buzz["X"])
        c2.metric("Reddit", buzz["Reddit"])
        c3.metric("Facebook", buzz["Facebook"])
        c4.metric("Instagram", buzz["Instagram"])

        st.subheader("Audience Sentiment")
        st.progress(get_sentiment(movie) / 100)

    # ------------------------------------------
    # YOUTUBE
    # ------------------------------------------
    with tabs[7]:

        st.header("▶️ Reviews & Blogs")

        movie = st.selectbox(
            "Movie Title",
            movies["title"].dropna().unique(),
            key="yt"
        )

        st.write(f"Top review for **{movie}**")
        st.video(get_youtube(movie))

    # ------------------------------------------
    # METRICS
    # ------------------------------------------
    with tabs[8]:

        st.header("📊 Model Evaluation")

        sample_users = st.slider(
            "Sample Users",
            25, 200, 100, 25
        )

        if st.button("Run Evaluation"):

            with st.spinner("Running..."):

                results = compare_models(
                    model=model,
                    val_df=val,
                    test_df=test,
                    top_n=top_n,
                    min_eval_rating=4.0,
                    sample_users=sample_users
                )

            st.dataframe(results, use_container_width=True)

    # ------------------------------------------
    # AI ASSISTANT
    # ------------------------------------------
    with tabs[9]:

        st.header("🤖 AI Movie Assistant")

        prompt = st.text_area(
            "What do you want to watch?",
            "Recommend exciting sci-fi movies"
        )

        if st.button("Ask Assistant"):

            response, recs = chatbot.recommend(
                prompt,
                top_n=top_n
            )

            st.write(response)

            if display == "Cards":
                show_movie_cards(recs, "weighted_score")
            else:
                show_movie_table(recs)

    # ------------------------------------------
    # INSIGHTS
    # ------------------------------------------
    with tabs[10]:

        st.header("👤 User Insights")

        ratings = profile_store.get_user_ratings(username)

        if ratings.empty:
            st.info("Rate movies first.")
        else:
            c1, c2, c3 = st.columns(3)

            c1.metric("Movies Rated", len(ratings))
            c2.metric(
                "Average Rating",
                round(ratings["rating"].mean(), 2)
            )
            c3.metric(
                "Highest",
                round(ratings["rating"].max(), 2)
            )

            st.dataframe(ratings, use_container_width=True)

    st.markdown("---")
    st.caption("Movie Recommender 5.0 | Production Build | Deploy Ready")

# ----------------------------------------------------------
# RUN
# ----------------------------------------------------------
if __name__ == "__main__":
    main()