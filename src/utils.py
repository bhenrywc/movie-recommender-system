import pandas as pd

def get_movie_titles(movie_ids, movies_df):
    return movies_df[movies_df['movieId'].isin(movie_ids)][['movieId', 'title']]

def filter_seen_movies(recommendations, user_history):
    return [m for m in recommendations if m not in user_history]

def format_recommendations(movie_ids, movies_df):
    return get_movie_titles(movie_ids, movies_df).to_dict(orient="records")
