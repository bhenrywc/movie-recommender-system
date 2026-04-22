import os
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


RATINGS_COLUMNS = ["userId", "movieId", "rating", "timestamp"]
MOVIES_COLUMNS = ["movieId", "title", "genres"]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_raw_data(base_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw_dir = os.path.join(base_dir, "data", "raw")
    ratings_path = os.path.join(raw_dir, "ratings.csv")
    movies_path = os.path.join(raw_dir, "movies.csv")

    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)

    return ratings, movies


def clean_movies(movies_df: pd.DataFrame) -> pd.DataFrame:
    movies = movies_df.copy()
    movies = movies.drop_duplicates(subset=["movieId"]).reset_index(drop=True)
    movies["title"] = movies["title"].fillna("Unknown Title")
    movies["genres"] = movies["genres"].fillna("(no genres listed)")
    return movies


def clean_ratings(ratings_df: pd.DataFrame) -> pd.DataFrame:
    ratings = ratings_df.copy()
    ratings = ratings.dropna(subset=["userId", "movieId", "rating"])
    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)
    ratings["rating"] = ratings["rating"].astype(float)

    if "timestamp" in ratings.columns:
        ratings["rated_at"] = pd.to_datetime(ratings["timestamp"], unit="s", errors="coerce")
    elif "rated_at" in ratings.columns:
        ratings["rated_at"] = pd.to_datetime(ratings["rated_at"], errors="coerce")
    else:
        ratings["rated_at"] = pd.NaT

    ratings = ratings.drop_duplicates(subset=["userId", "movieId", "rating", "rated_at"])
    ratings = ratings.sort_values(["userId", "rated_at", "movieId"], na_position="last").reset_index(drop=True)
    return ratings


def build_movie_stats(ratings_df: pd.DataFrame) -> pd.DataFrame:
    return (
        ratings_df.groupby("movieId")
        .agg(
            avg_rating=("rating", "mean"),
            rating_count=("rating", "count"),
            rating_std=("rating", "std"),
        )
        .reset_index()
        .fillna({"rating_std": 0.0})
    )


def build_user_stats(ratings_df: pd.DataFrame) -> pd.DataFrame:
    return (
        ratings_df.groupby("userId")
        .agg(
            avg_user_rating=("rating", "mean"),
            rating_count=("rating", "count"),
            min_rating=("rating", "min"),
            max_rating=("rating", "max"),
        )
        .reset_index()
    )


def build_genre_matrix(movies_df: pd.DataFrame) -> pd.DataFrame:
    genres = movies_df[["movieId", "genres"]].copy()
    genre_dummies = genres["genres"].str.get_dummies(sep="|")
    genre_matrix = pd.concat([genres[["movieId"]], genre_dummies], axis=1)
    return genre_matrix


def build_user_item_matrix(ratings_df: pd.DataFrame) -> pd.DataFrame:
    return ratings_df.pivot_table(index="userId", columns="movieId", values="rating")


def split_interactions(
    ratings_df: pd.DataFrame,
    test_size: float = 0.10,
    val_size: float = 0.10,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        ratings_df,
        test_size=(test_size + val_size),
        random_state=random_state,
    )

    relative_val_size = val_size / (test_size + val_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - relative_val_size),
        random_state=random_state,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def save_processed_artifacts(base_dir: str, artifacts: Dict[str, pd.DataFrame]) -> None:
    processed_dir = os.path.join(base_dir, "data", "processed")
    ensure_dir(processed_dir)

    for filename, df in artifacts.items():
        df.to_csv(os.path.join(processed_dir, filename), index=False)


def run_preprocessing(base_dir: str) -> Dict[str, pd.DataFrame]:
    ratings_raw, movies_raw = load_raw_data(base_dir)

    ratings_clean = clean_ratings(ratings_raw)
    movies_clean = clean_movies(movies_raw)

    train_df, val_df, test_df = split_interactions(ratings_clean)
    movie_stats = build_movie_stats(ratings_clean)
    user_stats = build_user_stats(ratings_clean)
    genre_encoded = build_genre_matrix(movies_clean)
    user_item_matrix = build_user_item_matrix(train_df).reset_index()

    artifacts = {
        "ratings_clean.csv": ratings_clean,
        "movies_clean.csv": movies_clean,
        "train.csv": train_df,
        "val.csv": val_df,
        "test.csv": test_df,
        "movie_stats.csv": movie_stats,
        "user_stats.csv": user_stats,
        "genre_encoded.csv": genre_encoded,
        "user_item_matrix.csv": user_item_matrix,
    }

    save_processed_artifacts(base_dir, artifacts)
    return artifacts


if __name__ == "__main__":
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    artifacts = run_preprocessing(BASE_DIR)
    print("Saved processed artifacts:")
    for name, df in artifacts.items():
        print(f"- {name}: {df.shape}")
