
import os
import pickle
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors


def build_popularity_df(train_df: pd.DataFrame, movies_df: pd.DataFrame) -> pd.DataFrame:
    popularity_df = (
        train_df.groupby("movieId")
        .agg(avg_rating=("rating", "mean"), rating_count=("rating", "count"))
        .reset_index()
    )
    global_mean = train_df["rating"].mean()
    vote_weight = popularity_df["rating_count"].mean()

    popularity_df["weighted_score"] = (
        (popularity_df["rating_count"] / (popularity_df["rating_count"] + vote_weight)) * popularity_df["avg_rating"]
        + (vote_weight / (popularity_df["rating_count"] + vote_weight)) * global_mean
    )

    popularity_df = popularity_df.merge(
        movies_df[["movieId", "title", "genres"]],
        on="movieId",
        how="left",
    ).sort_values("weighted_score", ascending=False).reset_index(drop=True)

    return popularity_df


def prepare_interaction_matrix(train_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, csr_matrix]:
    user_item = train_df.pivot_table(index="userId", columns="movieId", values="rating")
    user_item_filled = user_item.fillna(0.0)
    user_item_sparse = csr_matrix(user_item_filled.to_numpy())
    return user_item, user_item_filled, user_item_sparse


@dataclass
class HybridRecommender:
    train_df: pd.DataFrame
    movies_df: pd.DataFrame
    genre_df: pd.DataFrame
    n_components: int = 50
    neighbor_k: int = 15
    min_similarity: float = 0.10
    min_rating_for_profile: float = 4.0
    cf_weight: float = 0.30
    mf_weight: float = 0.45
    genre_weight: float = 0.15
    popularity_weight: float = 0.10

    def fit(self) -> "HybridRecommender":
        self.train_df = self.train_df.copy()
        self.movies_df = self.movies_df.copy()
        self.genre_df = self.genre_df.copy()

        self.user_item, self.user_item_filled, self.user_item_sparse = prepare_interaction_matrix(self.train_df)
        self.user_ids = self.user_item.index.to_list()
        self.movie_ids = self.user_item.columns.to_list()

        self.user_to_idx = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
        self.movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(self.movie_ids)}
        self.idx_to_user = {idx: user_id for user_id, idx in self.user_to_idx.items()}
        self.idx_to_movie = {idx: movie_id for movie_id, idx in self.movie_to_idx.items()}

        item_user_matrix = self.user_item_filled.T
        self.item_user_sparse = csr_matrix(item_user_matrix.to_numpy())

        self.knn_model = NearestNeighbors(
            metric="cosine",
            algorithm="brute",
            n_neighbors=max(self.neighbor_k + 1, 20),
            n_jobs=-1,
        )
        self.knn_model.fit(self.item_user_sparse)

        self.popularity_df = build_popularity_df(self.train_df, self.movies_df)
        self.popularity_lookup = self.popularity_df.set_index("movieId")
        self.global_mean = float(self.train_df["rating"].mean())

        self.genre_cols = [c for c in self.genre_df.columns if c != "movieId"]
        self.genre_lookup = self.genre_df.set_index("movieId")[self.genre_cols].copy()
        self.movie_meta = (
            self.movies_df[["movieId", "title", "genres"]]
            .drop_duplicates()
            .set_index("movieId")
        )

        max_components = max(2, min(self.n_components, self.user_item_filled.shape[0] - 1, self.user_item_filled.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=max_components, random_state=42)
        self.user_factors = self.svd.fit_transform(self.user_item_filled.to_numpy())
        self.item_factors = self.svd.components_
        self.mf_predictions = np.dot(self.user_factors, self.item_factors)
        self.pred_df = pd.DataFrame(
            self.mf_predictions,
            index=self.user_item.index,
            columns=self.user_item.columns,
        )

        self.existing_user_ids = set(self.train_df["userId"].unique())
        return self

    def save(self, filepath: str) -> None:
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: str) -> "HybridRecommender":
        with open(filepath, "rb") as f:
            return pickle.load(f)

    def find_movies_by_title(self, query: str, limit: int = 20) -> pd.DataFrame:
        mask = self.movies_df["title"].str.contains(query, case=False, na=False)
        return self.movies_df.loc[mask, ["movieId", "title", "genres"]].head(limit).reset_index(drop=True)

    def get_similar_movies(self, movie_id: int, n_neighbors: int = 10) -> pd.DataFrame:
        if movie_id not in self.movie_to_idx:
            return pd.DataFrame(columns=["movieId", "title", "genres", "similarity"])

        movie_idx = self.movie_to_idx[movie_id]
        distances, indices = self.knn_model.kneighbors(
            self.item_user_sparse[movie_idx],
            n_neighbors=n_neighbors + 1
        )

        rows = []
        for dist, idx in zip(distances.flatten(), indices.flatten()):
            neighbor_movie_id = self.idx_to_movie[idx]
            if neighbor_movie_id == movie_id:
                continue
            rows.append((neighbor_movie_id, float(1 - dist)))

        neighbors_df = pd.DataFrame(rows, columns=["movieId", "similarity"])
        neighbors_df = neighbors_df.merge(
            self.movies_df[["movieId", "title", "genres"]],
            on="movieId",
            how="left"
        ).sort_values("similarity", ascending=False).reset_index(drop=True)

        return neighbors_df

    def build_user_genre_profile(self, user_id: int) -> pd.Series:
        user_likes = self.train_df[
            (self.train_df["userId"] == user_id) &
            (self.train_df["rating"] >= self.min_rating_for_profile)
        ][["movieId", "rating"]].copy()

        if user_likes.empty:
            return pd.Series(0.0, index=self.genre_cols)

        liked_with_genres = user_likes.merge(
            self.genre_df[["movieId"] + self.genre_cols],
            on="movieId",
            how="left"
        ).fillna(0)

        weighted_genres = liked_with_genres[self.genre_cols].multiply(
            liked_with_genres["rating"], axis=0
        ).sum()

        if weighted_genres.max() > 0:
            weighted_genres = weighted_genres / weighted_genres.max()

        return weighted_genres

    def get_genre_overlap_score(self, movie_id: int, user_profile: pd.Series) -> float:
        if movie_id not in self.genre_lookup.index:
            return 0.0

        movie_vec = self.genre_lookup.loc[movie_id].astype(float)
        if float(movie_vec.sum()) == 0:
            return 0.0

        overlap = np.dot(movie_vec.values, user_profile.values) / movie_vec.sum()
        return float(overlap)

    def _popularity_fallback(self, top_n: int, exclude_ids: Optional[set] = None) -> pd.DataFrame:
        df = self.popularity_df.copy()
        if exclude_ids:
            df = df[~df["movieId"].isin(exclude_ids)]
        return df.head(top_n)[["movieId", "title", "genres", "weighted_score"]].reset_index(drop=True)

    def recommend_mf(self, user_id: int, top_n: int = 10) -> pd.DataFrame:
        if user_id not in self.pred_df.index:
            return self._popularity_fallback(top_n)

        user_row = self.pred_df.loc[user_id]
        already_rated = self.train_df[self.train_df["userId"] == user_id]["movieId"].tolist()

        recs = (
            user_row.drop(labels=already_rated, errors="ignore")
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )
        recs.columns = ["movieId", "mf_score"]
        recs = recs.merge(self.movies_df[["movieId", "title", "genres"]], on="movieId", how="left")
        return recs

    def recommend_hybrid(self, user_id: int, top_n: int = 10) -> pd.DataFrame:
        if user_id not in self.existing_user_ids:
            return self._popularity_fallback(top_n)

        user_history = self.train_df[self.train_df["userId"] == user_id].copy()
        if user_history.empty:
            return self._popularity_fallback(top_n)

        rated_movie_ids = set(user_history["movieId"].tolist())
        user_profile = self.build_user_genre_profile(user_id=user_id)
        candidate_scores: Dict[int, Dict[str, object]] = {}

        for _, row in user_history.iterrows():
            seed_movie_id = int(row["movieId"])
            user_rating = float(row["rating"])
            similar_movies = self.get_similar_movies(movie_id=seed_movie_id, n_neighbors=self.neighbor_k)

            for _, sim_row in similar_movies.iterrows():
                candidate_movie_id = int(sim_row["movieId"])
                similarity = float(sim_row["similarity"])

                if candidate_movie_id in rated_movie_ids:
                    continue
                if similarity < self.min_similarity:
                    continue

                weighted_contribution = similarity * user_rating
                if candidate_movie_id not in candidate_scores:
                    candidate_scores[candidate_movie_id] = {
                        "weighted_sum": 0.0,
                        "similarity_sum": 0.0,
                        "reasons": [],
                    }

                candidate_scores[candidate_movie_id]["weighted_sum"] += weighted_contribution
                candidate_scores[candidate_movie_id]["similarity_sum"] += similarity
                candidate_scores[candidate_movie_id]["reasons"].append((seed_movie_id, similarity, user_rating))

        if not candidate_scores:
            return self._popularity_fallback(top_n, exclude_ids=rated_movie_ids)

        rows = []
        mf_user_row = self.pred_df.loc[user_id] if user_id in self.pred_df.index else pd.Series(dtype=float)

        for candidate_movie_id, values in candidate_scores.items():
            cf_score = float(values["weighted_sum"] / max(values["similarity_sum"], 1e-9))
            genre_score = self.get_genre_overlap_score(candidate_movie_id, user_profile)

            pop_score = (
                float(self.popularity_lookup.loc[candidate_movie_id, "weighted_score"])
                if candidate_movie_id in self.popularity_lookup.index
                else self.global_mean
            )
            rating_count = (
                int(self.popularity_lookup.loc[candidate_movie_id, "rating_count"])
                if candidate_movie_id in self.popularity_lookup.index
                else 0
            )
            pop_score_norm = pop_score / 5.0

            mf_score_raw = float(mf_user_row.get(candidate_movie_id, self.global_mean))
            mf_score_norm = max(0.0, min(mf_score_raw / 5.0, 1.0))
            cf_score_norm = max(0.0, min(cf_score / 5.0, 1.0))

            hybrid_score = (
                self.cf_weight * cf_score_norm
                + self.mf_weight * mf_score_norm
                + self.genre_weight * genre_score
                + self.popularity_weight * pop_score_norm
            )

            best_reason = sorted(values["reasons"], key=lambda x: x[1] * x[2], reverse=True)[0]
            reason_movie_id = best_reason[0]
            reason_title = (
                self.movie_meta.loc[reason_movie_id, "title"]
                if reason_movie_id in self.movie_meta.index
                else str(reason_movie_id)
            )

            rows.append({
                "movieId": candidate_movie_id,
                "cf_score": round(cf_score, 4),
                "mf_score": round(mf_score_raw, 4),
                "genre_score": round(genre_score, 4),
                "popularity_score": round(pop_score, 4),
                "hybrid_score": round(hybrid_score, 4),
                "rating_count": rating_count,
                "reason": f"Because you liked: {reason_title}",
            })

        recs = pd.DataFrame(rows).merge(
            self.movies_df[["movieId", "title", "genres"]],
            on="movieId",
            how="left"
        )

        recs = recs.sort_values(
            ["hybrid_score", "mf_score", "cf_score", "popularity_score"],
            ascending=False
        ).head(top_n).reset_index(drop=True)

        return recs[
            [
                "movieId", "title", "genres", "cf_score", "mf_score",
                "genre_score", "popularity_score", "hybrid_score",
                "rating_count", "reason"
            ]
        ]

    def explain_recommendation(self, user_id: int, recommended_movie_id: int, neighbor_k: Optional[int] = None) -> pd.DataFrame:
        if user_id not in self.existing_user_ids:
            return pd.DataFrame(columns=["because_you_rated", "your_rating", "similarity"])

        if neighbor_k is None:
            neighbor_k = self.neighbor_k

        user_history = self.train_df[self.train_df["userId"] == user_id].copy()
        explanations = []

        for _, row in user_history.iterrows():
            seed_movie_id = int(row["movieId"])
            user_rating = float(row["rating"])
            similar_movies = self.get_similar_movies(seed_movie_id, n_neighbors=neighbor_k)
            match = similar_movies[similar_movies["movieId"] == recommended_movie_id]

            if not match.empty:
                sim = float(match.iloc[0]["similarity"])
                seed_title = self.movie_meta.loc[seed_movie_id, "title"] if seed_movie_id in self.movie_meta.index else str(seed_movie_id)
                explanations.append({
                    "because_you_rated": seed_title,
                    "your_rating": user_rating,
                    "similarity": round(sim, 4),
                })

        return pd.DataFrame(explanations).sort_values("similarity", ascending=False).reset_index(drop=True)


def hit_rate_at_k(model: HybridRecommender, eval_df: pd.DataFrame, top_n: int = 10, min_eval_rating: float = 4.0, sample_users: int = 100, method: str = "hybrid") -> float:
    eligible_users = []

    for user_id, user_eval in eval_df.groupby("userId"):
        if (user_eval["rating"] >= min_eval_rating).sum() == 0:
            continue
        eligible_users.append(user_id)

    eligible_users = eligible_users[:sample_users]
    hits = 0
    total = 0

    for user_id in eligible_users:
        actual_positive_movies = set(
            eval_df[
                (eval_df["userId"] == user_id) &
                (eval_df["rating"] >= min_eval_rating)
            ]["movieId"].tolist()
        )

        if not actual_positive_movies:
            continue

        if method == "hybrid":
            recs = model.recommend_hybrid(user_id=user_id, top_n=top_n)
        elif method == "mf":
            recs = model.recommend_mf(user_id=user_id, top_n=top_n)
        elif method == "popularity":
            seen = set(model.train_df.loc[model.train_df["userId"] == user_id, "movieId"])
            recs = model._popularity_fallback(top_n, exclude_ids=seen)
        else:
            raise ValueError("method must be one of: hybrid, mf, popularity")

        recommended_movies = set(recs["movieId"].tolist())
        hit = len(actual_positive_movies.intersection(recommended_movies)) > 0
        hits += int(hit)
        total += 1

    return hits / total if total > 0 else 0.0
