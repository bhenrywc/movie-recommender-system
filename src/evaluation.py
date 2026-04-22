from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from recommender import HybridRecommender, hit_rate_at_k


class KNNEvaluator:
    def __init__(self, train_df: pd.DataFrame, movies_df: pd.DataFrame, popularity_df: pd.DataFrame):
        self.train_df = train_df.copy()
        self.movies_df = movies_df.copy()
        self.popularity_df = popularity_df.copy()
        self._fit()

    def _fit(self) -> None:
        self.user_item_train = self.train_df.pivot_table(index="userId", columns="movieId", values="rating")
        self.user_item_train_filled = self.user_item_train.fillna(0)

        item_user_matrix = self.user_item_train_filled.T
        self.item_user_sparse = csr_matrix(item_user_matrix.to_numpy())
        self.knn_model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=20, n_jobs=-1)
        self.knn_model.fit(self.item_user_sparse)

        self.movie_ids = self.user_item_train.columns.to_list()
        self.movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(self.movie_ids)}
        self.idx_to_movie = {idx: movie_id for movie_id, idx in self.movie_to_idx.items()}

    def get_similar_movies(self, movie_id: int, n_neighbors: int = 10) -> pd.DataFrame:
        if movie_id not in self.movie_to_idx:
            return pd.DataFrame(columns=["movieId", "similarity"])

        movie_idx = self.movie_to_idx[movie_id]
        distances, indices = self.knn_model.kneighbors(self.item_user_sparse[movie_idx], n_neighbors=n_neighbors + 1)

        rows = []
        for dist, idx in zip(distances.flatten(), indices.flatten()):
            neighbor_movie_id = self.idx_to_movie[idx]
            if neighbor_movie_id == movie_id:
                continue
            rows.append((neighbor_movie_id, 1 - dist))

        return pd.DataFrame(rows, columns=["movieId", "similarity"]).sort_values("similarity", ascending=False).reset_index(drop=True)

    def recommend(self, user_id: int, top_n: int = 10, neighbor_k: int = 10, min_similarity: float = 0.10) -> pd.DataFrame:
        user_history = self.train_df[self.train_df["userId"] == user_id].copy()
        if user_history.empty:
            return self.popularity_df.head(top_n)[["movieId", "title", "genres", "weighted_score"]]

        rated_movie_ids = set(user_history["movieId"].tolist())
        candidate_scores: Dict[int, Dict[str, float]] = {}

        for _, row in user_history.iterrows():
            similar_movies = self.get_similar_movies(int(row["movieId"]), n_neighbors=neighbor_k)
            user_rating = float(row["rating"])

            for _, sim_row in similar_movies.iterrows():
                candidate_movie_id = int(sim_row["movieId"])
                similarity = float(sim_row["similarity"])

                if candidate_movie_id in rated_movie_ids or similarity < min_similarity:
                    continue

                if candidate_movie_id not in candidate_scores:
                    candidate_scores[candidate_movie_id] = {"weighted_sum": 0.0, "similarity_sum": 0.0}

                candidate_scores[candidate_movie_id]["weighted_sum"] += similarity * user_rating
                candidate_scores[candidate_movie_id]["similarity_sum"] += similarity

        if not candidate_scores:
            fallback = self.popularity_df[~self.popularity_df["movieId"].isin(rated_movie_ids)]
            return fallback.head(top_n)[["movieId", "title", "genres", "weighted_score"]]

        rows = []
        for movie_id, values in candidate_scores.items():
            rows.append((movie_id, values["weighted_sum"] / max(values["similarity_sum"], 1e-9)))

        recs = pd.DataFrame(rows, columns=["movieId", "knn_score"])
        recs = recs.merge(self.movies_df[["movieId", "title", "genres"]], on="movieId", how="left")
        recs = recs.merge(self.popularity_df[["movieId", "weighted_score", "rating_count"]], on="movieId", how="left")
        return recs.sort_values(["knn_score", "weighted_score"], ascending=False).head(top_n).reset_index(drop=True)


def precision_recall_coverage(recommended_items: List[int], relevant_items: set, catalog_size: int) -> Dict[str, float]:
    recommended_set = set(recommended_items)
    hits = len(recommended_set.intersection(relevant_items))
    precision = hits / len(recommended_items) if recommended_items else 0.0
    recall = hits / len(relevant_items) if relevant_items else 0.0
    coverage = len(recommended_set) / catalog_size if catalog_size > 0 else 0.0
    return {"precision": precision, "recall": recall, "coverage": coverage}


def evaluate_model(model: HybridRecommender, eval_df: pd.DataFrame, top_n: int = 10, min_eval_rating: float = 4.0, sample_users: int = 100, method: str = "hybrid") -> Dict[str, float]:
    eligible_users = []
    train_user_set = set(model.train_df["userId"].unique())

    for user_id, user_eval in eval_df.groupby("userId"):
        relevant = set(user_eval.loc[user_eval["rating"] >= min_eval_rating, "movieId"].tolist())
        if relevant and user_id in train_user_set:
            eligible_users.append(user_id)

    eligible_users = eligible_users[:sample_users]

    hit_count = 0
    precision_scores = []
    recall_scores = []
    all_recommended_items = set()
    catalog_size = len(set(model.movies_df["movieId"].tolist()))

    for user_id in eligible_users:
        relevant_items = set(
            eval_df[(eval_df["userId"] == user_id) & (eval_df["rating"] >= min_eval_rating)]["movieId"].tolist()
        )
        if not relevant_items:
            continue

        if method == "popularity":
            seen_items = set(model.train_df.loc[model.train_df["userId"] == user_id, "movieId"].tolist())
            recs = model._popularity_fallback(top_n=top_n, exclude_ids=seen_items)
        elif method == "mf":
            recs = model.recommend_mf(user_id=user_id, top_n=top_n)
        else:
            recs = model.recommend_hybrid(user_id=user_id, top_n=top_n)

        recommended_items = recs["movieId"].tolist()
        recommended_set = set(recommended_items)
        all_recommended_items.update(recommended_set)

        hits = len(recommended_set.intersection(relevant_items))
        hit_count += int(hits > 0)
        precision_scores.append(hits / top_n if top_n > 0 else 0.0)
        recall_scores.append(hits / len(relevant_items) if relevant_items else 0.0)

    return {
        "hit_rate@k": hit_count / len(precision_scores) if precision_scores else 0.0,
        "precision@k": float(np.mean(precision_scores)) if precision_scores else 0.0,
        "recall@k": float(np.mean(recall_scores)) if recall_scores else 0.0,
        "coverage@k": len(all_recommended_items) / catalog_size if catalog_size > 0 else 0.0,
        "users_evaluated": len(precision_scores),
    }


def evaluate_knn_model(knn_model: KNNEvaluator, eval_df: pd.DataFrame, top_n: int = 10, min_eval_rating: float = 4.0, sample_users: int = 100) -> Dict[str, float]:
    eligible_users = []
    train_user_set = set(knn_model.train_df["userId"].unique())

    for user_id, user_eval in eval_df.groupby("userId"):
        relevant = set(user_eval.loc[user_eval["rating"] >= min_eval_rating, "movieId"].tolist())
        if relevant and user_id in train_user_set:
            eligible_users.append(user_id)

    eligible_users = eligible_users[:sample_users]

    hit_count = 0
    precision_scores = []
    recall_scores = []
    all_recommended_items = set()
    catalog_size = len(set(knn_model.movies_df["movieId"].tolist()))

    for user_id in eligible_users:
        relevant_items = set(
            eval_df[(eval_df["userId"] == user_id) & (eval_df["rating"] >= min_eval_rating)]["movieId"].tolist()
        )
        if not relevant_items:
            continue

        recs = knn_model.recommend(user_id=user_id, top_n=top_n)
        recommended_items = recs["movieId"].tolist()
        recommended_set = set(recommended_items)
        all_recommended_items.update(recommended_set)

        hits = len(recommended_set.intersection(relevant_items))
        hit_count += int(hits > 0)
        precision_scores.append(hits / top_n if top_n > 0 else 0.0)
        recall_scores.append(hits / len(relevant_items) if relevant_items else 0.0)

    return {
        "hit_rate@k": hit_count / len(precision_scores) if precision_scores else 0.0,
        "precision@k": float(np.mean(precision_scores)) if precision_scores else 0.0,
        "recall@k": float(np.mean(recall_scores)) if recall_scores else 0.0,
        "coverage@k": len(all_recommended_items) / catalog_size if catalog_size > 0 else 0.0,
        "users_evaluated": len(precision_scores),
    }


def compare_models(model: HybridRecommender, val_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    knn = KNNEvaluator(model.train_df, model.movies_df, model.popularity_df)

    rows = []
    for split_name, split_df in [("Validation", val_df), ("Test", test_df)]:
        rows.append({"split": split_name, "model": "Popularity", **evaluate_model(model, split_df, method="popularity")})
        rows.append({"split": split_name, "model": "KNN Item-Item", **evaluate_knn_model(knn, split_df)})
        rows.append({"split": split_name, "model": "Matrix Factorization", **evaluate_model(model, split_df, method="mf")})
        rows.append({"split": split_name, "model": "Upgraded Hybrid", **evaluate_model(model, split_df, method="hybrid")})

    return pd.DataFrame(rows)
