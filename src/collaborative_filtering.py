import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


class HybridRecommender:
    """
    Hybrid recommender system that combines:
    1. User-based collaborative filtering
    2. Content-based filtering using genre features
    3. Popularity-based scoring

    Expected inputs:
    - train_df: DataFrame with columns ['userId', 'movieId', 'rating']
    - movies_df: DataFrame with at least ['movieId'] and optionally movie metadata like ['title']
    - genre_df: DataFrame containing genre-encoded movie features with a 'movieId' column
    """

    def __init__(
        self,
        train_df,
        movies_df,
        genre_df,
        neighbor_k=15,
        knn_neighbors=30,
        min_similarity=0.10,
        min_rating_for_profile=4.0,
        genre_weight=0.35,
        popularity_weight=0.15,
        min_candidate_rating_count=3,
        confidence_scaling=True,
    ):
        self.train_df = train_df.copy()
        self.movies_df = movies_df.copy()
        self.genre_df = genre_df.copy()

        self.neighbor_k = neighbor_k
        self.knn_neighbors = knn_neighbors
        self.min_similarity = min_similarity
        self.min_rating_for_profile = min_rating_for_profile
        self.genre_weight = genre_weight
        self.popularity_weight = popularity_weight
        self.min_candidate_rating_count = min_candidate_rating_count
        self.confidence_scaling = confidence_scaling

        self.cf_weight = 1.0 - genre_weight - popularity_weight
        if self.cf_weight < 0:
            raise ValueError(
                "genre_weight + popularity_weight must be <= 1.0"
            )

        self.user_item_matrix = None
        self.user_similarity_df = None
        self.knn_model = None
        self.movie_feature_matrix = None
        self.genre_feature_cols = None
        self.movie_popularity = None
        self.movie_mean_ratings = None
        self.global_mean = None
        self.user_profiles = None
        self.rated_items_by_user = None
        self.movie_id_to_index = None
        self.index_to_movie_id = None

    def fit(self):
        """
        Build all internal structures needed for recommendation.
        """
        required_train_cols = {"userId", "movieId", "rating"}
        if not required_train_cols.issubset(self.train_df.columns):
            raise ValueError(
                f"train_df must contain columns: {required_train_cols}"
            )

        if "movieId" not in self.movies_df.columns:
            raise ValueError("movies_df must contain 'movieId' column")

        if "movieId" not in self.genre_df.columns:
            raise ValueError("genre_df must contain 'movieId' column")

        # Ensure types are consistent
        self.train_df["userId"] = self.train_df["userId"].astype(int)
        self.train_df["movieId"] = self.train_df["movieId"].astype(int)
        self.movies_df["movieId"] = self.movies_df["movieId"].astype(int)
        self.genre_df["movieId"] = self.genre_df["movieId"].astype(int)

        # Remove duplicate movie rows if present
        self.movies_df = self.movies_df.drop_duplicates(subset=["movieId"]).reset_index(drop=True)
        self.genre_df = self.genre_df.drop_duplicates(subset=["movieId"]).reset_index(drop=True)

        # User-item matrix
        self.user_item_matrix = self.train_df.pivot_table(
            index="userId",
            columns="movieId",
            values="rating",
            fill_value=0
        )

        # User-user similarity
        user_sim = cosine_similarity(self.user_item_matrix)
        self.user_similarity_df = pd.DataFrame(
            user_sim,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )

        # KNN model for nearest neighbors
        self.knn_model = NearestNeighbors(
            metric="cosine",
            algorithm="brute",
            n_neighbors=min(self.knn_neighbors, len(self.user_item_matrix))
        )
        self.knn_model.fit(self.user_item_matrix)

        # Genre/content features
        self.genre_feature_cols = [col for col in self.genre_df.columns if col != "movieId"]
        if len(self.genre_feature_cols) == 0:
            raise ValueError("genre_df must contain genre feature columns besides 'movieId'")

        genre_features = self.genre_df[["movieId"] + self.genre_feature_cols].copy()
        self.movie_feature_matrix = genre_features.set_index("movieId")

        # Popularity and mean rating stats
        movie_stats = self.train_df.groupby("movieId").agg(
            rating_count=("rating", "count"),
            mean_rating=("rating", "mean")
        ).reset_index()

        self.movie_popularity = movie_stats.set_index("movieId")["rating_count"]
        self.movie_mean_ratings = movie_stats.set_index("movieId")["mean_rating"]
        self.global_mean = self.train_df["rating"].mean()

        # Rated items by each user
        self.rated_items_by_user = (
            self.train_df.groupby("userId")["movieId"]
            .apply(set)
            .to_dict()
        )

        # User content profiles
        self.user_profiles = self._build_user_profiles()

        # Mappings
        self.movie_id_to_index = {movie_id: idx for idx, movie_id in enumerate(self.movie_feature_matrix.index)}
        self.index_to_movie_id = {idx: movie_id for movie_id, idx in self.movie_id_to_index.items()}

        return self

    def _build_user_profiles(self):
        """
        Create a content profile vector for each user based on highly-rated movies.
        """
        user_profiles = {}

        for user_id, group in self.train_df.groupby("userId"):
            liked = group[group["rating"] >= self.min_rating_for_profile]

            if liked.empty:
                # fallback: use all rated movies weighted by rating
                liked = group.copy()

            liked = liked.merge(
                self.movie_feature_matrix.reset_index(),
                on="movieId",
                how="inner"
            )

            if liked.empty:
                user_profiles[user_id] = np.zeros(len(self.genre_feature_cols))
                continue

            feature_matrix = liked[self.genre_feature_cols].values
            weights = liked["rating"].values.reshape(-1, 1)

            weighted_profile = np.sum(feature_matrix * weights, axis=0)
            norm = np.linalg.norm(weighted_profile)

            if norm > 0:
                weighted_profile = weighted_profile / norm

            user_profiles[user_id] = weighted_profile

        return user_profiles

    def _get_knn_neighbors(self, user_id):
        """
        Retrieve nearest users using KNN.
        """
        if user_id not in self.user_item_matrix.index:
            return []

        user_vector = self.user_item_matrix.loc[user_id].values.reshape(1, -1)
        n_neighbors = min(self.knn_neighbors, len(self.user_item_matrix))

        distances, indices = self.knn_model.kneighbors(user_vector, n_neighbors=n_neighbors)

        neighbors = []
        for dist, idx in zip(distances[0], indices[0]):
            neighbor_user_id = self.user_item_matrix.index[idx]
            similarity = 1 - dist

            if neighbor_user_id == user_id:
                continue

            if similarity >= self.min_similarity:
                neighbors.append((neighbor_user_id, similarity))

        return neighbors[: self.neighbor_k]

    def _collaborative_score(self, user_id, candidate_movie_ids):
        """
        Predict score for candidate items using weighted neighbor ratings.
        """
        neighbors = self._get_knn_neighbors(user_id)
        scores = {}

        if not neighbors:
            return {movie_id: 0.0 for movie_id in candidate_movie_ids}

        for movie_id in candidate_movie_ids:
            numerator = 0.0
            denominator = 0.0
            contributing_neighbors = 0

            for neighbor_id, similarity in neighbors:
                rating = self.user_item_matrix.loc[neighbor_id, movie_id] if movie_id in self.user_item_matrix.columns else 0

                if rating > 0:
                    numerator += similarity * rating
                    denominator += abs(similarity)
                    contributing_neighbors += 1

            if denominator > 0:
                pred = numerator / denominator

                if self.confidence_scaling:
                    confidence = min(contributing_neighbors / max(self.neighbor_k, 1), 1.0)
                    pred *= confidence
            else:
                pred = 0.0

            scores[movie_id] = pred

        return scores

    def _content_score(self, user_id, candidate_movie_ids):
        """
        Score candidate items using cosine similarity between user profile and movie genre vector.
        """
        scores = {}

        if user_id not in self.user_profiles:
            return {movie_id: 0.0 for movie_id in candidate_movie_ids}

        user_profile = self.user_profiles[user_id]
        user_norm = np.linalg.norm(user_profile)

        for movie_id in candidate_movie_ids:
            if movie_id not in self.movie_feature_matrix.index:
                scores[movie_id] = 0.0
                continue

            movie_vec = self.movie_feature_matrix.loc[movie_id].values
            movie_norm = np.linalg.norm(movie_vec)

            if user_norm == 0 or movie_norm == 0:
                sim = 0.0
            else:
                sim = np.dot(user_profile, movie_vec) / (user_norm * movie_norm)

            scores[movie_id] = float(sim)

        return scores

    def _popularity_score(self, candidate_movie_ids):
        """
        Popularity score based on normalized rating counts and mean rating.
        """
        scores = {}

        if self.movie_popularity is None or len(self.movie_popularity) == 0:
            return {movie_id: 0.0 for movie_id in candidate_movie_ids}

        max_count = self.movie_popularity.max()
        min_count = self.movie_popularity.min()

        for movie_id in candidate_movie_ids:
            count = self.movie_popularity.get(movie_id, 0)
            mean_rating = self.movie_mean_ratings.get(movie_id, self.global_mean)

            if max_count > min_count:
                count_norm = (count - min_count) / (max_count - min_count)
            else:
                count_norm = 0.0

            rating_norm = mean_rating / 5.0
            scores[movie_id] = 0.7 * count_norm + 0.3 * rating_norm

        return scores

    def _build_candidate_pool(self, user_id):
        """
        Candidate items are unseen movies with at least a minimum number of ratings.
        """
        all_movies = set(self.movies_df["movieId"].unique())
        rated_movies = self.rated_items_by_user.get(user_id, set())

        unseen_movies = all_movies - rated_movies

        filtered_candidates = [
            movie_id for movie_id in unseen_movies
            if self.movie_popularity.get(movie_id, 0) >= self.min_candidate_rating_count
        ]

        return filtered_candidates

    def recommend(self, user_id, top_n=10, return_scores=False):
        """
        Generate top-N recommendations for a user.
        """
        if self.user_item_matrix is None:
            raise ValueError("Model is not fitted yet. Call .fit() first.")

        if user_id not in self.user_item_matrix.index:
            raise ValueError(f"user_id {user_id} not found in training data")

        candidate_movie_ids = self._build_candidate_pool(user_id)

        if not candidate_movie_ids:
            return pd.DataFrame(columns=["movieId", "title", "hybrid_score"])

        cf_scores = self._collaborative_score(user_id, candidate_movie_ids)
        content_scores = self._content_score(user_id, candidate_movie_ids)
        popularity_scores = self._popularity_score(candidate_movie_ids)

        rows = []
        for movie_id in candidate_movie_ids:
            cf = cf_scores.get(movie_id, 0.0)
            content = content_scores.get(movie_id, 0.0)
            pop = popularity_scores.get(movie_id, 0.0)

            hybrid_score = (
                self.cf_weight * cf +
                self.genre_weight * content +
                self.popularity_weight * pop
            )

            rows.append({
                "movieId": movie_id,
                "cf_score": cf,
                "content_score": content,
                "popularity_score": pop,
                "hybrid_score": hybrid_score
            })

        recs = pd.DataFrame(rows).sort_values("hybrid_score", ascending=False).head(top_n)

        recs = recs.merge(
            self.movies_df.drop_duplicates(subset=["movieId"]),
            on="movieId",
            how="left"
        )

        cols = ["movieId"]
        if "title" in recs.columns:
            cols.append("title")

        if return_scores:
            cols.extend(["cf_score", "content_score", "popularity_score", "hybrid_score"])
        else:
            cols.append("hybrid_score")

        return recs[cols].reset_index(drop=True)

    def recommend_for_new_user(self, liked_movie_ids, top_n=10):
        """
        Recommend for a cold-start user based only on liked movies.
        """
        liked_movie_ids = [int(m) for m in liked_movie_ids if m in self.movie_feature_matrix.index]

        if not liked_movie_ids:
            raise ValueError("No valid liked_movie_ids found in movie feature matrix.")

        liked_features = self.movie_feature_matrix.loc[liked_movie_ids].values
        user_profile = liked_features.mean(axis=0)

        norm = np.linalg.norm(user_profile)
        if norm > 0:
            user_profile = user_profile / norm

        seen_movies = set(liked_movie_ids)
        candidate_movie_ids = [
            movie_id for movie_id in self.movies_df["movieId"].unique()
            if movie_id not in seen_movies and self.movie_popularity.get(movie_id, 0) >= self.min_candidate_rating_count
        ]

        rows = []
        for movie_id in candidate_movie_ids:
            if movie_id not in self.movie_feature_matrix.index:
                continue

            movie_vec = self.movie_feature_matrix.loc[movie_id].values
            movie_norm = np.linalg.norm(movie_vec)

            if norm == 0 or movie_norm == 0:
                content_score = 0.0
            else:
                content_score = np.dot(user_profile, movie_vec) / movie_norm

            pop_score = self._popularity_score([movie_id]).get(movie_id, 0.0)

            hybrid_score = (
                (1.0 - self.genre_weight - self.popularity_weight) * 0.0 +
                self.genre_weight * content_score +
                self.popularity_weight * pop_score
            )

            rows.append({
                "movieId": movie_id,
                "content_score": content_score,
                "popularity_score": pop_score,
                "hybrid_score": hybrid_score
            })

        recs = pd.DataFrame(rows).sort_values("hybrid_score", ascending=False).head(top_n)
        recs = recs.merge(self.movies_df, on="movieId", how="left")

        cols = ["movieId"]
        if "title" in recs.columns:
            cols.append("title")
        cols.extend(["content_score", "popularity_score", "hybrid_score"])

        return recs[cols].reset_index(drop=True)