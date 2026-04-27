
import os
import pickle
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors


class HybridRecommender:
    def __init__(
        self,
        train_df,
        movies_df,
        genre_df,
        n_components=50,
        neighbor_k=15,
        min_similarity=0.10,
        min_rating_for_profile=4.0,
        cf_weight=0.30,
        mf_weight=0.45,
        genre_weight=0.15,
        popularity_weight=0.10,
    ):
        self.train_df = train_df.copy()
        self.movies_df = movies_df.copy()
        self.genre_df = genre_df.copy()
        self.n_components = n_components
        self.neighbor_k = neighbor_k
        self.min_similarity = min_similarity
        self.min_rating_for_profile = min_rating_for_profile
        self.cf_weight = cf_weight
        self.mf_weight = mf_weight
        self.genre_weight = genre_weight
        self.popularity_weight = popularity_weight

    def fit(self):
        self.user_item = self.train_df.pivot_table(index="userId", columns="movieId", values="rating")
        self.user_item_filled = self.user_item.fillna(0)
        self.user_ids = self.user_item.index.tolist()
        self.movie_ids = self.user_item.columns.tolist()
        self.user_to_idx = {u:i for i,u in enumerate(self.user_ids)}
        self.movie_to_idx = {m:i for i,m in enumerate(self.movie_ids)}
        self.idx_to_movie = {i:m for m,i in self.movie_to_idx.items()}

        self.matrix_sparse = csr_matrix(self.user_item_filled.values)
        safe_components = min(self.n_components, min(self.user_item_filled.shape)-1)
        self.svd = TruncatedSVD(n_components=safe_components, random_state=42)
        self.user_factors = self.svd.fit_transform(self.matrix_sparse)
        self.movie_factors = self.svd.components_.T
        self.pred_matrix = np.dot(self.user_factors, self.movie_factors.T)
        self.pred_df = pd.DataFrame(self.pred_matrix, index=self.user_ids, columns=self.movie_ids)

        self.item_user_sparse = csr_matrix(self.user_item_filled.T.values)
        self.knn_model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=min(50,len(self.movie_ids)))
        self.knn_model.fit(self.item_user_sparse)
        return self

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path,"wb") as f:
            pickle.dump(self,f)

def hit_rate_at_k(*args, **kwargs):
    return 0.0
