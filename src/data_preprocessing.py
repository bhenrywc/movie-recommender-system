    import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity(matrix):
    return cosine_similarity(matrix.fillna(0))

def recommend_for_user(user_id, matrix, similarity, n=10):
    user_idx = user_id - 1
    sim_scores = similarity[user_idx]

    similar_users = np.argsort(sim_scores)[::-1]

    recommendations = []
    for other_user in similar_users:
        if other_user == user_idx:
            continue
        
        user_ratings = matrix.iloc[other_user]
        top_movies = user_ratings[user_ratings > 4].index.tolist()
        recommendations.extend(top_movies)

        if len(recommendations) >= n:
            break

    return recommendations[:n]
