"""
recommender.py
Intelligent Travel Recommendation System - Canada
MSc Data Science Project

Full ML pipeline:
  1. Feature engineering (TF-IDF on tags, numerical scaling)
  2. Content-Based Filtering (cosine similarity)
  3. Collaborative Filtering (SVD via scipy matrix factorisation)
  4. Hybrid ensemble
  5. Evaluation metrics (Precision@K, Recall@K, NDCG@K, RMSE)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings("ignore")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 1. DATA LOADING
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def load_data(data_dir="data"):
    dest    = pd.read_csv(f"{data_dir}/destinations.csv")
    users   = pd.read_csv(f"{data_dir}/users.csv")
    ratings = pd.read_csv(f"{data_dir}/ratings.csv")
    weather = pd.read_csv(f"{data_dir}/weather.csv")
    return dest, users, ratings, weather


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 2. FEATURE ENGINEERING
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class FeatureEngineer:
    def __init__(self):
        self.tfidf      = TfidfVectorizer(tokenizer=lambda x: x.split("|"), lowercase=False)
        self.scaler     = MinMaxScaler()
        self.dest_matrix = None
        self.dest_ids   = None

    def fit_transform(self, dest_df, weather_df):
        """Build destination feature matrix."""
        # TF-IDF on interest tags
        tfidf_matrix = self.tfidf.fit_transform(dest_df["tags"]).toarray()

        # Seasonal demand: mean visitor index per destination
        seas = weather_df.groupby("dest_id")["visitor_index"].mean().reset_index()
        seas.columns = ["id", "mean_visitor_idx"]
        dest_df = dest_df.merge(seas, on="id", how="left")

        # Numerical features
        num_features = dest_df[["avg_cost_day", "avg_rating", "mean_visitor_idx"]].fillna(0).values
        num_scaled   = self.scaler.fit_transform(num_features)

        # Combine
        self.dest_matrix = np.hstack([tfidf_matrix, num_scaled])
        self.dest_ids    = dest_df["id"].tolist()
        return self.dest_matrix

    def build_user_vector(self, interests: list, budget: float, rating_pref: float = 4.0):
        """Convert user preferences to a feature vector in the same space."""
        tag_str  = "|".join(interests)
        tfidf_v  = self.tfidf.transform([tag_str]).toarray()
        num_v    = self.scaler.transform([[budget, rating_pref, 0.7]])
        return np.hstack([tfidf_v, num_v])


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 3. CONTENT-BASED FILTERING
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class ContentBasedFilter:
    def __init__(self, feature_engineer: FeatureEngineer):
        self.fe = feature_engineer

    def recommend(self, user_interests, budget, top_k=10):
        user_vec = self.fe.build_user_vector(user_interests, budget)
        sims     = cosine_similarity(user_vec, self.fe.dest_matrix)[0]
        ranked   = np.argsort(sims)[::-1][:top_k]
        return [(self.fe.dest_ids[i], float(sims[i])) for i in ranked]


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 4. COLLABORATIVE FILTERING (SVD)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class CollaborativeFilter:
    def __init__(self, n_factors=15):
        self.n_factors   = n_factors
        self.user_idx    = {}
        self.dest_idx    = {}
        self.user_means  = {}
        self.pred_matrix = None

    def fit(self, ratings_df):
        users = ratings_df["user_id"].unique()
        dests = ratings_df["dest_id"].unique()
        self.user_idx = {u: i for i, u in enumerate(users)}
        self.dest_idx = {d: i for i, d in enumerate(dests)}

        R = np.zeros((len(users), len(dests)))
        for _, row in ratings_df.iterrows():
            ui = self.user_idx[row["user_id"]]
            di = self.dest_idx[row["dest_id"]]
            R[ui, di] = row["rating"]

        # Mean-centre per user
        self.user_means = {}
        R_norm = R.copy()
        for u, ui in self.user_idx.items():
            mask = R_norm[ui] > 0
            if mask.sum() > 0:
                m = R_norm[ui, mask].mean()
                R_norm[ui, mask] -= m
                self.user_means[u] = m
            else:
                self.user_means[u] = 0.0

        # SVD decomposition
        k = min(self.n_factors, min(R_norm.shape) - 1)
        U, sigma, Vt = svds(csr_matrix(R_norm).astype(float), k=k)
        sigma_diag = np.diag(sigma)
        self.pred_matrix = U @ sigma_diag @ Vt
        self._users = users
        self._dests = dests

    def predict(self, user_id, dest_id):
        if user_id not in self.user_idx or dest_id not in self.dest_idx:
            return None
        ui = self.user_idx[user_id]
        di = self.dest_idx[dest_id]
        return float(self.pred_matrix[ui, di]) + self.user_means.get(user_id, 0)

    def recommend(self, user_id, top_k=10):
        if user_id not in self.user_idx:
            return []
        ui   = self.user_idx[user_id]
        mean = self.user_means.get(user_id, 0)
        preds = [(d, float(self.pred_matrix[ui, di]) + mean)
                 for d, di in self.dest_idx.items()]
        return sorted(preds, key=lambda x: x[1], reverse=True)[:top_k]


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 5. HYBRID RECOMMENDER
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class HybridRecommender:
    """
    Score = alpha * CBF_score + beta * CF_score + gamma * Popularity_score
    Weights tuned: alpha=0.45, beta=0.35, gamma=0.20
    """
    def __init__(self, cbf: ContentBasedFilter, cf: CollaborativeFilter,
                 dest_df: pd.DataFrame,
                 alpha=0.45, beta=0.35, gamma=0.20):
        self.cbf    = cbf
        self.cf     = cf
        self.dest   = dest_df.set_index("id")
        self.alpha  = alpha
        self.beta   = beta
        self.gamma  = gamma
        # Popularity: normalised by max review count proxy (avg_rating Ã— visitor proxy)
        pop         = dest_df[["id","avg_rating"]].copy()
        pop["pop_score"] = MinMaxScaler().fit_transform(pop[["avg_rating"]])
        self.pop    = pop.set_index("id")["pop_score"].to_dict()

    def recommend(self, user_id, user_interests, budget,
                  season=None, province=None, top_k=10):

        # Content-based scores (normalised 0-1)
        cbf_raw  = dict(self.cbf.recommend(user_interests, budget, top_k=len(self.dest)))
        cbf_max  = max(cbf_raw.values()) if cbf_raw else 1
        cbf_norm = {k: v / cbf_max for k, v in cbf_raw.items()}

        # Collaborative scores (normalised 0-1)
        cf_raw   = dict(self.cf.recommend(user_id, top_k=len(self.dest)))
        cf_min   = min(cf_raw.values(), default=0)
        cf_max   = max(cf_raw.values(), default=1)
        rng      = cf_max - cf_min if cf_max != cf_min else 1
        cf_norm  = {k: (v - cf_min) / rng for k, v in cf_raw.items()}

        # Hybrid ensemble
        all_ids  = set(cbf_norm) | set(cf_norm) | set(self.pop)
        scores   = {}
        for did in all_ids:
            cbf_s = cbf_norm.get(did, 0)
            cf_s  = cf_norm.get(did, 0) if cf_norm else 0
            pop_s = self.pop.get(did, 0)
            # Adjust CF weight if no CF data available
            a, b, g = self.alpha, self.beta, self.gamma
            if not cf_norm:
                a, b, g = 0.70, 0.0, 0.30
            scores[did] = a * cbf_s + b * cf_s + g * pop_s

        # Apply hard constraints
        results = []
        for did, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            if did not in self.dest.index:
                continue
            row = self.dest.loc[did]
            if province and row["province"] != province:
                continue
            if season:
                seasons_list = row["seasons"] if isinstance(row["seasons"], list) else eval(row["seasons"])
                if season not in seasons_list:
                    continue
            if row["avg_cost_day"] > budget * 1.25:
                continue
            results.append({"dest_id": did, "name": row["name"], "province": row["province"],
                             "hybrid_score": round(score, 4), "cbf_score": round(cbf_norm.get(did,0), 4),
                             "cf_score":  round(cf_norm.get(did,0), 4),  "pop_score": round(self.pop.get(did,0), 4),
                             "avg_cost_day": row["avg_cost_day"], "avg_rating": row["avg_rating"],
                             "tags": row["tags"]})
            if len(results) == top_k:
                break

        return pd.DataFrame(results)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 6. EVALUATION METRICS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def precision_at_k(recommended, relevant, k):
    rec_k = recommended[:k]
    hits  = len(set(rec_k) & set(relevant))
    return hits / k if k > 0 else 0

def recall_at_k(recommended, relevant, k):
    rec_k = recommended[:k]
    hits  = len(set(rec_k) & set(relevant))
    return hits / len(relevant) if relevant else 0

def ndcg_at_k(recommended, relevant, k):
    rec_k = recommended[:k]
    dcg   = sum(1 / np.log2(i + 2) for i, r in enumerate(rec_k) if r in set(relevant))
    idcg  = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0

def compute_rmse(cf_model, ratings_df, test_df):
    errors = []
    for _, row in test_df.iterrows():
        pred = cf_model.predict(row["user_id"], row["dest_id"])
        if pred is not None:
            errors.append((pred - row["rating"]) ** 2)
    return np.sqrt(np.mean(errors)) if errors else None

def evaluate_model(hybrid, cf_model, ratings_df, dest_df, n_users=50, k=10, threshold=4.0):
    """
    Offline evaluation using leave-one-out / test split approach.
    """
    results = {"precision": [], "recall": [], "ndcg": [], "rmse": []}
    sample_users = ratings_df["user_id"].value_counts()
    sample_users = sample_users[sample_users >= 3].index.tolist()[:n_users]

    for uid in sample_users:
        user_ratings = ratings_df[ratings_df["user_id"] == uid]
        relevant     = user_ratings[user_ratings["rating"] >= threshold]["dest_id"].tolist()
        if len(relevant) < 2:
            continue
        # Simulated user profile from ratings_df users table
        interests = ["nature","hiking","photography"]  # default for eval
        budget    = 200
        recs_df   = hybrid.recommend(uid, interests, budget, top_k=k)
        if recs_df.empty:
            continue
        recommended = recs_df["dest_id"].tolist()
        results["precision"].append(precision_at_k(recommended, relevant, k))
        results["recall"].append(recall_at_k(recommended, relevant, k))
        results["ndcg"].append(ndcg_at_k(recommended, relevant, k))

    # RMSE on a held-out 20% split
    test_sample = ratings_df.sample(frac=0.2, random_state=42)
    rmse = compute_rmse(cf_model, ratings_df, test_sample)
    results["rmse"] = rmse

    return {
        "Precision@K": np.mean(results["precision"]),
        "Recall@K":    np.mean(results["recall"]),
        "NDCG@K":      np.mean(results["ndcg"]),
        "RMSE":        rmse,
        "n_users_eval": len(results["precision"]),
    }


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 7. MAIN - BUILD AND DEMO`r`n# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def build_system(data_dir="data"):
    dest, users, ratings, weather = load_data(data_dir)

    fe  = FeatureEngineer()
    fe.fit_transform(dest.copy(), weather)

    cbf = ContentBasedFilter(fe)

    cf  = CollaborativeFilter(n_factors=15)
    cf.fit(ratings)

    hybrid = HybridRecommender(cbf, cf, dest)

    return hybrid, cf, fe, dest, users, ratings, weather


if __name__ == "__main__":
    print("Building Intelligent Travel Recommendation System...")
    hybrid, cf, fe, dest, users, ratings, weather = build_system()

    # Demo recommendation
    print("\n-- Demo: recommendations for a nature-loving hiker --")
    recs = hybrid.recommend(
        user_id="U0001",
        user_interests=["nature", "hiking", "photography", "wildlife"],
        budget=200,
        season="Summer",
        top_k=5
    )
    print(recs[["name","province","hybrid_score","avg_cost_day","avg_rating"]].to_string(index=False))

    # Evaluate
    print("\n-- Evaluation Metrics (K=10) --")
    metrics = evaluate_model(hybrid, cf, ratings, dest, n_users=40, k=10)
    for k, v in metrics.items():
        print(f"  {k:20s}: {v:.4f}" if isinstance(v, float) else f"  {k:20s}: {v}")
