"""
api/main.py
Intelligent Travel Recommendation System - Canada
MSc Data Science Project

FastAPI REST API exposing the hybrid recommender.

Run with:
    pip install fastapi uvicorn
    uvicorn api.main:app --reload
"""

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from recommender import build_system, evaluate_model
import pandas as pd

# Build the system once at startup
print("Loading recommendation system...")
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
hybrid, cf, fe, dest_df, users_df, ratings_df, weather_df = build_system(DATA_DIR)
hotels_df = pd.read_csv(os.path.join(DATA_DIR, "hotels.csv"))
meals_df = pd.read_csv(os.path.join(DATA_DIR, "meals.csv"))
attractions_df = pd.read_csv(os.path.join(DATA_DIR, "attractions.csv"))
review_counts = ratings_df.groupby("dest_id").size().to_dict()
print("System ready.")

app = FastAPI(
    title="Intelligent Travel Recommendation System - Canada",
    description="MSc Data Science Project | Hybrid ML Recommender using Canadian Tourism Data",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# â”€â”€ Schemas â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class RecommendRequest(BaseModel):
    user_id:   Optional[str]  = Field(None,  description="Existing user ID (for CF). Leave empty for new users.")
    interests: List[str]      = Field(...,   description="List of interest tags e.g. ['nature','hiking']")
    budget:    float          = Field(200.0, description="Daily budget in CAD")
    season:    Optional[str]  = Field(None,  description="Preferred season: Winter|Spring|Summer|Fall")
    province:  Optional[str]  = Field(None,  description="Filter by province e.g. 'British Columbia'")
    top_k:     int            = Field(10,    ge=1, le=20, description="Number of results to return")

class DestinationOut(BaseModel):
    dest_id:       str
    name:          str
    province:      str
    hybrid_score:  float
    cbf_score:     float
    cf_score:      float
    avg_cost_day:  float
    avg_rating:    float
    tags:          str

class RecommendResponse(BaseModel):
    total:    int
    model:    str
    results:  List[DestinationOut]

class MetricsResponse(BaseModel):
    precision_at_k: float
    recall_at_k:    float
    ndcg_at_k:      float
    rmse:           Optional[float]
    n_users_eval:   int
    k:              int = 10


def serialize_tags(raw_tags):
    if isinstance(raw_tags, str):
        return eval(raw_tags) if raw_tags.startswith("[") else raw_tags.split("|")
    return raw_tags


def serialize_seasons(raw_seasons):
    if isinstance(raw_seasons, str):
        return eval(raw_seasons) if raw_seasons.startswith("[") else [raw_seasons]
    return raw_seasons


def serialize_destination_row(row):
    return {
        "id": row["id"],
        "name": row["name"],
        "province": row["province"],
        "lat": float(row["lat"]),
        "lng": float(row["lon"]),
        "tags": serialize_tags(row["tags"]),
        "avg_cost_day": float(row["avg_cost_day"]),
        "avg_rating": float(row["avg_rating"]),
        "seasons": serialize_seasons(row["seasons"]),
        "type": row["type"],
        "review_count": int(review_counts.get(row["id"], 0)),
    }


def related_records(df: pd.DataFrame, dest_id: str, kind: str):
    rows = df[df["destination_id"] == dest_id].copy()
    if rows.empty:
        return []
    if kind == "hotel":
        return [
            {"name": record["name"], "cost": float(record["cost"]), "lat": float(record["lat"]), "lng": float(record["lng"])}
            for record in rows[["name", "cost", "lat", "lng"]].to_dict(orient="records")
        ]
    return [
        {
            "name": record["name"],
            "cost": float(record["cost"]),
            "type": record["type"],
            "lat": float(record["lat"]),
            "lng": float(record["lng"]),
        }
        for record in rows[["name", "cost", "type", "lat", "lng"]].to_dict(orient="records")
    ]


def weather_records(dest_id: str):
    rows = weather_df[weather_df["dest_id"] == dest_id].copy()
    if rows.empty:
        return []
    rows = rows.rename(columns={"avg_temp_c": "temp", "precip_mm": "precip"})
    return [
        {
            "month": int(record["month"]),
            "season": record["season"],
            "temp": float(record["temp"]),
            "precip": float(record["precip"]),
            "visitor_index": float(record["visitor_index"]),
        }
        for record in rows[["month", "season", "temp", "precip", "visitor_index"]].to_dict(orient="records")
    ]


# â”€â”€ Endpoints â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Intelligent Travel Recommendation System - Canada"}


@app.post("/recommend", response_model=RecommendResponse, tags=["Recommender"])
def recommend(req: RecommendRequest):
    """
    Get personalised travel recommendations.

    - **New users**: provide `interests` and `budget`. CF component falls back to popularity.
    - **Existing users**: provide `user_id` for full hybrid recommendations.
    """
    try:
        uid = req.user_id or "U_NEW"
        recs = hybrid.recommend(
            user_id=uid,
            user_interests=req.interests,
            budget=req.budget,
            season=req.season,
            province=req.province,
            top_k=req.top_k,
        )
        if recs.empty:
            raise HTTPException(status_code=404, detail="No destinations matched the given filters.")

        results = []
        for _, row in recs.iterrows():
            results.append(DestinationOut(
                dest_id=row["dest_id"],
                name=row["name"],
                province=row["province"],
                hybrid_score=row["hybrid_score"],
                cbf_score=row["cbf_score"],
                cf_score=row["cf_score"],
                avg_cost_day=row["avg_cost_day"],
                avg_rating=row["avg_rating"],
                tags=str(row["tags"]),
            ))
        return RecommendResponse(total=len(results), model="Hybrid (CBF+CF+Popularity)", results=results)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/destinations", tags=["Data"])
def list_destinations(province: Optional[str] = None):
    """Return all destination records, optionally filtered by province."""
    df = dest_df.copy()
    if province:
        df = df[df["province"] == province]
    return [serialize_destination_row(row) for _, row in df.iterrows()]


@app.get("/destinations/{dest_id}", tags=["Data"])
def get_destination(dest_id: str):
    """Return a single destination by ID."""
    row = dest_df[dest_df["id"] == dest_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Destination not found.")
    dest = serialize_destination_row(row.iloc[0])
    dest["hotels"] = related_records(hotels_df, dest_id, "hotel")
    dest["meals"] = related_records(meals_df, dest_id, "meal")
    dest["attractions"] = related_records(attractions_df, dest_id, "attraction")
    dest["weather"] = weather_records(dest_id)
    return dest


@app.get("/users", tags=["Data"])
def list_users(limit: int = Query(20, ge=1, le=500)):
    """Return sample user profiles."""
    return users_df.head(limit).to_dict(orient="records")


@app.get("/metrics", response_model=MetricsResponse, tags=["Evaluation"])
def model_metrics():
    """
    Run offline evaluation of the hybrid model.
    Returns Precision@K, Recall@K, NDCG@K, RMSE.
    Note: may take a few seconds.
    """
    m = evaluate_model(hybrid, cf, ratings_df, dest_df, n_users=40, k=10)
    return MetricsResponse(
        precision_at_k=round(m["Precision@K"], 4),
        recall_at_k=round(m["Recall@K"], 4),
        ndcg_at_k=round(m["NDCG@K"], 4),
        rmse=round(m["RMSE"], 4) if m["RMSE"] else None,
        n_users_eval=m["n_users_eval"],
    )


@app.get("/provinces", tags=["Data"])
def list_provinces():
    """Return all provinces represented in the dataset."""
    return {"provinces": sorted(dest_df["province"].unique().tolist())}


@app.get("/interests", tags=["Data"])
def list_interests():
    """Return all interest tags available for filtering."""
    tags = set()
    for t in dest_df["tags"]:
        tags.update(eval(t) if isinstance(t, str) and t.startswith("[") else t.split("|"))
    return {"interests": sorted(tags)}
