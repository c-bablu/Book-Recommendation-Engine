from fastapi import FastAPI, HTTPException
from typing import Optional
from recommender import build_engine, get_recommendations

df, sim_matrix = build_engine("Amazon_BestSelling_Books_500.csv")

app = FastAPI()

@app.get("/api/stats")
def stats():
    return {
        "total_books": len(df),
        "avg_rating": round(df['rating'].mean(), 2),
        "total_reviews": int(df['reviews'].sum())
    }

@app.get("/api/books")
def books(search: Optional[str] = None):
    result = df
    if search:
        mask = (
            df['title'].str.contains(search, case=False, na=False) |
            df['author'].str.contains(search, case=False, na=False)
        )
        result = df[mask]
    return result.to_dict(orient="records")

@app.get("/api/recommend/{title}")
def recommend(title: str):
    results = get_recommendations(title, df, sim_matrix)
    if results is None:
        raise HTTPException(status_code=404, detail=f"Book '{title}' not found.")
    return results.to_dict(orient="records")
