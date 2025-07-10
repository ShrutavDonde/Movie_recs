"""
recommender.py
Hybrid (collaborative + content) movie recommender for Streamlit demo.
Requires the following small files under ./data/ :

  movie.csv               MovieLens movies metadata
  link.csv                MovieLens → IMDb/TMDb IDs
  ml_principals.tsv       Filtered top-5 actors per movie      (≈ 4 MB)
  ml_crew.tsv             Filtered directors per movie         (≈ 1 MB)
  ml_names.tsv            Filtered nconst → name look-up       (≈ 1 MB)
  svd_item_factors.pkl.gz Gzipped pickle of item latent vectors (≈ 6 MB)

Everything fits in the free-tier limits of Streamlit Cloud / HF Spaces.
"""

from __future__ import annotations

import csv
import gzip
import itertools
import io
import os
import pickle
import urllib.request
from collections import defaultdict
from math import sqrt
from typing import Dict, List, Set

import numpy as np
import pandas as pd
from numpy.linalg import norm
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm


# ------------------------------------------------------------------
#  Tiny wrapper class for the latent-factor pickle
#  (must be defined *before* pickle.load)
# ------------------------------------------------------------------
class ItemFactorModel:
    def __init__(self, factors, idx_map):
        self.factors = factors
        self.idx_map = idx_map
    def vector(self, movie_id: int):
        return self.factors[self.idx_map[movie_id]]


# ───────────────────────────────────────────────────────────────
#  Paths / core CSVs
# ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.join(os.path.dirname(__file__), "data")

movies = pd.read_csv(os.path.join(BASE_DIR, "movie.csv"))
links  = pd.read_csv(os.path.join(BASE_DIR, "link.csv"))

# ───────────────────────────────────────────────────────────────
#  Load tiny item-factor model (≈ 6 MB)  →  item_model.vector(id)
# ───────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(BASE_DIR, "svd_item_factors.pkl.gz")

with gzip.open(MODEL_PATH, "rb") as f:
    item_model = pickle.load(f)        # has .vector() and .idx_map

movies_with_cf: Set[int] = set(item_model.idx_map.keys())

# ───────────────────────────────────────────────────────────────
#  Genre one-hot matrix
# ───────────────────────────────────────────────────────────────
movies["genres_list"] = movies["genres"].str.split("|")

mlb = MultiLabelBinarizer()
genre_hot = mlb.fit_transform(movies["genres_list"])

genre_df = pd.DataFrame(
    genre_hot,
    index=movies["movieId"],
    columns=mlb.classes_
)

# ───────────────────────────────────────────────────────────────
#  Actors & directors dictionaries
# ───────────────────────────────────────────────────────────────
PRIN_PATH = os.path.join(BASE_DIR, "ml_principals.tsv")
CREW_PATH = os.path.join(BASE_DIR, "ml_crew.tsv")

prin = pd.read_csv(PRIN_PATH, sep="\t",
                   dtype={"tconst": str, "nconst": str, "ordering": int})
crew = pd.read_csv(CREW_PATH, sep="\t", dtype=str)

tconst_to_mid = links.assign(
    tconst=links["imdbId"].apply(lambda n: f"tt{int(n):07d}")
).set_index("tconst")["movieId"].to_dict()

prin["movieId"] = prin["tconst"].map(tconst_to_mid)
crew["movieId"] = crew["tconst"].map(tconst_to_mid)

prin = prin.sort_values(["movieId", "ordering"])
actors_by_movie: Dict[int, List[str]] = defaultdict(list)
for row in prin.itertuples(index=False):
    lst = actors_by_movie[row.movieId]
    if len(lst) < 5:
        lst.append(row.nconst)

directors_by_movie: Dict[int, List[str]] = {
    mid: row["directors"].split(",")
    for mid, row in crew.set_index("movieId").iterrows()
}

# ───────────────────────────────────────────────────────────────
#  nconst  →  human name   (downloads once if missing)
# ───────────────────────────────────────────────────────────────
NAMES_TSV = os.path.join(BASE_DIR, "ml_names.tsv")
NAMES_URL = "https://datasets.imdbws.com/name.basics.tsv.gz"

if not os.path.exists(NAMES_TSV):
    needed: Set[str] = (
        set(itertools.chain.from_iterable(actors_by_movie.values())) |
        set(itertools.chain.from_iterable(directors_by_movie.values()))
    )
    with urllib.request.urlopen(NAMES_URL) as resp, \
         gzip.GzipFile(fileobj=resp) as gz, \
         open(NAMES_TSV, "w", newline="", encoding="utf8") as fout:
        reader  = csv.DictReader(io.TextIOWrapper(gz, encoding="utf8"), delimiter="\t")
        writer  = csv.DictWriter(fout, fieldnames=["nconst", "primaryName"], delimiter="\t")
        writer.writeheader()
        for row in tqdm(reader, desc="filtering name.basics.tsv.gz"):
            if row["nconst"] in needed:
                writer.writerow({"nconst": row["nconst"], "primaryName": row["primaryName"]})

names_df = pd.read_csv(NAMES_TSV, sep="\t", dtype=str)
NCONST_TO_NAME: Dict[str, str] = dict(zip(names_df["nconst"], names_df["primaryName"]))

# ───────────────────────────────────────────────────────────────
#  Similarity functions
# ───────────────────────────────────────────────────────────────
def cf_sim(mid_a: int, mid_b: int) -> float:
    try:
        va = item_model.vector(mid_a)
        vb = item_model.vector(mid_b)
        return float(va @ vb / (norm(va) * norm(vb) + 1e-12))
    except KeyError:
        return 0.0

def genre_sim(mid_a: int, mid_b: int) -> float:
    va, vb = genre_df.loc[mid_a].values, genre_df.loc[mid_b].values
    return float(va @ vb / (norm(va) * norm(vb) + 1e-12))

def _set_cosine(a: set, b: set) -> float:
    return 0.0 if (not a or not b) else len(a & b) / sqrt(len(a) * len(b))

def actor_sim(mid_a: int, mid_b: int) -> float:
    return _set_cosine(set(actors_by_movie.get(mid_a, [])),
                       set(actors_by_movie.get(mid_b, [])))

def director_sim(mid_a: int, mid_b: int) -> float:
    return _set_cosine(set(directors_by_movie.get(mid_a, [])),
                       set(directors_by_movie.get(mid_b, [])))

def blended_score(mid_a: int, mid_b: int,
                  w_cf=.25, w_gen=.25, w_act=.25, w_dir=.25) -> float:
    return (
        w_cf    * cf_sim(mid_a, mid_b) +
        w_gen   * genre_sim(mid_a, mid_b) +
        w_act   * actor_sim(mid_a, mid_b) +
        w_dir   * director_sim(mid_a, mid_b)
    )

# ───────────────────────────────────────────────────────────────
#  Public API
# ───────────────────────────────────────────────────────────────
def recommend_like(
        title_query: str,
        top_k: int = 20,
        include_genres: Set[str] | None = None,
        w_cf=.25, w_gen=.25, w_act=.25, w_dir=.25
    ) -> pd.DataFrame:
    if include_genres is None:
        include_genres = {"Crime", "Thriller"}

    cand = movies[movies["title"].str.contains(title_query, case=False)]
    if cand.empty:
        raise ValueError(f"No movie title containing '{title_query}' found.")
    target_mid = cand.iloc[0]["movieId"]

    mask = (
        movies["movieId"].isin(movies_with_cf) &
        movies["genres_list"].apply(lambda g: bool(set(g) & include_genres))
    )
    pool = movies.loc[mask, "movieId"]

    scored = [
        (mid, blended_score(target_mid, mid, w_cf, w_gen, w_act, w_dir))
        for mid in pool
    ]
    top = sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]

    out = movies.set_index("movieId").loc[[m for m, _ in top],
                                          ["title", "genres"]].copy()
    out["director"] = out.index.map(
        lambda m: ", ".join(NCONST_TO_NAME.get(nc, nc)
                            for nc in directors_by_movie.get(m, [])[:2])
    )
    out["actors"] = out.index.map(
        lambda m: ", ".join(NCONST_TO_NAME.get(nc, nc)
                            for nc in actors_by_movie.get(m, [])[:3])
    )
    out["score"] = [s for _, s in top]
    return out.reset_index(drop=True)

__all__ = ["recommend_like"]
