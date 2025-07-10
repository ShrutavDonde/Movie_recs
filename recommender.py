# recommender.py  – section: load core CSV assets
import os, glob, pandas as pd

BASE_DIR = os.path.join(os.path.dirname(__file__), "data")

# movies & links are needed by the recommender
movies = pd.read_csv(os.path.join(BASE_DIR, "movie.csv"))
links  = pd.read_csv(os.path.join(BASE_DIR, "link.csv"))

# load all rating shards and concatenate
rating_files = sorted(glob.glob(os.path.join(BASE_DIR, "rating_part*.csv")))
ratings = pd.concat([pd.read_csv(f) for f in rating_files], ignore_index=True)

# (optional debug print)
if __name__ == "__main__":
    print("Loaded:",
          len(movies), "movies –",
          len(links),  "links –",
          len(ratings), "ratings")


# ---------- clean timestamps ------------------------------------------
ratings["timestamp"] = pd.to_datetime(
    ratings["timestamp"], errors="coerce", format="%Y-%m-%d %H:%M:%S"
)

bad_rows = ratings["timestamp"].isna().sum()
if bad_rows:
    # Drop any rows that failed to parse (expected to be zero for MovieLens 20M)
    ratings = ratings.dropna(subset=["timestamp"])

# ----------  chronological 80/20 split  -------------------------------
cutoff_ts = ratings["timestamp"].quantile(0.80)

train = ratings[ratings["timestamp"] <= cutoff_ts].copy()

# build zero-based indices for Surprise / implicit libs
user2idx = {u: i for i, u in enumerate(train["userId"].unique())}
item2idx = {m: i for i, m in enumerate(train["movieId"].unique())}

train["u_idx"] = train["userId"].map(user2idx)
train["i_idx"] = train["movieId"].map(item2idx)

# ────────────────────────────────────────────────────────────────
#  Pre-trained SVD model  ▸  load from data/svd_model.pkl
# ────────────────────────────────────────────────────────────────
import os, joblib
from surprise import SVD   # only for type hints; no training happens here

BASE_DIR   = os.path.join(os.path.dirname(__file__), "data")
MODEL_PATH = os.path.join(BASE_DIR, "svd_model.pkl")   # the file you exported

try:
    algo: SVD = joblib.load(MODEL_PATH)   # loads in ~1 s, ~25-35 MB
except FileNotFoundError as e:
    raise RuntimeError(
        f"Pre-trained model not found at {MODEL_PATH}. "
        "Train it once in the notebook, save it there, and commit to repo."
    ) from e

# recommender.py  – section: genre one-hot matrix
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd

# split the pipe-separated genre strings
movies["genres_list"] = movies["genres"].str.split("|")

mlb = MultiLabelBinarizer()
genre_hot = mlb.fit_transform(movies["genres_list"])

# DataFrame indexed by movieId for quick lookup in genre_sim()
genre_df = pd.DataFrame(
    genre_hot,
    index=movies["movieId"],
    columns=mlb.classes_
)



# recommender.py  –  section: actors_by_movie / directors_by_movie
import os, collections, pandas as pd

# TSV subsets produced earlier and checked into repo under data/
IMDB_DIR = BASE_DIR                                  # same data folder
PRIN_PATH = os.path.join(IMDB_DIR, "ml_principals.tsv")
CREW_PATH = os.path.join(IMDB_DIR, "ml_crew.tsv")

# --- load the tiny, pre-filtered IMDb files --------------------------
prin = pd.read_csv(PRIN_PATH,
                   sep="\t",
                   dtype={"tconst": str, "nconst": str, "ordering": int})

crew = pd.read_csv(CREW_PATH,
                   sep="\t",
                   dtype=str)

# --- map tconst → MovieLens movieId ----------------------------------
ml_map = links.assign(
    tconst=links["imdbId"].apply(lambda n: f"tt{int(n):07d}")
).set_index("tconst")["movieId"].to_dict()

prin["movieId"] = prin["tconst"].map(ml_map)
crew["movieId"] = crew["tconst"].map(ml_map)

# --- ACTORS: keep top-5 billed per film ------------------------------
prin = prin.sort_values(["movieId", "ordering"])
actors_by_movie: dict[int, list[str]] = collections.defaultdict(list)

for row in prin.itertuples(index=False):
    slot = actors_by_movie[row.movieId]
    if len(slot) < 5:           # cap at 5 actors
        slot.append(row.nconst)

# --- DIRECTORS: one row may hold multiple nconst IDs -----------------
directors_by_movie: dict[int, list[str]] = {
    mid: row["directors"].split(",")
    for mid, row in crew.set_index("movieId").iterrows()
}



# ───────────────────────────────────────────────────────────────
#  Similarity functions  &  public API  recommend_like()
# ───────────────────────────────────────────────────────────────
from math import sqrt
from numpy.linalg import norm
import pandas as pd

# ---------- collaborative & genre helpers ---------------------
def cf_sim(mid_a: int, mid_b: int) -> float:
    """Cosine similarity of latent factors; 0 if either movie unseen."""
    try:
        va, vb = algo.qi[item2idx[mid_a]], algo.qi[item2idx[mid_b]]
        return float(va @ vb / (norm(va) * norm(vb) + 1e-12))
    except KeyError:                   # cold-start ID
        return 0.0

def genre_sim(mid_a: int, mid_b: int) -> float:
    va, vb = genre_df.loc[mid_a].values, genre_df.loc[mid_b].values
    return float(va @ vb / (norm(va) * norm(vb) + 1e-12))

# ---------- actor / director helpers (set-based) ---------------
def _set_cosine(a: set, b: set) -> float:
    return 0.0 if (not a or not b) else len(a & b) / sqrt(len(a) * len(b))

def actor_sim(mid_a: int, mid_b: int) -> float:
    return _set_cosine(set(actors_by_movie.get(mid_a, [])),
                       set(actors_by_movie.get(mid_b, [])))

def director_sim(mid_a: int, mid_b: int) -> float:
    return _set_cosine(set(directors_by_movie.get(mid_a, [])),
                       set(directors_by_movie.get(mid_b, [])))

# ---------- blended score --------------------------------------
def blended_score(mid_target: int, mid_cand: int,
                  w_cf: float = .25, w_genre: float = .25,
                  w_actor: float = .25, w_dir: float = .25) -> float:
    return (
        w_cf    * cf_sim(mid_target, mid_cand) +
        w_genre * genre_sim(mid_target, mid_cand) +
        w_actor * actor_sim(mid_target, mid_cand) +
        w_dir   * director_sim(mid_target, mid_cand)
    )

# ---------- public API  ----------------------------------------
def recommend_like(
        title_query: str,
        top_k: int = 20,
        include_genres: set[str] = {"Crime", "Thriller"},
        w_cf: float = .25, w_genre: float = .25,
        w_actor: float = .25, w_dir: float = .25
    ) -> pd.DataFrame:
    """
    Return a DataFrame of the `top_k` movies most similar to `title_query`,
    blending four similarity signals (CF, genre, actors, directors).

    Columns: title • genres • director(s) • top actors • score
    """
    # 1 ─ resolve free-text title to movieId
    cand = movies[movies["title"].str.contains(title_query, case=False)]
    if cand.empty:
        raise ValueError(f"No movie title containing '{title_query}' found.")
    target_mid = cand.iloc[0]["movieId"]

    # 2 ─ candidate pool: has CF vector *and* matches genre filter
    mask = (
        movies["movieId"].isin(item2idx) &
        movies["genres_list"].apply(lambda g: bool(set(g) & include_genres))
    )
    pool = movies.loc[mask, "movieId"]

    # 3 ─ score and rank
    scored = [
        (mid, blended_score(target_mid, mid,
                            w_cf, w_genre, w_actor, w_dir))
        for mid in pool
    ]
    top = sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]

    # 4 ─ build pretty output with human names
    out = movies.set_index("movieId").loc[[m for m, _ in top],
                                          ["title", "genres"]].copy()

    out["director"] = out.index.map(
        lambda m: ", ".join(
            NCONST_TO_NAME.get(nc, nc)
            for nc in directors_by_movie.get(m, [])[:2]
        )
    )
    out["actors"] = out.index.map(
        lambda m: ", ".join(
            NCONST_TO_NAME.get(nc, nc)
            for nc in actors_by_movie.get(m, [])[:3]
        )
    )
    out["score"] = [s for _, s in top]
    return out.reset_index(drop=True)

# expose just the public function
__all__ = ["recommend_like"]




# recommender.py ──────────────────────────────────────────────────────────
#  Build or load  NCONST_TO_NAME  (IMDb person-ID → human name)
#  ────────────────────────────────────────────────────────────────────────
import os, urllib.request, gzip, csv, io, itertools, pandas as pd
from tqdm import tqdm                                        # add to requirements.txt

IMDB_DIR   = BASE_DIR                                        # same /data folder
NAMES_TSV  = os.path.join(IMDB_DIR, "ml_names.tsv")
NAMES_URL  = "https://datasets.imdbws.com/name.basics.tsv.gz"

# ---- create the filtered file only if it doesn't exist -----------------
if not os.path.exists(NAMES_TSV):
    # collect all nconst IDs we actually reference
    needed_nconst: set[str] = (
        set(itertools.chain.from_iterable(actors_by_movie.values())) |
        set(itertools.chain.from_iterable(directors_by_movie.values()))
    )

    # stream-download and filter on the fly (≈ 10 s, < 1 MB output)
    with urllib.request.urlopen(NAMES_URL) as resp, \
         gzip.GzipFile(fileobj=resp) as gz, \
         open(NAMES_TSV, "w", newline="", encoding="utf8") as fout:

        reader  = csv.DictReader(io.TextIOWrapper(gz, encoding="utf8"), delimiter="\t")
        writer  = csv.DictWriter(fout, fieldnames=["nconst", "primaryName"], delimiter="\t")
        writer.writeheader()

        for row in tqdm(reader, desc="filtering name.basics.tsv.gz"):
            if row["nconst"] in needed_nconst:
                writer.writerow({"nconst": row["nconst"], "primaryName": row["primaryName"]})

# ---- load into a fast lookup dict --------------------------------------
names_df = pd.read_csv(NAMES_TSV, sep="\t", dtype=str)
NCONST_TO_NAME: dict[str, str] = dict(zip(names_df["nconst"], names_df["primaryName"]))

if __name__ == "__main__":          # optional debug print
    print(f"{len(NCONST_TO_NAME):,} names in NCONST_TO_NAME")



