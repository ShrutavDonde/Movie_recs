import streamlit as st
from recommender import recommend_like

st.set_page_config(page_title="Hybrid Movie Recommender", layout="wide")

st.title("🎬 Hybrid Movie Recommender")

with st.sidebar:
    st.markdown("**Weights**")
    w_cf   = st.slider("Collaborative", 0.0, 1.0, 0.25, 0.05)
    w_gen  = st.slider("Genre",         0.0, 1.0, 0.25, 0.05)
    w_act  = st.slider("Actors",        0.0, 1.0, 0.25, 0.05)
    w_dir  = st.slider("Directors",     0.0, 1.0, 0.25, 0.05)
    top_k  = st.slider("How many results?", 5, 30, 10)

query = st.text_input("I like…", "Godfather")
genres = st.multiselect(
    "Filter genres",
    ["Action","Adventure","Animation","Children","Comedy","Crime","Drama",
     "Fantasy","Film-Noir","Horror","Musical","Mystery","Romance",
     "Sci-Fi","Thriller","War","Western"],
    ["Crime","Thriller"]
)

if st.button("Recommend"):
    with st.spinner("Crunching…"):
        df = recommend_like(
            title_query=query,
            include_genres=set(genres),
            top_k=top_k,
            w_cf=w_cf, w_gen=w_gen, w_act=w_act, w_dir=w_dir
        )
    st.dataframe(df, hide_index=True, use_container_width=True)
