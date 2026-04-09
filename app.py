import streamlit as st
import pickle
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def fetch_poster(movie_id):
    response = requests.get(
        f'https://api.themoviedb.org/3/movie/{movie_id}?api_key=e8a6cad86556c1770bb5b3a3a957c9f8&language=en-US'
    )
    data = response.json()
    poster_path = data.get('poster_path')

    if poster_path:
        return "https://image.tmdb.org/t/p/w500/" + poster_path
    return ""



movie_list = pickle.load(open("movie_list.pkl", "rb"))
movie_name = movie_list['title'].values



@st.cache_data
def compute_similarity(movie_list):
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(movie_list['tags']).toarray()
    similarity = cosine_similarity(vectors)
    return similarity


similarity = compute_similarity(movie_list)



def recommend(movie):
    index = movie_list[movie_list['title'] == movie].index[0]
    distances = sorted(
        list(enumerate(similarity[index])),
        reverse=True,
        key=lambda x: x[1]
    )

    recommended_movie_names = []
    recommended_movie_posters = []

    for i in distances[1:6]:
        movie_id = movie_list.iloc[i[0]].movie_id

        recommended_movie_names.append(movie_list.iloc[i[0]].title)
        recommended_movie_posters.append(fetch_poster(movie_id))

    return recommended_movie_names, recommended_movie_posters



st.title('Movie Recommender System')

selected_movie = st.selectbox(
    "Select a Movie to get recommendations",
    movie_name
)

if st.button("Show Recommendations"):
    recommended_movie_names, recommended_movie_posters = recommend(selected_movie)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.text(recommended_movie_names[0])
        st.image(recommended_movie_posters[0])
    with col2:
        st.text(recommended_movie_names[1])
        st.image(recommended_movie_posters[1])
    with col3:
        st.text(recommended_movie_names[2])
        st.image(recommended_movie_posters[2])
    with col4:
        st.text(recommended_movie_names[3])
        st.image(recommended_movie_posters[3])
    with col5:
        st.text(recommended_movie_names[4])
        st.image(recommended_movie_posters[4])


