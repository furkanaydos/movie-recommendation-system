import os
import streamlit as st
import requests
import pickle

# Constants
PLACEHOLDER_POSTER = "https://via.placeholder.com/150" # Placeholder poster URL for missing movie posters
API_KEY = "8265bd1679663a7ea12ac168da84d2e8"  # API key for TMDB API
BASE_POSTER_URL = "https://image.tmdb.org/t/p/w500" # Base URL for fetching movie posters
FAVORITE_FILE = "processed_data/favorite.pkl" # Path to save/load favorite movies

# Ensure the processed_data folder exists
if not os.path.exists("processed_data"):
    os.makedirs("processed_data")

#Load pre-processed data and models
movies = pickle.load(open('processed_data/movie_list.pkl', 'rb')) # Movie data
optimized_knn = pickle.load(open('processed_data/optimized_knn.pkl', 'rb')) # Optimized KNN model
tfidf = pickle.load(open('processed_data/tfidf_vectorizer.pkl', 'rb')) # TF-IDF vectorizer
vector = pickle.load(open('processed_data/vectorized_data.pkl', 'rb')) # Precomputed movie vectors

#Fetch movie poster from TMDB
def fetch_poster(movie_id):
    try:
        response = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
        )
        response.raise_for_status()
        poster_path = response.json().get('poster_path') # Get poster path from the API response
        return f"{BASE_POSTER_URL}/{poster_path}" if poster_path else PLACEHOLDER_POSTER
    except requests.RequestException as e:
        st.error(f"Error fetching poster: {e}")
        return PLACEHOLDER_POSTER

# Recommend movies based on KNN model
def recommend_knn(movie, k=5):
    try:
        # Find the index of the selected movie
        index = movies[movies['title'] == movie].index[0]
        
         # Get recommendations using KNN
        distances, indices = optimized_knn.kneighbors([vector[index]], n_neighbors=k + 10)
        recommendations = [movies.iloc[i].title for i in indices[0][1:]]
        return recommendations
    except IndexError:
        return ["Movie not found in dataset."]

# Load or initialize favorite file
def load_favorites():
    if os.path.exists(FAVORITE_FILE):
        try:
            return pickle.load(open(FAVORITE_FILE, 'rb'))
        except Exception as e:
            st.error(f"Error loading favorites: {e}")
    return {}

# Save favorite movies to file
def save_favorites(favorites):
    try:
        with open(FAVORITE_FILE, 'wb') as f:
            pickle.dump(favorites, f)
        print(f"Favorites saved successfully: {favorites}")
    except Exception as e:
        print(f"Error saving favorites: {e}")

# Streamlit UI
st.title("Movie Recommendation System 🎥🎬🍿")

# Initialize session state for favorites
if 'favorites' not in st.session_state:
    st.session_state.favorites = load_favorites()

# Movie selection dropdown
selected_movie = st.selectbox(
    "Search or select a movie from the dropdown",
    movies['title'].values
)

# Clicking the Show Recommendations button loads recommendations
if st.button('Show Recommendations'):
    recommendations = recommend_knn(selected_movie, k=5)
    recommendations = list(dict.fromkeys(recommendations[:5]))  # Get only the top 5 recommendations
    st.session_state.recommendations = recommendations  
    recommended_posters = []

    for movie in recommendations:
        try:
            movie_id = movies[movies['title'] == movie].iloc[0].id
            poster_url = fetch_poster(movie_id)
        except IndexError:
            poster_url = PLACEHOLDER_POSTER
        recommended_posters.append(poster_url)

    st.session_state.recommended_posters = recommended_posters  

# Show recommendations if any
if 'recommendations' in st.session_state:
    cols = st.columns(len(st.session_state.recommendations))
    selected_favorites = []  # List of user-selected favorites

    for col, name, poster in zip(cols, st.session_state.recommendations, st.session_state.recommended_posters):
        with col:
            st.image(poster, use_container_width=True) # Display movie poster
            st.markdown(f"**{name}**", unsafe_allow_html=True) # Display movie name

            # Checkbox for adding to favorites
            checkbox_key = f"checkbox-{selected_movie}-{name}"
            if checkbox_key not in st.session_state:
                st.session_state[checkbox_key] = False  # Default unchecked

            # Show checkbox and update state
            if st.checkbox(f"❤️", key=checkbox_key):
                selected_favorites.append(name)  

    # Confirm button to save selected favorites
    if st.button("Confirm Selections"):
        if selected_movie not in st.session_state.favorites:
            st.session_state.favorites[selected_movie] = []
        for movie in selected_favorites:
            if movie not in st.session_state.favorites[selected_movie]:
                st.session_state.favorites[selected_movie].append(movie)

        # Save favorites and notify user
        save_favorites(st.session_state.favorites)
        st.success(f"Favorites updated for '{selected_movie}'!")

# Display favorites in a subheader
if selected_movie in st.session_state.favorites:
    st.subheader(f"Favorites for {selected_movie}")
    
    #Present favorites in boxes
    favorites_list = st.session_state.favorites[selected_movie]
    if favorites_list:
        for movie in favorites_list:
            st.markdown(
                f"""
                <div style="padding: 10px; margin: 5px 0; border: 1px solid #ccc; border-radius: 5px; background-color: #0000;">
                    <strong>{movie}</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown("No favorites added yet.")



# Display favorites in the sidebar with delete option
with st.sidebar:
    st.header("All Favorites ❤️")
    if st.session_state.favorites and any(st.session_state.favorites.values()): 
        for movie, favorites in st.session_state.favorites.items():
            for favorite in favorites:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"- **{favorite}**")
                with col2:
                    if st.button(f"🗑️", key=f"remove-{favorite}"):
                        # Remove from favorites
                        for movie_key in st.session_state.favorites:
                            if favorite in st.session_state.favorites[movie_key]:
                                st.session_state.favorites[movie_key].remove(favorite)
                                break
                        save_favorites(st.session_state.favorites)
                        st.rerun() 
    else:
        st.write("No favorites added yet.")


st.markdown(
    """
    <style>
    .stApp h1 {
        font-size: 50px; 
        text-align: center; 
    }

    .stButton > button {
        padding: 10px 20px;
        font-size: 20px;
        font-weight: bold;
        border-radius: 5px;
        margin: auto;
        display: block;
        margin-bottom: 10px;
        margin-top: 10px;
    }

    .stMarkdown > div {
        text-align: center;
    }

    section[data-testid="stSidebar"] .stMarkdown > div {
        text-align: left; 
    }

    </style>
    """,
    unsafe_allow_html=True
)