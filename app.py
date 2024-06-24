import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import streamlit as st

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('data.csv')

movies_metadata = load_data()

# Preprocess the data
movies_metadata['director_name'] = movies_metadata['director_name'].fillna('')
movies_metadata['actor_1_name'] = movies_metadata['actor_1_name'].fillna('')
movies_metadata['actor_2_name'] = movies_metadata['actor_2_name'].fillna('')
movies_metadata['actor_3_name'] = movies_metadata['actor_3_name'].fillna('')
movies_metadata['genres'] = movies_metadata['genres'].fillna('')
movies_metadata['movie_title'] = movies_metadata['movie_title'].fillna('')

# Combine the relevant features into a single string
movies_metadata['combined_features'] = movies_metadata['director_name'] + ' ' + \
                                       movies_metadata['actor_1_name'] + ' ' + \
                                       movies_metadata['actor_2_name'] + ' ' + \
                                       movies_metadata['actor_3_name'] + ' ' + \
                                       movies_metadata['genres']

# Create a TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_metadata['combined_features'])

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Create a reverse mapping of movie titles to indices
movies_metadata['movie_title'] = movies_metadata['movie_title'].str.lower()
indices = pd.Series(movies_metadata.index, index=movies_metadata['movie_title']).drop_duplicates()

# Define the recommendation function
def get_recommendations(title, cosine_sim=cosine_sim):
    title = title.strip().lower()
    if title not in indices:
        return pd.Series(dtype='str')
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movies_metadata['movie_title'].iloc[movie_indices]

# Streamlit app layout
st.title('Movie Recommendation System')

# Input box for movie title
movie_title = st.text_input('Enter a movie title')

# Display recommendations when the button is clicked
if st.button('Recommend'):
    if movie_title:
        recommendations = get_recommendations(movie_title)
        if not recommendations.empty :
            st.write(f'Recommendations for "{movie_title}":')
            for idx, title in enumerate(recommendations, start=1):
                st.write(f'{idx}. {title.title()}')
        else:
            st.write(f'No recommendations found for "{movie_title}". Please check the title and try again.')
    else:
        st.write('Please enter a movie title.')
