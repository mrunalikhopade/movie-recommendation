import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib

# Load datasets
movies_raw = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# Merge on title
movies_df = movies_raw.merge(credits, on='title')

# Clean dataset
movies_df = movies_df[['id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew',
                       'vote_average', 'release_date']]

def get_director(crew):
    try:
        for person in eval(crew):
            if person['job'] == 'Director':
                return person['name']
    except:
        return ''
    return ''

def get_top_names(text):
    try:
        return [person['name'] for person in eval(text)[:3]]
    except:
        return []

movies_df['director'] = movies_df['crew'].apply(get_director)
movies_df['cast'] = movies_df['cast'].apply(get_top_names)
movies_df['genres'] = movies_df['genres'].apply(get_top_names)
movies_df['keywords'] = movies_df['keywords'].apply(get_top_names)

# Combine text for similarity
movies_df['comb'] = (
    movies_df['overview'].fillna('') + ' ' +
    movies_df['director'].fillna('') + ' ' +
    movies_df['cast'].apply(lambda x: ' '.join(x)) + ' ' +
    movies_df['genres'].apply(lambda x: ' '.join(x)) + ' ' +
    movies_df['keywords'].apply(lambda x: ' '.join(x))
)
movies_df['comb'] = movies_df['comb'].fillna('')

# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
vector_matrix = vectorizer.fit_transform(movies_df['comb'])
similarity = cosine_similarity(vector_matrix)

# Main function
def get_recommendations(movie_name):
    movie_name = movie_name.lower()
    all_titles = movies_df['title'].str.lower().tolist()

    close_matches = difflib.get_close_matches(movie_name, all_titles, n=1, cutoff=0.5)
    if not close_matches:
        return []

    match = close_matches[0]
    idx = movies_df[movies_df['title'].str.lower() == match].index[0]

    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]

    recommendations = []
    for i in movie_indices:
        movie = movies_df.iloc[i]
        recommendations.append({
            'title': movie['title'],
            'genres': ', '.join(movie['genres']),
            'rating': movie['vote_average'],
            'year': movie['release_date'].split('-')[0] if pd.notna(movie['release_date']) else 'N/A'
        })

    return recommendations
