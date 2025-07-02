# recommender.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib

movies_df = None
similarity = None

def load_data():
    global movies_df, similarity

    if movies_df is not None and similarity is not None:
        return  # Already loaded

    movies_raw = pd.read_csv('tmdb_5000_movies.csv')
    credits = pd.read_csv('tmdb_5000_credits.csv')

    movies_df_raw = movies_raw.merge(credits, on='title')
    movies_df_clean = movies_df_raw[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew',
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

    movies_df_clean['director'] = movies_df_clean['crew'].apply(get_director)
    movies_df_clean['cast'] = movies_df_clean['cast'].apply(get_top_names)
    movies_df_clean['genres'] = movies_df_clean['genres'].apply(get_top_names)
    movies_df_clean['keywords'] = movies_df_clean['keywords'].apply(get_top_names)

    movies_df_clean['comb'] = (
        movies_df_clean['overview'].fillna('') + ' ' +
        movies_df_clean['director'].fillna('') + ' ' +
        movies_df_clean['cast'].apply(lambda x: ' '.join(x)) + ' ' +
        movies_df_clean['genres'].apply(lambda x: ' '.join(x)) + ' ' +
        movies_df_clean['keywords'].apply(lambda x: ' '.join(x))
    )

    vectorizer = TfidfVectorizer(stop_words='english')
    vector_matrix = vectorizer.fit_transform(movies_df_clean['comb'].fillna(''))

    similarity_matrix = cosine_similarity(vector_matrix)

    movies_df = movies_df_clean
    similarity = similarity_matrix


def get_recommendations(movie_name):
    load_data()  # lazy load on first request

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
            'poster': "https://via.placeholder.com/300x450?text=No+Image",
            'genres': ', '.join(movie['genres']),
            'rating': movie['vote_average'],
            'year': movie['release_date'].split('-')[0] if pd.notna(movie['release_date']) else 'N/A'
        })

    return recommendations
