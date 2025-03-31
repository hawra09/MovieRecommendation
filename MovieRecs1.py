import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

movie_data = pd.read_csv('movies.csv')
rating_data = pd.read_csv('ratings.csv')
merged_data = pd.merge(movie_data, rating_data, on='movieId')
merged_data = merged_data.drop(columns='timestamp')

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(merged_data['genres'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

movie_indices = pd.Series(merged_data.index, index=merged_data['title']).drop_duplicates()
print(merged_data)

def get_similar_movies(title, top_n=10):
    if title not in movie_indices:
        return []
    indx = movie_indices[title]
    sim_scores = list(enumerate(cosine_sim[indx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse = True)[1:top_n+1]

    return merged_data['title'].iloc[[i[0] for i in sim_scores]].to_list()
print(get_similar_movies('Inception', 10))