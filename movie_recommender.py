import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')


class MovieRecommender:
    def __init__(self):
        self.movies_df = None
        self.ratings_df = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.user_item_matrix = None
        self.knn_model = None

    def load_sample_data(self):
        """Load sample movie data"""
        # Sample movies dataset
        movies_data = {
            'movie_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'title': [
                'The Shawshank Redemption', 'The Godfather', 'The Dark Knight',
                'Pulp Fiction', 'Forrest Gump', 'Inception', 'The Matrix',
                'Goodfellas', 'The Silence of the Lambs', 'Interstellar'
            ],
            'genres': [
                'Drama', 'Crime Drama', 'Action Thriller',
                'Crime Drama', 'Drama Romance', 'Sci-Fi Thriller', 'Sci-Fi Action',
                'Crime Drama', 'Thriller Horror', 'Sci-Fi Drama'
            ],
            'description': [
                'Two imprisoned men bond over years finding redemption',
                'The aging patriarch of crime dynasty transfers control',
                'Batman fights chaos unleashed by criminal mastermind Joker',
                'Various criminals intertwine in four tales of violence',
                'Man with low IQ accomplishes great things in life',
                'Thief steals secrets through dream-sharing technology',
                'Hacker discovers reality is simulated by machines',
                'Story of life in mob through eyes of Henry Hill',
                'FBI trainee seeks help of cannibal to catch serial killer',
                'Team of explorers travel through wormhole in space'
            ]
        }

        # Sample ratings dataset
        ratings_data = {
            'user_id': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5,
                       1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            'movie_id': [1, 2, 3, 1, 4, 5, 2, 3, 6, 1, 7, 8, 4, 9, 10,
                        6, 7, 8, 9, 1, 10, 3, 4, 5, 6],
            'rating': [5, 4, 5, 5, 4, 3, 4, 5, 4, 5, 5, 4, 3, 4, 5,
                      4, 5, 5, 4, 4, 3, 4, 4, 5, 5]
        }

        self.movies_df = pd.DataFrame(movies_data)
        self.ratings_df = pd.DataFrame(ratings_data)

    def content_based_setup(self):
        """Setup content-based filtering using movie descriptions and genres"""
        # Combine genres and description for better recommendations
        self.movies_df['content'] = self.movies_df['genres'] + ' ' + self.movies_df['description']

        # Create TF-IDF matrix
        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.movies_df['content'])

        # Compute cosine similarity
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

    def collaborative_filtering_setup(self):
        """Setup collaborative filtering using user ratings"""
        # Create user-item matrix
        self.user_item_matrix = self.ratings_df.pivot_table(
            index='user_id',
            columns='movie_id',
            values='rating'
        ).fillna(0)

        # Create sparse matrix for efficiency
        user_item_sparse = csr_matrix(self.user_item_matrix.values)

        # Train KNN model
        self.knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
        self.knn_model.fit(user_item_sparse)

    def get_content_recommendations(self, movie_title, n=5):
        """Get recommendations based on movie content similarity"""
        try:
            # Get movie index
            idx = self.movies_df[self.movies_df['title'] == movie_title].index[0]

            # Get similarity scores
            sim_scores = list(enumerate(self.cosine_sim[idx]))

            # Sort by similarity
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

            # Get top N similar movies (excluding the movie itself)
            sim_scores = sim_scores[1:n+1]

            # Get movie indices
            movie_indices = [i[0] for i in sim_scores]

            # Return recommendations
            recommendations = self.movies_df.iloc[movie_indices][['title', 'genres']].copy()
            recommendations['similarity_score'] = [round(i[1], 3) for i in sim_scores]

            return recommendations

        except IndexError:
            return pd.DataFrame()

    def get_collaborative_recommendations(self, user_id, n=5):
        """Get recommendations based on similar users' preferences"""
        try:
            # Get user index
            user_idx = self.user_item_matrix.index.get_loc(user_id)

            # Find similar users
            distances, indices = self.knn_model.kneighbors(
                self.user_item_matrix.iloc[user_idx, :].values.reshape(1, -1),
                n_neighbors=min(4, len(self.user_item_matrix))
            )

            # Get movies rated by similar users
            similar_users = self.user_item_matrix.iloc[indices.flatten()[1:]]

            # Find movies not yet rated by target user
            user_ratings = self.user_item_matrix.iloc[user_idx]
            unrated_movies = user_ratings[user_ratings == 0].index

            # Calculate average ratings from similar users
            recommendations = []
            for movie_id in unrated_movies:
                avg_rating = similar_users[movie_id].mean()
                if avg_rating > 0:
                    movie_title = self.movies_df[self.movies_df['movie_id'] == movie_id]['title'].values[0]
                    movie_genre = self.movies_df[self.movies_df['movie_id'] == movie_id]['genres'].values[0]
                    recommendations.append({
                        'title': movie_title,
                        'genres': movie_genre,
                        'predicted_rating': round(avg_rating, 2)
                    })

            # Sort by predicted rating
            recommendations = sorted(recommendations, key=lambda x: x['predicted_rating'], reverse=True)

            return pd.DataFrame(recommendations[:n])

        except Exception:
            return pd.DataFrame()

    def get_popular_movies(self, n=5):
        """Get most popular movies based on average ratings"""
        movie_stats = self.ratings_df.groupby('movie_id').agg({
            'rating': ['mean', 'count']
        }).reset_index()

        movie_stats.columns = ['movie_id', 'avg_rating', 'num_ratings']

        # Filter movies with at least 2 ratings
        movie_stats = movie_stats[movie_stats['num_ratings'] >= 2]

        # Sort by average rating
        movie_stats = movie_stats.sort_values('avg_rating', ascending=False)

        # Merge with movie titles
        popular = movie_stats.merge(self.movies_df[['movie_id', 'title', 'genres']], on='movie_id')

        return popular[['title', 'genres', 'avg_rating', 'num_ratings']].head(n)


# ---------- Streamlit app UI ----------

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")
st.title("🎬 Movie Recommender System")

recommender = MovieRecommender()

# Sidebar controls
st.sidebar.header("Controls")
if st.sidebar.button("Load sample data"):
    recommender.load_sample_data()
    st.sidebar.success("Sample data loaded")

if st.sidebar.button("Setup content-based"):
    if recommender.movies_df is None:
        st.sidebar.error("Load data first")
    else:
        recommender.content_based_setup()
        st.sidebar.success("Content-based setup complete")

if st.sidebar.button("Setup collaborative"):
    if recommender.ratings_df is None:
        st.sidebar.error("Load data first")
    else:
        recommender.collaborative_filtering_setup()
        st.sidebar.success("Collaborative setup complete")

st.sidebar.markdown("---")

# Provide quick setup for convenience (runs on first load)
if recommender.movies_df is None or recommender.ratings_df is None:
    recommender.load_sample_data()
    recommender.content_based_setup()
    recommender.collaborative_filtering_setup()

# Main layout
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("Available Movies")
    st.dataframe(recommender.movies_df[['movie_id', 'title', 'genres']])

    st.subheader("Popular Movies")
    top_n = st.slider("Show top N popular movies", min_value=1, max_value=10, value=5)
    popular = recommender.get_popular_movies(n=top_n)
    st.table(popular)

with col2:
    st.subheader("Content-based Recommendations")
    movie_choice = st.selectbox("Choose a movie", recommender.movies_df['title'].tolist())
    num_rec = st.number_input("Number of recommendations", min_value=1, max_value=10, value=5)
    if st.button("Get content-based recommendations"):
        recs = recommender.get_content_recommendations(movie_choice, n=num_rec)
        if recs.empty:
            st.info("No recommendations found. Make sure the movie exists in the dataset.")
        else:
            st.table(recs)

    st.markdown("---")

    st.subheader("Collaborative Filtering Recommendations")
    user_choice = st.selectbox("Choose a user ID", recommender.user_item_matrix.index.tolist())
    num_rec_user = st.number_input("Number of recommendations (user)", min_value=1, max_value=10, value=5, key='user_n')
    if st.button("Get collaborative recommendations"):
        recs_user = recommender.get_collaborative_recommendations(user_choice, n=num_rec_user)
        if recs_user.empty:
            st.info("No collaborative recommendations found for this user.")
        else:
            st.table(recs_user)

st.markdown("---")
st.subheader("Ratings (sample)")
st.dataframe(recommender.ratings_df.head(25))

st.caption("This Streamlit app wraps a simple content-based + collaborative recommender using sample data.")

# End of file
