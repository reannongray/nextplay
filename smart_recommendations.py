import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import logging
import warnings
warnings.filterwarnings('ignore')

class GameRecommendationEngine:
    """
    A smart recommendation system that finds games you'll love based on multiple factors.
    Uses advanced ML techniques but explains everything in human terms.
    """
    
    def __init__(self, games_df):
        self.df = games_df.copy()
        self.feature_matrix = None
        self.similarity_matrix = None
        self.scaler = StandardScaler()
        self.clusters = None
        self.setup_recommendation_engine()
    
    def setup_recommendation_engine(self):
        """
        Prepares all the behind-the-scenes magic for making recommendations.
        """
        logging.info("Setting up recommendation engine...")
        
        # Clean and prepare the data
        self.prepare_features()
        
        # Create similarity matrices
        self.build_similarity_matrix()
        
        # Create game clusters for diversity
        self.create_game_clusters()
        
        logging.info("Recommendation engine ready!")
    
    def prepare_features(self):
        """
        Converts game attributes into numbers that algorithms can understand.
        """
        features = pd.DataFrame(index=self.df.index)
        
        # Numeric features (already numbers)
        numeric_features = ["rating", "avg_playtime", "metacritic_score", "rating_count"]
        for feature in numeric_features:
            if feature in self.df.columns:
                # Fill missing values with median and normalize
                values = self.df[feature].fillna(self.df[feature].median())
                features[feature] = values
        
        # Genre features (convert text to numbers using TF-IDF)
        if "genres" in self.df.columns:
            genre_text = self.df["genres"].fillna("").astype(str)
            tfidf_genres = TfidfVectorizer(max_features=50, stop_words=None)
            genre_matrix = tfidf_genres.fit_transform(genre_text).toarray()
            
            # Add genre features to our matrix
            genre_cols = [f"genre_{i}" for i in range(genre_matrix.shape[1])]
            genre_df = pd.DataFrame(genre_matrix, columns=genre_cols, index=self.df.index)
            features = pd.concat([features, genre_df], axis=1)
        
        # Platform features
        if "platforms" in self.df.columns:
            platform_text = self.df["platforms"].fillna("").astype(str)
            tfidf_platforms = TfidfVectorizer(max_features=30, stop_words=None)
            platform_matrix = tfidf_platforms.fit_transform(platform_text).toarray()
            
            platform_cols = [f"platform_{i}" for i in range(platform_matrix.shape[1])]
            platform_df = pd.DataFrame(platform_matrix, columns=platform_cols, index=self.df.index)
            features = pd.concat([features, platform_df], axis=1)
        
        # Scale all features to same range
        self.feature_matrix = self.scaler.fit_transform(features.fillna(0))
        logging.info(f"Created feature matrix with {self.feature_matrix.shape[1]} features")
    
    def build_similarity_matrix(self):
        """
        Calculates how similar each game is to every other game.
        """
        if self.feature_matrix is not None:
            self.similarity_matrix = cosine_similarity(self.feature_matrix)
            logging.info("Built game similarity matrix")
    
    def create_game_clusters(self, n_clusters=8):
        """
        Groups games into clusters to ensure diverse recommendations.
        """
        if self.feature_matrix is not None:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.clusters = kmeans.fit_predict(self.feature_matrix)
            self.df["cluster"] = self.clusters
            logging.info(f"Created {n_clusters} game clusters")
    
    def find_similar_games(self, game_title, num_recommendations=10, diversity_factor=0.3):
        """
        Finds games similar to the one you specify. The magic happens here!
        
        diversity_factor: 0 = most similar games, 1 = most diverse games
        """
        # Find the game in our dataset
        game_matches = self.df[self.df["name"].str.contains(game_title, case=False, na=False)]
        
        if game_matches.empty:
            logging.warning(f"Game '{game_title}' not found in dataset")
            return pd.DataFrame()
        
        # Use the first match if multiple found
        base_game_idx = game_matches.index[0]
        base_game = self.df.loc[base_game_idx]
        
        # Get similarity scores for all games
        similarity_scores = self.similarity_matrix[base_game_idx]
        
        # Create recommendation candidates
        candidates = []
        for idx, score in enumerate(similarity_scores):
            if idx != base_game_idx:  # Don't recommend the same game
                game = self.df.iloc[idx]
                
                # Calculate diversity bonus (different cluster = more diverse)
                diversity_bonus = 0
                if "cluster" in self.df.columns:
                    if game["cluster"] != base_game["cluster"]:
                        diversity_bonus = diversity_factor
                
                # Combined score: similarity + diversity
                final_score = score * (1 - diversity_factor) + diversity_bonus
                
                candidates.append({
                    "game_index": idx,
                    "similarity_score": score,
                    "diversity_bonus": diversity_bonus,
                    "final_score": final_score,
                    "name": game["name"],
                    "rating": game.get("rating", 0),
                    "genres": game.get("genres", ""),
                    "release_year": game.get("release_year", "Unknown")
                })
        
        # Sort by final score and return top recommendations
        candidates.sort(key=lambda x: x["final_score"], reverse=True)
        top_candidates = candidates[:num_recommendations]
        
        recommendations_df = pd.DataFrame(top_candidates)
        
        logging.info(f"Found {len(recommendations_df)} recommendations for '{game_title}'")
        return recommendations_df
    
    def discover_by_preferences(self, preferred_genres=None, min_rating=3.0, 
                               preferred_playtime=None, exclude_genres=None, 
                               num_recommendations=15):
        """
        Finds games based on your preferences rather than similarity to a specific game.
        Perfect for discovering something completely new!
        """
        filtered_games = self.df.copy()
        
        # Apply genre preferences
        if preferred_genres:
            genre_filter = "|".join(preferred_genres)  # Match any of the preferred genres
            filtered_games = filtered_games[
                filtered_games["genres"].str.contains(genre_filter, case=False, na=False)
            ]
        
        # Exclude unwanted genres
        if exclude_genres:
            for genre in exclude_genres:
                filtered_games = filtered_games[
                    ~filtered_games["genres"].str.contains(genre, case=False, na=False)
                ]
        
        # Rating filter
        if "rating" in filtered_games.columns:
            filtered_games = filtered_games[filtered_games["rating"] >= min_rating]
        
        # Playtime preferences
        if preferred_playtime and "avg_playtime" in filtered_games.columns:
            if preferred_playtime == "short":
                filtered_games = filtered_games[filtered_games["avg_playtime"] <= 15]
            elif preferred_playtime == "medium":
                filtered_games = filtered_games[
                    (filtered_games["avg_playtime"] > 15) & 
                    (filtered_games["avg_playtime"] <= 50)
                ]
            elif preferred_playtime == "long":
                filtered_games = filtered_games[filtered_games["avg_playtime"] > 50]
        
        # Sort by a combination of rating and popularity
        if "popularity_score" in filtered_games.columns:
            filtered_games = filtered_games.sort_values("popularity_score", ascending=False)
        elif "rating" in filtered_games.columns:
            filtered_games = filtered_games.sort_values("rating", ascending=False)
        
        recommendations = filtered_games.head(num_recommendations)
        
        logging.info(f"Found {len(recommendations)} games matching preferences")
        return recommendations
    
    def find_hidden_gems(self, min_rating=4.0, max_popularity_threshold=500, 
                        preferred_genres=None, num_recommendations=10):
        """
        Discovers amazing games that might be flying under the radar.
        These are highly rated games with relatively few ratings.
        """
        potential_gems = self.df.copy()
        
        # Must be highly rated
        if "rating" in potential_gems.columns:
            potential_gems = potential_gems[potential_gems["rating"] >= min_rating]
        
        # But not too popular (hidden gems!)
        if "rating_count" in potential_gems.columns:
            potential_gems = potential_gems[potential_gems["rating_count"] <= max_popularity_threshold]
            # But not too obscure either
            potential_gems = potential_gems[potential_gems["rating_count"] >= 10]
        
        # Genre preferences
        if preferred_genres:
            genre_filter = "|".join(preferred_genres)
            potential_gems = potential_gems[
                potential_gems["genres"].str.contains(genre_filter, case=False, na=False)
            ]
        
        # Sort by rating, then by recency
        sort_columns = []
        if "rating" in potential_gems.columns:
            sort_columns.append("rating")
        if "release_year" in potential_gems.columns:
            sort_columns.append("release_year")
        
        if sort_columns:
            potential_gems = potential_gems.sort_values(sort_columns, ascending=[False, False])
        
        gems = potential_gems.head(num_recommendations)
        
        logging.info(f"Discovered {len(gems)} hidden gems")
        return gems
    
    def get_trending_games(self, time_window_years=2, min_rating=3.5, num_recommendations=20):
        """
        Finds games that are recently released and gaining popularity.
        """
        from datetime import datetime
        current_year = datetime.now().year
        cutoff_year = current_year - time_window_years
        
        trending_games = self.df.copy()
        
        # Recent releases only
        if "release_year" in trending_games.columns:
            trending_games = trending_games[trending_games["release_year"] >= cutoff_year]
        
        # Good quality filter
        if "rating" in trending_games.columns:
            trending_games = trending_games[trending_games["rating"] >= min_rating]
        
        # Sort by combination of rating and rating count (popularity)
        if "rating" in trending_games.columns and "rating_count" in trending_games.columns:
            # Calculate momentum score: rating * log(rating_count + 1)
            trending_games["momentum_score"] = (
                trending_games["rating"] * np.log1p(trending_games["rating_count"])
            )
            trending_games = trending_games.sort_values("momentum_score", ascending=False)
        
        trending = trending_games.head(num_recommendations)
        
        logging.info(f"Found {len(trending)} trending games from last {time_window_years} years")
        return trending
    
    def create_game_playlist(self, theme="variety", num_games=10):
        """
        Creates curated game playlists based on different themes.
        Like Spotify playlists, but for games!
        """
        playlist = pd.DataFrame()
        
        if theme == "variety":
            # One game from each cluster for maximum variety
            if "cluster" in self.df.columns:
                for cluster_id in sorted(self.df["cluster"].unique()):
                    cluster_games = self.df[self.df["cluster"] == cluster_id]
                    if not cluster_games.empty:
                        # Pick the highest rated game from this cluster
                        if "rating" in cluster_games.columns:
                            best_game = cluster_games.loc[cluster_games["rating"].idxmax()]
                            playlist = pd.concat([playlist, best_game.to_frame().T])
        
        elif theme == "weekend_warriors":
            # Short games perfect for weekend gaming
            weekend_games = self.discover_by_preferences(
                preferred_playtime="short",
                min_rating=4.0,
                num_recommendations=num_games
            )
            playlist = weekend_games
        
        elif theme == "deep_dives":
            # Long, immersive games for dedicated sessions
            deep_games = self.discover_by_preferences(
                preferred_playtime="long",
                min_rating=4.0,
                num_recommendations=num_games
            )
            playlist = deep_games
        
        elif theme == "critic_favorites":
            # Games with high Metacritic scores
            if "metacritic_score" in self.df.columns:
                critic_games = self.df[self.df["metacritic_score"] >= 80]
                critic_games = critic_games.sort_values("metacritic_score", ascending=False)
                playlist = critic_games.head(num_games)
        
        logging.info(f"Created '{theme}' playlist with {len(playlist)} games")
        return playlist
    
    def explain_recommendation(self, recommended_game, base_game=None):
        """
        Explains why a game was recommended in human-readable terms.
        """
        explanation = []
        
        if base_game is not None:
            # Compare genres
            if "genres" in self.df.columns:
                base_genres = set(base_game.get("genres", "").split(", "))
                rec_genres = set(recommended_game.get("genres", "").split(", "))
                common_genres = base_genres.intersection(rec_genres)
                
                if common_genres:
                    explanation.append(f"Shares genres: {', '.join(common_genres)}")
            
            # Compare ratings
            if "rating" in recommended_game and "rating" in base_game:
                rating_diff = abs(recommended_game["rating"] - base_game["rating"])
                if rating_diff < 0.5:
                    explanation.append("Similar quality rating")
            
            # Compare playtime
            if "avg_playtime" in recommended_game and "avg_playtime" in base_game:
                playtime_ratio = recommended_game["avg_playtime"] / max(base_game["avg_playtime"], 1)
                if 0.5 <= playtime_ratio <= 2.0:
                    explanation.append("Similar time commitment")
        
        # General positive attributes
        if recommended_game.get("rating", 0) >= 4.0:
            explanation.append("Highly rated by players")
        
        if recommended_game.get("metacritic_score", 0) >= 80:
            explanation.append("Critically acclaimed")
        
        return "; ".join(explanation) if explanation else "Recommended based on overall compatibility"
    
    def get_recommendation_stats(self):
        """
        Returns statistics about the recommendation engine's capabilities.
        """
        stats = {
            "total_games": len(self.df),
            "feature_dimensions": self.feature_matrix.shape[1] if self.feature_matrix is not None else 0,
            "clusters_created": len(set(self.clusters)) if self.clusters is not None else 0,
            "similarity_matrix_size": self.similarity_matrix.shape if self.similarity_matrix is not None else (0, 0)
        }
        
        # Genre diversity
        if "genres" in self.df.columns:
            all_genres = set([
                g.strip() for genres in self.df["genres"].dropna() 
                for g in genres.split(",")
            ])
            stats["unique_genres"] = len(all_genres)
        
        return stats