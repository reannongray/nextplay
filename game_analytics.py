import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Dict, List, Any, Tuple, Union, cast
import warnings
warnings.filterwarnings('ignore')

# Set up our visual style - dark theme for that gaming aesthetic
plt.style.use('dark_background')

# Create a custom neon-like color palette for gaming theme
neon_colors = ["#00d4ff", "#ff6b6b", "#4ecdc4", "#45b7d1", "#96ceb4", "#ffeaa7", "#fd79a8", "#6c5ce7"]
sns.set_palette(neon_colors)

class GameAnalytics:
    """
    Your one-stop shop for gaming data insights. 
    Makes pretty charts and finds interesting patterns in game data.
    """
    
    def __init__(self, games_df: pd.DataFrame) -> None:
        self.df: pd.DataFrame = games_df.copy()
        self.prepare_data()
    
    def prepare_data(self) -> None:
        """
        Gets the data ready for analysis - handles missing values and creates useful columns.
        """
        # Ensure we have the columns we need
        required_cols = ["rating", "release_year", "genres", "platforms"]
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        
        if missing_cols:
            print(f"Warning: Missing columns {missing_cols}. Some analyses might not work.")
        
        # Create analysis-friendly columns
        if "release_year" in self.df.columns:
            self.df["decade"] = (self.df["release_year"] // 10) * 10  # type: ignore
        
        if "rating" in self.df.columns:
            self.df["rating_category"] = pd.cut(self.df["rating"],  # type: ignore
                                               bins=[0, 2.5, 3.5, 4.0, 4.5, 5.0],
                                               labels=["Poor", "Fair", "Good", "Great", "Excellent"])
    
    def rating_landscape(self, interactive: bool = True) -> Optional[go.Figure]:
        """
        Shows the overall rating distribution - are most games good or meh?
        """
        if "rating" not in self.df.columns or self.df["rating"].isna().all():  # type: ignore
            print("No rating data available for analysis")
            return None
        
        if interactive:
            fig = px.histogram(self.df, x="rating", nbins=20,  # type: ignore
                             title="Game Rating Distribution - How Good Are Games Really?",
                             labels={"rating": "Game Rating (1-5 stars)", "count": "Number of Games"},
                             color_discrete_sequence=["#00d4ff"])
            
            fig.add_vline(x=self.df["rating"].mean(), line_dash="dash",  # type: ignore
                         annotation_text=f"Average: {self.df['rating'].mean():.2f}")  # type: ignore
            
            fig.update_layout(  # type: ignore
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="white"
            )
            
            return fig
        else:
            plt.figure(figsize=(12, 6))  # type: ignore
            sns.histplot(data=self.df, x="rating", bins=20, kde=True)  # type: ignore
            plt.axvline(self.df["rating"].mean(), color='red', linestyle='--',  # type: ignore
                       label=f'Average: {self.df["rating"].mean():.2f}')  # type: ignore
            plt.title("Game Rating Distribution", fontsize=16)  # type: ignore
            plt.xlabel("Rating (1-5 stars)")  # type: ignore
            plt.ylabel("Number of Games")  # type: ignore
            plt.legend()  # type: ignore
            plt.tight_layout()  # type: ignore
            plt.show()  # type: ignore
            return None
    
    def genre_popularity_contest(self, top_n: int = 12, interactive: bool = True) -> Optional[go.Figure]:
        """
        Which genres rule the gaming world? Shows the most popular genres.
        """
        if "genres" not in self.df.columns:
            print("No genre data available")
            return None
        
        # Split and count genres
        all_genres = self.df["genres"].str.split(", ").explode()
        genre_counts = all_genres.value_counts().head(top_n)
        
        if interactive:
            fig = px.bar(x=genre_counts.values, y=genre_counts.index.tolist(),
                        orientation='h',
                        title=f"Top {top_n} Most Popular Game Genres",
                        labels={"x": "Number of Games", "y": "Genre"},
                        color=genre_counts.values,
                        color_continuous_scale="viridis")
            
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                showlegend=False
            )
            
            return fig
        else:
            plt.figure(figsize=(12, 8))
            sns.barplot(x=genre_counts.values, y=genre_counts.index.tolist(), palette="viridis")
            plt.title(f"Top {top_n} Most Popular Game Genres", fontsize=16)
            plt.xlabel("Number of Games")
            plt.ylabel("Genre")
            plt.tight_layout()
            plt.show()
            return None
    
    def genre_quality_analysis(self, min_games: int = 5) -> Optional[Tuple[go.Figure, pd.DataFrame]]:
        """
        Which genres consistently produce the highest quality games?
        """
        if "genres" not in self.df.columns or "rating" not in self.df.columns:
            print("Need both genre and rating data for this analysis")
            return None
        
        # Explode genres and analyze ratings
        genre_ratings: List[Dict[str, Union[str, float]]] = []
        
        for i in range(len(self.df)):
            genres_str = self.df.iloc[i]["genres"]
            rating_val = self.df.iloc[i]["rating"]
            
            if pd.notna(genres_str) and pd.notna(rating_val) and isinstance(genres_str, str):
                genres = genres_str.split(", ")
                # Safe float conversion with type checking
                try:
                    rating_float = float(rating_val) if rating_val not in (None, "", "N/A") else 0.0
                except (ValueError, TypeError):
                    rating_float = 0.0
                
                for genre in genres:
                    genre_ratings.append({"genre": genre, "rating": rating_float})
        
        genre_df = pd.DataFrame(genre_ratings)
        
        # Calculate stats per genre
        genre_stats = genre_df.groupby("genre").agg({
            "rating": ["mean", "count", "std"]
        }).round(2)
        
        genre_stats.columns = ["avg_rating", "game_count", "rating_std"]
        genre_stats = genre_stats.reset_index()
        
        # Filter out genres with too few games
        genre_stats = genre_stats[genre_stats["game_count"] >= min_games]
        genre_stats = genre_stats.sort_values("avg_rating", ascending=False)
        
        # Create visualization
        fig = px.scatter(genre_stats, x="game_count", y="avg_rating", 
                        size="rating_std", hover_name="genre",
                        title="Genre Quality vs Quantity Analysis",
                        labels={"game_count": "Number of Games", 
                               "avg_rating": "Average Rating",
                               "rating_std": "Rating Consistency"},
                        color="avg_rating",
                        color_continuous_scale="RdYlGn")
        
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white"
        )
        
        return fig, genre_stats
    
    def gaming_evolution_timeline(self) -> Optional[go.Figure]:
        """
        How has gaming evolved over the decades? Shows trends in releases and quality.
        """
        if "release_year" not in self.df.columns:
            print("No release year data available")
            return None
        
        # Filter out unrealistic years and group by year
        valid_years = self.df[(self.df["release_year"] >= 1970) & 
                             (self.df["release_year"] <= 2025)]
        
        yearly_stats = valid_years.groupby("release_year").agg({
            "rating": "mean",
            "name": "count"
        }).reset_index()
        
        yearly_stats.columns = ["year", "avg_rating", "game_count"]
        
        # Create dual-axis plot
        fig = cast(go.Figure, make_subplots(specs=[[{"secondary_y": True}]]))
        
        # Game count over time
        fig.add_trace(
            go.Scatter(x=yearly_stats["year"], y=yearly_stats["game_count"],
                      name="Games Released", line=dict(color="#00d4ff")),
            secondary_y=False
        )
        
        # Average rating over time
        fig.add_trace(
            go.Scatter(x=yearly_stats["year"], y=yearly_stats["avg_rating"],
                      name="Average Rating", line=dict(color="#ff6b6b")),
            secondary_y=True
        )
        
        # Fix for Pylance linting - cast to proper type
        fig.update_xaxes(title_text="Year")
        fig.update_yaxes(title_text="Number of Games Released", secondary_y=False)  # type: ignore
        fig.update_yaxes(title_text="Average Rating", secondary_y=True)  # type: ignore
        
        fig.update_layout(
            title="The Evolution of Gaming: Quantity vs Quality",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white"
        )
        
        return fig
    
    def platform_ecosystem_analysis(self) -> Optional[Tuple[go.Figure, pd.DataFrame]]:
        """
        Analyzes the gaming platform landscape - which platforms get the best games?
        """
        if "platforms" not in self.df.columns:
            print("No platform data available")
            return None
        
        # Extract and analyze platforms
        platform_data: List[Dict[str, Union[str, float]]] = []
        
        for i in range(len(self.df)):
            platforms_str = self.df.iloc[i]["platforms"]
            rating_val = self.df.iloc[i]["rating"]  
            game_name = self.df.iloc[i]["name"]
            
            if pd.notna(platforms_str) and pd.notna(rating_val) and isinstance(platforms_str, str):
                platforms = platforms_str.split(", ")
                # Safe float conversion with type checking
                try:
                    rating_float = float(rating_val) if rating_val not in (None, "", "N/A") else 0.0
                except (ValueError, TypeError):
                    rating_float = 0.0
                
                name_str = str(game_name) if pd.notna(game_name) else ""
                
                for platform in platforms:
                    platform_data.append({
                        "platform": platform,
                        "rating": rating_float,
                        "game_name": name_str
                    })
        
        platform_df = pd.DataFrame(platform_data)
        
        # Calculate platform stats
        platform_stats = platform_df.groupby("platform").agg({
            "rating": ["mean", "count"],
            "game_name": "nunique"
        }).round(2)
        
        platform_stats.columns = ["avg_rating", "total_entries", "unique_games"]
        platform_stats = platform_stats.reset_index()
        platform_stats = platform_stats[platform_stats["unique_games"] >= 5]  # Filter platforms with few games
        platform_stats = platform_stats.sort_values("avg_rating", ascending=True)
        
        # Create horizontal bar chart
        fig = px.bar(platform_stats, x="avg_rating", y="platform",
                    orientation="h",
                    title="Gaming Platform Quality Rankings",
                    labels={"avg_rating": "Average Game Rating", "platform": "Platform"},
                    color="avg_rating",
                    color_continuous_scale="RdYlGn")
        
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            showlegend=False
        )
        
        return fig, platform_stats
    
    def find_hidden_gems(self, min_rating: float = 4.2, max_rating_count: int = 500) -> Optional[Tuple[go.Figure, pd.DataFrame]]:
        """
        Discovers amazing games that might be flying under the radar.
        """
        if "rating" not in self.df.columns or "rating_count" not in self.df.columns:
            print("Need rating and rating_count data to find hidden gems")
            return None
        
        # Find highly rated games with relatively few ratings
        gems = self.df[
            (self.df["rating"] >= min_rating) & 
            (self.df["rating_count"] <= max_rating_count) &
            (self.df["rating_count"] >= 10)  # But not too few
        ].copy()
        
        if gems.empty:
            print(f"No hidden gems found with criteria (rating >= {min_rating}, rating_count <= {max_rating_count})")
            return None
        
        # Sort by rating and select top gems
        gems = gems.sort_values("rating", ascending=False)
        
        # Create bubble chart
        fig = px.scatter(gems.head(20), x="rating_count", y="rating",
                        size="avg_playtime", hover_name="name",
                        title="Hidden Gaming Gems - Great Games You Might Have Missed",
                        labels={"rating_count": "Number of Ratings", 
                               "rating": "Average Rating",
                               "avg_playtime": "Average Playtime (hours)"},
                        color="rating",
                        color_continuous_scale="viridis")
        
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white"
        )
        
        return fig, gems[["name", "rating", "rating_count", "genres", "release_year"]].head(10)
    
    def metacritic_vs_user_ratings(self) -> Optional[Tuple[go.Figure, float]]:
        """
        Do critics and players agree? Compares Metacritic scores vs user ratings.
        """
        if "metacritic_score" not in self.df.columns or "rating" not in self.df.columns:
            print("Need both Metacritic and user rating data")
            return None
        
        # Filter games with both scores
        metacritic_series = self.df["metacritic_score"].copy()
        rating_series = self.df["rating"].copy()
        
        comparison_df = self.df[
            (~metacritic_series.isna()) & 
            (~rating_series.isna())
        ].copy()
        
        if comparison_df.empty:
            print("No games with both Metacritic and user ratings found")
            return None
        
        # Convert user rating to 0-100 scale for comparison
        comparison_df["user_rating_scaled"] = comparison_df["rating"] * 20
        
        # Calculate correlation
        correlation = comparison_df["metacritic_score"].corr(comparison_df["user_rating_scaled"])
        
        # Create scatter plot
        fig = px.scatter(comparison_df, x="metacritic_score", y="user_rating_scaled",
                        hover_name="name", hover_data=["genres"],
                        title=f"Critics vs Players: Do They Agree? (Correlation: {correlation:.2f})",
                        labels={"metacritic_score": "Metacritic Score (0-100)",
                               "user_rating_scaled": "User Rating (0-100 scale)"},
                        color="release_year",
                        color_continuous_scale="viridis")
        
        # Add perfect agreement line
        fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], 
                                mode="lines", 
                                name="Perfect Agreement",
                                line=dict(dash="dash", color="red")))
        
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white"
        )
        
        return fig, correlation
    
    def generate_insights_report(self) -> List[str]:
        """
        Creates a comprehensive report with key insights about the gaming data.
        """
        insights: List[str] = []
        
        # Basic stats
        total_games = len(self.df)
        avg_rating = self.df["rating"].mean() if "rating" in self.df.columns else None
        
        insights.append(f"Dataset contains {total_games:,} games")
        
        if avg_rating:
            insights.append(f"Overall average rating: {avg_rating:.2f}/5.0")
        
        # Genre insights
        if "genres" in self.df.columns:
            all_genres = self.df["genres"].str.split(", ").explode()
            genre_counts = all_genres.value_counts()
            if len(genre_counts) > 0:
                # Safe string conversion from index
                top_genre_val = genre_counts.index[0]
                top_genre = str(top_genre_val) if pd.notna(top_genre_val) else "Unknown"
                insights.append(f"Most popular genre: {top_genre}")
        
        # Year insights
        if "release_year" in self.df.columns:
            valid_years = self.df[self.df["release_year"].between(1970, 2025)]
            if not valid_years.empty:
                year_counts = valid_years["release_year"].value_counts()
                if len(year_counts) > 0:
                    # Safe int conversion from index
                    peak_year_val = year_counts.index[0]
                    try:
                        peak_year = str(int(peak_year_val)) if pd.notna(peak_year_val) else "Unknown"
                    except (ValueError, TypeError):
                        peak_year = "Unknown"
                    insights.append(f"Most active release year: {peak_year}")
        
        # Rating insights
        if "rating" in self.df.columns:
            high_rated = len(self.df[self.df["rating"] >= 4.0])
            percentage = (high_rated / total_games) * 100
            insights.append(f"{percentage:.1f}% of games are highly rated (4.0+ stars)")
        
        return insights
    
    def create_dashboard_summary(self) -> Dict[str, Union[int, float, str]]:
        """
        Returns key metrics for a dashboard display.
        """
        summary: Dict[str, Union[int, float, str]] = {}
        
        summary["total_games"] = len(self.df)
        summary["avg_rating"] = round(float(self.df["rating"].mean()), 2) if "rating" in self.df.columns and not self.df["rating"].isna().all() else "N/A"
        summary["rating_std"] = round(float(self.df["rating"].std()), 2) if "rating" in self.df.columns and not self.df["rating"].isna().all() else "N/A"
        
        if "genres" in self.df.columns:
            all_genres = self.df["genres"].str.split(", ").explode()
            summary["total_genres"] = all_genres.nunique()
            genre_counts = all_genres.value_counts()
            if len(genre_counts) > 0:
                top_genre_val = genre_counts.index[0]
                summary["top_genre"] = str(top_genre_val) if pd.notna(top_genre_val) else "Unknown"
        
        if "release_year" in self.df.columns:
            valid_years = self.df[self.df["release_year"].between(1970, 2025)]
            if not valid_years.empty:
                try:
                    min_year_val = valid_years["release_year"].min()
                    max_year_val = valid_years["release_year"].max()
                    min_year = int(min_year_val) if pd.notna(min_year_val) else 1970
                    max_year = int(max_year_val) if pd.notna(max_year_val) else 2025
                    summary["year_range"] = f"{min_year}-{max_year}"
                except (ValueError, TypeError):
                    summary["year_range"] = "Unknown"
        
        return summary