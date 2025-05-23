import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import logging

# Import our custom modules
try:
    from game_explorer import (
        fetch_popular_games, search_games_by_title, discover_games_by_genre,
        get_game_recommendations, browse_by_platform, smart_search,
        get_friendly_genre_list, translate_user_genre
    )
    from data_processor import (
        transform_raw_games, enrich_with_categories, 
        filter_games_smartly, create_genre_breakdown
    )
    from game_analytics import GameAnalytics
except ImportError as e:
    st.error(f"❌ Import Error: {e}")
    st.info("🔧 Make sure all project files are in the same directory and dependencies are installed!")
    st.stop()

# Configure page settings
st.set_page_config(
    page_title="NextPlay - Gaming Analytics Dashboard",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for gaming theme
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 100%);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #16213e 0%, #0f4c75 100%);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #00d4ff;
        margin: 0.5rem 0;
    }
    
    .game-card {
        background: rgba(22, 33, 62, 0.8);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #00d4ff;
        margin: 0.5rem 0;
    }
    
    .stSelectbox > div > div {
        background-color: #16213e;
        color: white;
    }
    
    .stTextInput > div > div > input {
        background-color: #16213e;
        color: white;
    }
    
    h1, h2, h3 {
        color: #00d4ff;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.3);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
        color: white;
        border: none;
        border-radius: 5px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Cache data to avoid repeated API calls
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_game_data(data_type="popular", search_term="", count=50):
    """
    Loads and caches game data from the API
    """
    try:
        if data_type == "popular":
            raw_games = fetch_popular_games(count=count)
        elif data_type == "search":
            raw_games = search_games_by_title(search_term, limit=count)
        elif data_type == "genre":
            raw_games = discover_games_by_genre(search_term, count=count)
        else:
            raw_games = fetch_popular_games(count=count)
        
        if raw_games:
            df = transform_raw_games(raw_games)
            df = enrich_with_categories(df)
            return df
        else:
            return pd.DataFrame()
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def display_game_card(game_info):
    """
    Creates a styled card for displaying game information
    """
    with st.container():
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if game_info.get("background_image"):
                st.image(game_info["background_image"], width=150)
        
        with col2:
            st.markdown(f"""
            <div class="game-card">
                <h3>🎮 {game_info.get('name', 'Unknown Game')}</h3>
                <p><strong>Rating:</strong> ⭐ {game_info.get('rating', 'N/A')}/5.0</p>
                <p><strong>Released:</strong> {game_info.get('release_date', 'Unknown')}</p>
                <p><strong>Genres:</strong> {game_info.get('genres', 'N/A')}</p>
                <p><strong>Platforms:</strong> {game_info.get('platforms', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)

def main_dashboard():
    """
    Main dashboard interface
    """
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1>🎮 NextPlay Gaming Analytics</h1>
        <p style="font-size: 1.2em; color: #00d4ff;">Discover, Analyze, and Explore the Gaming Universe</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for navigation and controls
    with st.sidebar:
        st.markdown("## 🕹️ Control Panel")
        
        page = st.selectbox(
            "Choose Your Adventure:",
            ["🏠 Dashboard", "🔍 Game Discovery", "📊 Analytics Deep Dive", "🎯 Recommendations"]
        )
        
        st.markdown("---")
        
        # Data loading controls
        st.markdown("### 📥 Data Source")
        data_source = st.radio(
            "What games do you want to explore?",
            ["Popular Games", "Search by Title", "Browse by Genre"]
        )
        
        search_params = {}
        if data_source == "Search by Title":
            search_params["search_term"] = st.text_input("🔍 Enter game title:")
        elif data_source == "Browse by Genre":
            genres = get_friendly_genre_list()
            search_params["genre"] = st.selectbox("🎲 Select genre:", [""] + genres)
        
        data_count = st.slider("📈 Number of games to analyze:", 20, 100, 50)
        
        if st.button("🚀 Load Data"):
            st.session_state["reload_data"] = True
    
    # Load data based on user selection
    if data_source == "Popular Games":
        df = load_game_data("popular", count=data_count)
    elif data_source == "Search by Title" and search_params.get("search_term"):
        df = load_game_data("search", search_params["search_term"], data_count)
    elif data_source == "Browse by Genre" and search_params.get("genre"):
        df = load_game_data("genre", search_params["genre"], data_count)
    else:
        df = load_game_data("popular", count=data_count)
    
    if df.empty:
        st.warning("No games found with current criteria. Try different search terms!")
        return
    
    # Display content based on selected page
    if page == "🏠 Dashboard":
        show_dashboard_overview(df)
    elif page == "🔍 Game Discovery":
        show_game_discovery(df)
    elif page == "📊 Analytics Deep Dive":
        show_analytics_deep_dive(df)
    elif page == "🎯 Recommendations":
        show_recommendations_engine(df)

def show_dashboard_overview(df):
    """
    Main dashboard overview with key metrics and charts
    """
    st.markdown("## 📊 Gaming Landscape Overview")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎮 Total Games</h3>
            <h2>{len(df):,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_rating = df["rating"].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3>⭐ Average Rating</h3>
            <h2>{avg_rating:.2f}/5.0</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if "release_year" in df.columns:
            year_range = f"{df['release_year'].min():.0f}-{df['release_year'].max():.0f}"
        else:
            year_range = "N/A"
        st.markdown(f"""
        <div class="metric-card">
            <h3>📅 Year Range</h3>
            <h2>{year_range}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        if "genres" in df.columns:
            total_genres = len(set([g.strip() for genres in df["genres"].dropna() for g in genres.split(",")]))
        else:
            total_genres = "N/A"
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎲 Unique Genres</h3>
            <h2>{total_genres}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts row
    col1, col2 = st.columns(2)
    
    # Initialize analytics
    analytics = GameAnalytics(df)
    
    with col1:
        st.markdown("### 🌟 Rating Distribution")
        rating_fig = analytics.rating_landscape(interactive=True)
        if rating_fig:
            st.plotly_chart(rating_fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Genre Popularity")
        genre_fig = analytics.genre_popularity_contest(top_n=10, interactive=True)
        if genre_fig:
            st.plotly_chart(genre_fig, use_container_width=True)
    
    # Gaming evolution timeline
    st.markdown("### 🚀 Gaming Evolution Over Time")
    timeline_fig = analytics.gaming_evolution_timeline()
    if timeline_fig:
        st.plotly_chart(timeline_fig, use_container_width=True)

def show_game_discovery(df):
    """
    Game discovery and search interface
    """
    st.markdown("## 🔍 Game Discovery Hub")
    
    # Search and filter controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        rating_filter = st.slider("⭐ Minimum Rating", 0.0, 5.0, 3.0, 0.1)
    
    with col2:
        if "release_year" in df.columns:
            year_range = st.slider(
                "📅 Release Year Range",
                int(df["release_year"].min()),
                int(df["release_year"].max()),
                (int(df["release_year"].min()), int(df["release_year"].max()))
            )
        else:
            year_range = None
    
    with col3:
        if "genres" in df.columns:
            all_genres = set([g.strip() for genres in df["genres"].dropna() for g in genres.split(",")])
            selected_genre = st.selectbox("🎲 Filter by Genre", ["Any"] + list(sorted(all_genres)))
        else:
            selected_genre = "Any"
    
    # Apply filters
    filtered_df = df.copy()
    if rating_filter > 0:
        filtered_df = filtered_df[filtered_df["rating"] >= rating_filter]
    
    if year_range and "release_year" in df.columns:
        filtered_df = filtered_df[
            (filtered_df["release_year"] >= year_range[0]) & 
            (filtered_df["release_year"] <= year_range[1])
        ]
    
    if selected_genre != "Any" and "genres" in df.columns:
        filtered_df = filtered_df[filtered_df["genres"].str.contains(selected_genre, na=False)]
    
    st.markdown(f"### 🎮 Found {len(filtered_df)} Games Matching Your Criteria")
    
    # Display games
    for idx, game in filtered_df.head(20).iterrows():
        display_game_card(game)

def show_analytics_deep_dive(df):
    """
    Advanced analytics and insights
    """
    st.markdown("## 📊 Analytics Deep Dive")
    
    analytics = GameAnalytics(df)
    
    # Analysis selection
    analysis_type = st.selectbox(
        "Choose Analysis Type:",
        ["Genre Quality Analysis", "Platform Ecosystem", "Hidden Gems", "Critics vs Players"]
    )
    
    if analysis_type == "Genre Quality Analysis":
        st.markdown("### 🎯 Genre Quality vs Quantity Analysis")
        fig, stats = analytics.genre_quality_analysis()
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("#### 📈 Detailed Genre Statistics")
            st.dataframe(stats)
    
    elif analysis_type == "Platform Ecosystem":
        st.markdown("### 🎮 Gaming Platform Analysis")
        fig, stats = analytics.platform_ecosystem_analysis()
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("#### 📊 Platform Rankings")
            st.dataframe(stats)
    
    elif analysis_type == "Hidden Gems":
        st.markdown("### 💎 Hidden Gaming Gems")
        result = analytics.find_hidden_gems(min_rating=4.0, max_rating_count=1000)
        if result:
            fig, gems_df = result
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("#### 🏆 Top Hidden Gems")
            st.dataframe(gems_df)
    
    elif analysis_type == "Critics vs Players":
        st.markdown("### 🎭 Critics vs Players: The Eternal Debate")
        result = analytics.metacritic_vs_user_ratings()
        if result:
            fig, correlation = result
            st.plotly_chart(fig, use_container_width=True)
            st.info(f"Correlation coefficient: {correlation:.3f} - {'Strong agreement' if abs(correlation) > 0.7 else 'Moderate agreement' if abs(correlation) > 0.4 else 'Weak agreement'}")

def show_recommendations_engine(df):
    """
    Game recommendation system
    """
    st.markdown("## 🎯 Game Recommendation Engine")
    
    st.markdown("### 🔍 Find Games Similar to Your Favorites")
    
    # Game selection for recommendations
    game_names = df["name"].tolist() if "name" in df.columns else []
    
    if game_names:
        selected_game = st.selectbox("🎮 Choose a game you love:", [""] + game_names)
        
        if selected_game:
            # Simple similarity based on genres and rating
            base_game = df[df["name"] == selected_game].iloc[0]
            
            # Find similar games
            if "genres" in df.columns and pd.notna(base_game["genres"]):
                base_genres = set(base_game["genres"].split(", "))
                
                similarities = []
                for idx, game in df.iterrows():
                    if game["name"] == selected_game:
                        continue
                    
                    if pd.notna(game["genres"]):
                        game_genres = set(game["genres"].split(", "))
                        genre_overlap = len(base_genres.intersection(game_genres)) / len(base_genres.union(game_genres))
                        
                        rating_similarity = 1 - abs(base_game["rating"] - game["rating"]) / 5
                        
                        overall_similarity = (genre_overlap * 0.7) + (rating_similarity * 0.3)
                        similarities.append((idx, overall_similarity))
                
                # Sort by similarity and get top recommendations
                similarities.sort(key=lambda x: x[1], reverse=True)
                top_recommendations = similarities[:10]
                
                st.markdown(f"### 🎮 Games Similar to '{selected_game}'")
                
                for idx, similarity in top_recommendations:
                    recommended_game = df.iloc[idx]
                    similarity_percentage = similarity * 100
                    
                    st.markdown(f"""
                    <div class="game-card">
                        <h4>🎯 {recommended_game['name']} ({similarity_percentage:.0f}% match)</h4>
                        <p><strong>Rating:</strong> ⭐ {recommended_game['rating']}/5.0</p>
                        <p><strong>Genres:</strong> {recommended_game['genres']}</p>
                        <p><strong>Released:</strong> {recommended_game.get('release_date', 'Unknown')}</p>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("Load some game data first to get recommendations!")

if __name__ == "__main__":
    main_dashboard()