# Real-time gaming data pipeline for NextPlay dashboard
# Integrates multiple APIs: RAWG, Steam, Reddit, Price tracking

import asyncio
import aiohttp
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import time
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GameTrend:
    """Real-time game trend data structure"""
    game_id: str
    title: str
    trend_score: float
    reddit_mentions: int
    steam_players: int
    price_change: float
    sentiment_score: float
    last_updated: datetime

class LiveGamingAPI:
    """
    Multi-source gaming API integration for real-time dashboard data.
    
    Gamer Features:
    - Live player counts and trends
    - Reddit community buzz tracking  
    - Price drop alerts and tracking
    - Real-time review sentiment
    - "What's hot right now" detection
    
    Technical Features:
    - Async API calls for performance
    - Smart caching with TTL
    - Rate limit handling
    - Fallback mechanisms
    - Error recovery
    """
    
    def __init__(self):
        self.rawg_key = os.getenv("RAWG_API_KEY")  # Read from .env file
        self.steam_key = os.getenv("STEAM_API_KEY")  # Optional - add to .env if you get one
        self.session = None
        self.cache = {}
        self.last_api_calls = {}
        
        # Validate RAWG API key
        if not self.rawg_key:
            logging.warning("RAWG_API_KEY not found in environment variables. Live features will use demo data.")
        
        # Rate limiting (calls per minute)
        self.rate_limits = {
            'rawg': 60,      # RAWG: 20,000 per month
            'steam': 100,    # Steam: Generous limits
            'reddit': 60     # Reddit: 60 per minute
        }
    
    async def __aenter__(self):
        """Async context manager setup"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup async session"""
        if self.session:
            await self.session.close()
    
    def _check_rate_limit(self, api_name: str) -> bool:
        """Smart rate limiting to avoid API blocks"""
        now = time.time()
        if api_name not in self.last_api_calls:
            self.last_api_calls[api_name] = []
        
        # Remove calls older than 1 minute
        self.last_api_calls[api_name] = [
            call_time for call_time in self.last_api_calls[api_name]
            if now - call_time < 60
        ]
        
        # Check if we can make another call
        if len(self.last_api_calls[api_name]) < self.rate_limits[api_name]:
            self.last_api_calls[api_name].append(now)
            return True
        return False
    
    def _get_cache_key(self, endpoint: str, params: Dict) -> str:
        """Generate cache key for API responses"""
        return f"{endpoint}_{hash(str(sorted(params.items())))}"
    
    def _is_cache_valid(self, cache_key: str, ttl_minutes: int = 15) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.cache:
            return False
        
        cached_time = self.cache[cache_key].get('timestamp')
        if not cached_time:
            return False
        
        return datetime.now() - cached_time < timedelta(minutes=ttl_minutes)
    
    async def get_trending_games(self, limit: int = 20) -> List[Dict]:
        """
        Get currently trending games across multiple platforms.
        GAMER VALUE: "What's hot right now?"
        """
        cache_key = f"trending_games_{limit}"
        
        if self._is_cache_valid(cache_key, ttl_minutes=10):
            logger.info("📱 Returning cached trending games")
            return self.cache[cache_key]['data']
        
        try:
            trending_games = []
            
            # Get popular games from RAWG (if API key available)
            if self.rawg_key and self._check_rate_limit('rawg'):
                rawg_url = "https://api.rawg.io/api/games"
                params = {
                    'key': self.rawg_key,
                    'ordering': '-added',  # Most recently added/popular
                    'page_size': limit,
                    'dates': f"{datetime.now().year}-01-01,{datetime.now().year}-12-31"
                }
                
                async with self.session.get(rawg_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for game in data.get('results', []):
                            trending_games.append({
                                'id': game['id'],
                                'name': game['name'],
                                'rating': game.get('rating', 0),
                                'released': game.get('released'),
                                'platforms': [p['platform']['name'] for p in game.get('platforms', [])],
                                'genres': [g['name'] for g in game.get('genres', [])],
                                'background_image': game.get('background_image'),
                                'metacritic': game.get('metacritic'),
                                'playtime': game.get('playtime', 0),
                                'source': 'rawg'
                            })
                        
                        logger.info(f"🔥 Fetched {len(trending_games)} games from RAWG API")
                    else:
                        logger.warning(f"RAWG API returned status {response.status}")
            
            # Fallback to demo data if no API key or API fails
            if not trending_games:
                logger.info("🎮 Using demo trending games data")
                trending_games = self._generate_demo_trending_games(limit)
            
            # Cache the results
            self.cache[cache_key] = {
                'data': trending_games,
                'timestamp': datetime.now()
            }
            
            logger.info(f"🔥 Found {len(trending_games)} trending games")
            return trending_games
            
        except Exception as e:
            logger.error(f"❌ Error fetching trending games: {e}")
            # Return cached data if available, even if expired
            if cache_key in self.cache:
                return self.cache[cache_key]['data']
            # Last resort: demo data
            return self._generate_demo_trending_games(limit)
    
    def _generate_demo_trending_games(self, limit: int) -> List[Dict]:
        """Generate demo trending games for when API is unavailable"""
        demo_games = [
            {"id": 1, "name": "Cyberpunk 2077", "rating": 4.1, "released": "2020-12-10", 
             "platforms": ["PC", "PlayStation", "Xbox"], "genres": ["Action", "RPG"], 
             "background_image": None, "metacritic": 86, "playtime": 60, "source": "demo"},
            {"id": 2, "name": "The Witcher 3", "rating": 4.7, "released": "2015-05-19",
             "platforms": ["PC", "PlayStation", "Xbox"], "genres": ["RPG", "Adventure"],
             "background_image": None, "metacritic": 95, "playtime": 120, "source": "demo"},
            {"id": 3, "name": "Elden Ring", "rating": 4.5, "released": "2022-02-25",
             "platforms": ["PC", "PlayStation", "Xbox"], "genres": ["Action", "RPG"],
             "background_image": None, "metacritic": 96, "playtime": 80, "source": "demo"},
            {"id": 4, "name": "Grand Theft Auto V", "rating": 4.3, "released": "2013-09-17",
             "platforms": ["PC", "PlayStation", "Xbox"], "genres": ["Action", "Crime"],
             "background_image": None, "metacritic": 97, "playtime": 40, "source": "demo"},
            {"id": 5, "name": "Red Dead Redemption 2", "rating": 4.6, "released": "2018-10-26",
             "platforms": ["PC", "PlayStation", "Xbox"], "genres": ["Action", "Western"],
             "background_image": None, "metacritic": 97, "playtime": 60, "source": "demo"},
        ]
        return demo_games[:limit]
    
    async def get_reddit_buzz(self, game_name: str) -> Dict:
        """
        Get Reddit community buzz for a specific game.
        GAMER VALUE: "What are people saying about this game?"
        """
        cache_key = f"reddit_buzz_{game_name.lower().replace(' ', '_')}"
        
        if self._is_cache_valid(cache_key, ttl_minutes=30):
            return self.cache[cache_key]['data']
        
        try:
            if self._check_rate_limit('reddit'):
                # Reddit search API (no auth required for basic search)
                reddit_url = "https://www.reddit.com/search.json"
                params = {
                    'q': f'"{game_name}" (subreddit:gaming OR subreddit:Games OR subreddit:pcgaming)',
                    'sort': 'hot',
                    'limit': 25,
                    't': 'week'  # Past week
                }
                
                headers = {'User-Agent': 'NextPlay/1.0'}
                
                async with self.session.get(reddit_url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        posts = data.get('data', {}).get('children', [])
                        
                        # Analyze sentiment and engagement
                        total_score = sum(post['data'].get('score', 0) for post in posts)
                        total_comments = sum(post['data'].get('num_comments', 0) for post in posts)
                        
                        # Simple sentiment analysis based on scores
                        positive_posts = len([p for p in posts if p['data'].get('score', 0) > 10])
                        sentiment_score = positive_posts / len(posts) if posts else 0.5
                        
                        buzz_data = {
                            'mentions': len(posts),
                            'total_upvotes': total_score,
                            'total_comments': total_comments,
                            'sentiment_score': sentiment_score,
                            'trending_posts': [
                                {
                                    'title': post['data']['title'],
                                    'score': post['data']['score'],
                                    'comments': post['data']['num_comments'],
                                    'url': f"https://reddit.com{post['data']['permalink']}"
                                }
                                for post in posts[:5]  # Top 5 posts
                            ]
                        }
                        
                        self.cache[cache_key] = {
                            'data': buzz_data,
                            'timestamp': datetime.now()
                        }
                        
                        return buzz_data
            
        except Exception as e:
            logger.error(f"❌ Error fetching Reddit buzz for {game_name}: {e}")
        
        # Return default data if API fails
        return {
            'mentions': 0,
            'total_upvotes': 0,
            'total_comments': 0,
            'sentiment_score': 0.5,
            'trending_posts': []
        }
    
    async def get_steam_player_stats(self, game_name: str) -> Dict:
        """
        Get Steam concurrent player statistics.
        GAMER VALUE: "How active is this game's community?"
        """
        cache_key = f"steam_stats_{game_name.lower().replace(' ', '_')}"
        
        if self._is_cache_valid(cache_key, ttl_minutes=5):  # Shorter cache for live data
            return self.cache[cache_key]['data']
        
        try:
            if self.steam_key and self._check_rate_limit('steam'):
                # First try to get Steam app ID from Steam API
                app_id = await self._get_steam_appid(game_name)
                
                if app_id:
                    # Get current player count from Steam API
                    steam_url = "https://api.steampowered.com/ISteamUserStats/GetNumberOfCurrentPlayers/v1/"
                    params = {
                        'key': self.steam_key,
                        'appid': app_id
                    }
                    
                    async with self.session.get(steam_url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            current_players = data.get('response', {}).get('player_count', 0)
                            
                            # Also try SteamSpy for additional stats
                            steamspy_data = await self._get_steamspy_data(app_id)
                            
                            player_stats = {
                                'current_players': current_players,
                                'peak_players': steamspy_data.get('peak', 0),
                                'owners': steamspy_data.get('owners', '0'),
                                'positive_reviews': steamspy_data.get('positive', 0),
                                'negative_reviews': steamspy_data.get('negative', 0),
                                'average_playtime': steamspy_data.get('average_forever', 0)
                            }
                            
                            self.cache[cache_key] = {
                                'data': player_stats,
                                'timestamp': datetime.now()
                            }
                            
                            logger.info(f"🎮 Steam stats for {game_name}: {current_players} current players")
                            return player_stats
            
            # Fallback to demo data
            return self._generate_demo_steam_stats()
            
        except Exception as e:
            logger.error(f"❌ Error fetching Steam stats for {game_name}: {e}")
            return self._generate_demo_steam_stats()
    
    async def _get_steamspy_data(self, app_id: str) -> Dict:
        """Get additional stats from SteamSpy API"""
        try:
            steamspy_url = "https://steamspy.com/api.php"
            params = {
                'request': 'appdetails',
                'appid': app_id
            }
            
            async with self.session.get(steamspy_url, params=params) as response:
                if response.status == 200:
                    return await response.json()
        except Exception as e:
            logger.warning(f"SteamSpy API error: {e}")
        
        return {}
    
    def _generate_demo_steam_stats(self) -> Dict:
        """Generate demo Steam stats when API is unavailable"""
        return {
            'current_players': np.random.randint(1000, 50000),
            'peak_players': np.random.randint(10000, 100000),
            'owners': f"{np.random.randint(100000, 10000000):,}",
            'positive_reviews': np.random.randint(5000, 50000),
            'negative_reviews': np.random.randint(500, 5000),
            'average_playtime': np.random.randint(10, 100)
        }
    
    async def _get_steam_appid(self, game_name: str) -> Optional[str]:
        """Helper to find Steam App ID for a game using Steam API"""
        try:
            if not self.steam_key:
                return None
            
            # Use Steam's GetAppList API to search for the game
            steam_url = "https://api.steampowered.com/ISteamApps/GetAppList/v2/"
            
            async with self.session.get(steam_url) as response:
                if response.status == 200:
                    data = await response.json()
                    apps = data.get('applist', {}).get('apps', [])
                    
                    # Simple name matching (could be improved with fuzzy matching)
                    game_name_lower = game_name.lower()
                    for app in apps:
                        app_name_lower = app['name'].lower()
                        if game_name_lower in app_name_lower or app_name_lower in game_name_lower:
                            logger.info(f"🎮 Found Steam App ID {app['appid']} for {game_name}")
                            return str(app['appid'])
            
        except Exception as e:
            logger.warning(f"Error finding Steam App ID for {game_name}: {e}")
        
        # Fallback: return None for demo data
        return None
    
    async def get_price_tracking_data(self, game_name: str) -> Dict:
        """
        Track game prices across platforms for deal alerts.
        GAMER VALUE: "Should I buy now or wait for a sale?"
        """
        cache_key = f"price_tracking_{game_name.lower().replace(' ', '_')}"
        
        if self._is_cache_valid(cache_key, ttl_minutes=60):  # Prices don't change that often
            return self.cache[cache_key]['data']
        
        # This would integrate with price tracking APIs like IsThereAnyDeal
        # For demo, return mock data
        mock_price_data = {
            'current_price': np.random.uniform(10, 60),
            'lowest_price': np.random.uniform(5, 30),
            'price_change_24h': np.random.uniform(-5, 5),
            'discount_percentage': np.random.uniform(0, 75),
            'platforms': {
                'Steam': np.random.uniform(15, 60),
                'Epic': np.random.uniform(15, 60),
                'GOG': np.random.uniform(15, 60)
            },
            'price_history': [
                {'date': (datetime.now() - timedelta(days=i)).isoformat(), 
                 'price': np.random.uniform(20, 60)}
                for i in range(30, 0, -1)
            ]
        }
        
        self.cache[cache_key] = {
            'data': mock_price_data,
            'timestamp': datetime.now()
        }
        
        return mock_price_data

class GamingTrendAnalyzer:
    """
    Analyzes gaming trends and generates insights for dashboard.
    
    GAMER FEATURES:
    - "Games rising in popularity"
    - "What genre is trending?"
    - "Best deals right now"
    - "Community favorites"
    """
    
    def __init__(self, api_client: LiveGamingAPI):
        self.api = api_client
    
    async def analyze_trending_genres(self) -> Dict:
        """Analyze which genres are currently trending"""
        trending_games = await self.api.get_trending_games(50)
        
        genre_counts = {}
        genre_ratings = {}
        
        for game in trending_games:
            for genre in game.get('genres', []):
                if genre not in genre_counts:
                    genre_counts[genre] = 0
                    genre_ratings[genre] = []
                
                genre_counts[genre] += 1
                genre_ratings[genre].append(game.get('rating', 0))
        
        # Calculate trend scores
        trend_data = {}
        for genre in genre_counts:
            avg_rating = np.mean(genre_ratings[genre]) if genre_ratings[genre] else 0
            trend_score = genre_counts[genre] * avg_rating  # Popularity × Quality
            
            trend_data[genre] = {
                'count': genre_counts[genre],
                'avg_rating': avg_rating,
                'trend_score': trend_score
            }
        
        return dict(sorted(trend_data.items(), key=lambda x: x[1]['trend_score'], reverse=True))
    
    async def get_hot_games_now(self, limit: int = 10) -> List[Dict]:
        """Get the hottest games right now based on multiple factors"""
        trending_games = await self.api.get_trending_games(30)
        
        # Score each game based on multiple factors
        scored_games = []
        
        for game in trending_games:
            # Get additional data
            reddit_buzz = await self.api.get_reddit_buzz(game['name'])
            steam_stats = await self.api.get_steam_player_stats(game['name'])
            
            # Calculate composite hotness score
            rating_score = game.get('rating', 0) * 10
            reddit_score = min(reddit_buzz['mentions'] * reddit_buzz['sentiment_score'], 100)
            player_score = min(steam_stats['current_players'] / 1000, 100)  # Scale down
            recency_score = 50 if game.get('released') and game['released'] >= '2023-01-01' else 25
            
            hotness_score = rating_score + reddit_score + player_score + recency_score
            
            scored_games.append({
                **game,
                'hotness_score': hotness_score,
                'reddit_buzz': reddit_buzz,
                'steam_stats': steam_stats
            })
        
        # Sort by hotness score and return top games
        return sorted(scored_games, key=lambda x: x['hotness_score'], reverse=True)[:limit]

# Streamlit Integration Functions
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_live_gaming_data():
    """Load live gaming data for Streamlit dashboard"""
    async def fetch_data():
        async with LiveGamingAPI() as api:
            analyzer = GamingTrendAnalyzer(api)
            
            return {
                'trending_games': await api.get_trending_games(20),
                'hot_games': await analyzer.get_hot_games_now(10),
                'trending_genres': await analyzer.analyze_trending_genres()
            }
    
    # Run async function in event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        return loop.run_until_complete(fetch_data())
    finally:
        loop.close()

def display_live_trends_sidebar():
    """Add live trends to Streamlit sidebar"""
    st.sidebar.markdown("## 🔥 Live Gaming Trends")
    
    try:
        live_data = load_live_gaming_data()
        
        # Hot Games Section
        st.sidebar.markdown("### 🎮 Hot Right Now")
        for game in live_data['hot_games'][:5]:
            score = game.get('hotness_score', 0)
            st.sidebar.markdown(f"**{game['name']}** ({score:.0f}🔥)")
        
        # Trending Genres
        st.sidebar.markdown("### 📈 Trending Genres")
        for genre, data in list(live_data['trending_genres'].items())[:5]:
            st.sidebar.markdown(f"**{genre}** ({data['count']} games)")
            
    except Exception as e:
        st.sidebar.error("⚡ Live data temporarily unavailable")
        logger.error(f"Dashboard integration error: {e}")

if __name__ == "__main__":
    # Test the pipeline
    async def test_pipeline():
        async with LiveGamingAPI() as api:
            analyzer = GamingTrendAnalyzer(api)
            
            print("🎮 Testing Live Gaming API Pipeline")
            print("=" * 50)
            
            # Test trending games
            trending = await api.get_trending_games(5)
            print(f"📈 Found {len(trending)} trending games")
            
            # Test Reddit buzz
            if trending:
                buzz = await api.get_reddit_buzz(trending[0]['name'])
                print(f"💬 Reddit mentions: {buzz['mentions']}")
            
            # Test hot games analysis
            hot_games = await analyzer.get_hot_games_now(3)
            print(f"🔥 Top 3 hot games:")
            for i, game in enumerate(hot_games, 1):
                print(f"  {i}. {game['name']} (Score: {game['hotness_score']:.1f})")
            
            print("\n✅ Pipeline test complete!")
    
    # Run test
    asyncio.run(test_pipeline())