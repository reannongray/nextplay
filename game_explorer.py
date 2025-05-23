import requests
import time
import logging

# API Configuration - keep your key secure in production!
API_KEY = "c02a210d2724482eafa08f53853ce928"
RAWG_BASE = "https://api.rawg.io/api"
REQUEST_HEADERS = {
    "User-Agent": "NextPlayApp/1.0 - Gaming Analytics Dashboard"
}

# Smart genre mapping - what humans search for vs what the API expects
GENRE_TRANSLATIONS = {
    # Popular genres people actually search for
    "action": "action",
    "adventure": "adventure", 
    "rpg": "role-playing-games-rpg",
    "role playing": "role-playing-games-rpg",
    "strategy": "strategy",
    "racing": "racing",
    "sports": "sports",
    "simulation": "simulation",
    "sim": "simulation",
    "puzzle": "puzzle",
    "shooter": "shooter",
    "fps": "shooter",
    "fighting": "fighting",
    "platformer": "platformer",
    "arcade": "arcade",
    "indie": "indie",
    "casual": "casual",
    "family": "family",
    "horror": "action",  # Often tagged under action
    "survival": "action",
    "mmo": "massively-multiplayer",
    "multiplayer": "massively-multiplayer",
    "board games": "board-games",
    "card": "card"
}

# Cache to avoid hammering the API
_genre_cache = None
_platform_cache = None

def get_friendly_genre_list():
    """
    Returns a list of genre names that regular humans would actually search for.
    No need to memorize API slugs - just think like a gamer!
    """
    return list(GENRE_TRANSLATIONS.keys())

def translate_user_genre(user_input):
    """
    Takes what a user typed (like 'rpg' or 'role playing') and finds the right API slug.
    Pretty forgiving - handles different ways people might search.
    """
    clean_input = user_input.lower().strip()
    return GENRE_TRANSLATIONS.get(clean_input, clean_input)

def fetch_popular_games(count=20, year_filter=None):
    """
    Grabs the most popular games right now. Perfect for discovering what's trending.
    """
    endpoint = f"{RAWG_BASE}/games"
    params = {
        "key": API_KEY,
        "page_size": min(count, 40),  # API max is 40
        "ordering": "-rating"  # Start with highest rated
    }
    
    if year_filter:
        params["dates"] = f"{year_filter}-01-01,{year_filter}-12-31"
    
    try:
        response = requests.get(endpoint, headers=REQUEST_HEADERS, params=params)
        response.raise_for_status()
        
        games = response.json().get("results", [])
        logging.info(f"Found {len(games)} popular games")
        return games
        
    except requests.RequestException as e:
        logging.error(f"Couldn't fetch popular games: {e}")
        return []

def search_games_by_title(query, limit=10):
    """
    Search for games by name - handles typos pretty well thanks to RAWG's search.
    """
    endpoint = f"{RAWG_BASE}/games"
    params = {
        "key": API_KEY,
        "search": query,
        "page_size": min(limit, 40),
        "ordering": "-rating"  # Best matches first
    }
    
    try:
        response = requests.get(endpoint, headers=REQUEST_HEADERS, params=params)
        response.raise_for_status()
        
        games = response.json().get("results", [])
        logging.info(f"Search for '{query}' returned {len(games)} games")
        return games
        
    except requests.RequestException as e:
        logging.error(f"Search failed for '{query}': {e}")
        return []

def discover_games_by_genre(genre_name, count=15, min_rating=3.0):
    """
    Find awesome games in a specific genre. Takes human-friendly genre names!
    """
    api_genre = translate_user_genre(genre_name)
    endpoint = f"{RAWG_BASE}/games"
    
    params = {
        "key": API_KEY,
        "genres": api_genre,
        "page_size": min(count, 40),
        "ordering": "-rating",
        "rating": f"{min_rating},5"  # Only show well-rated games
    }
    
    try:
        response = requests.get(endpoint, headers=REQUEST_HEADERS, params=params)
        response.raise_for_status()
        
        games = response.json().get("results", [])
        logging.info(f"Found {len(games)} great {genre_name} games")
        return games
        
    except requests.RequestException as e:
        logging.error(f"Genre search failed for '{genre_name}': {e}")
        return []

def get_game_recommendations(game_title, similarity_count=5):
    """
    Find games similar to one you already love. Uses RAWG's smart recommendation engine.
    """
    # First, find the exact game
    search_results = search_games_by_title(game_title, limit=1)
    if not search_results:
        logging.warning(f"Couldn't find '{game_title}' for recommendations")
        return []
    
    base_game = search_results[0]
    game_id = base_game.get("id")
    
    endpoint = f"{RAWG_BASE}/games/{game_id}/suggested"
    params = {
        "key": API_KEY,
        "page_size": similarity_count
    }
    
    try:
        response = requests.get(endpoint, headers=REQUEST_HEADERS, params=params)
        response.raise_for_status()
        
        suggestions = response.json().get("results", [])
        logging.info(f"Found {len(suggestions)} games similar to '{game_title}'")
        return suggestions
        
    except requests.RequestException as e:
        logging.error(f"Recommendations failed for '{game_title}': {e}")
        return []

def get_detailed_game_info(game_id_or_slug):
    """
    Get everything we know about a specific game - reviews, screenshots, the works.
    """
    endpoint = f"{RAWG_BASE}/games/{game_id_or_slug}"
    params = {"key": API_KEY}
    
    try:
        response = requests.get(endpoint, headers=REQUEST_HEADERS, params=params)
        response.raise_for_status()
        
        game_details = response.json()
        logging.info(f"Retrieved details for game: {game_details.get('name', 'Unknown')}")
        return game_details
        
    except requests.RequestException as e:
        logging.error(f"Failed to get details for game {game_id_or_slug}: {e}")
        return None

def browse_by_platform(platform_name, count=20):
    """
    Discover great games on a specific platform (PC, PlayStation, Xbox, etc.)
    """
    platform_mapping = {
        "pc": "4",
        "playstation": "18,19,20,21,187",  # All PlayStation platforms
        "xbox": "1,14,80,186",  # All Xbox platforms  
        "nintendo": "7,8,9,13,83",  # Nintendo platforms
        "mobile": "3,21",  # iOS and Android
        "switch": "7"
    }
    
    platform_id = platform_mapping.get(platform_name.lower())
    if not platform_id:
        logging.warning(f"Unknown platform: {platform_name}")
        return []
    
    endpoint = f"{RAWG_BASE}/games"
    params = {
        "key": API_KEY,
        "platforms": platform_id,
        "page_size": min(count, 40),
        "ordering": "-rating"
    }
    
    try:
        response = requests.get(endpoint, headers=REQUEST_HEADERS, params=params)
        response.raise_for_status()
        
        games = response.json().get("results", [])
        logging.info(f"Found {len(games)} great {platform_name} games")
        return games
        
    except requests.RequestException as e:
        logging.error(f"Platform search failed for '{platform_name}': {e}")
        return []

def smart_search(query, search_type="auto"):
    """
    One function to rule them all - figures out what the user wants and finds it.
    """
    query = query.lower().strip()
    
    # Is it a genre?
    if query in GENRE_TRANSLATIONS or search_type == "genre":
        return discover_games_by_genre(query)
    
    # Is it a platform?
    platforms = ["pc", "playstation", "xbox", "nintendo", "mobile", "switch"]
    if any(platform in query for platform in platforms) or search_type == "platform":
        for platform in platforms:
            if platform in query:
                return browse_by_platform(platform)
    
    # Default to title search
    return search_games_by_title(query)

# Utility function for consistent API throttling
def gentle_api_pause(seconds=0.5):
    """
    Be nice to RAWG's servers - they're giving us free data after all!
    """
    time.sleep(seconds)