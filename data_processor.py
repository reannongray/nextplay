import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
import re

def transform_raw_games(raw_game_list: List[Dict[str, Any]], focus_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Takes the messy JSON from RAWG API and turns it into a clean DataFrame.
    Handles all the nested stuff and missing data gracefully.
    """
    if not raw_game_list:
        logging.warning("No games to process - got an empty list")
        return pd.DataFrame()
    
    # Extract and flatten all the useful stuff from each game
    processed_games: List[Dict[str, Any]] = []
    
    for game in raw_game_list:
        # Basic info that's always useful
        game_record: Dict[str, Any] = {
            "id": game.get("id"),
            "name": game.get("name", "Unknown Title"),
            "release_date": game.get("released"),
            "rating": game.get("rating", 0),
            "rating_count": game.get("ratings_count", 0),
            "metacritic_score": game.get("metacritic"),
            "avg_playtime": game.get("playtime", 0),
            "background_image": game.get("background_image")
        }
        
        # Handle the tricky nested lists - genres, platforms, etc.
        game_record["genres"] = extract_names_from_list(game.get("genres", []))
        game_record["platforms"] = extract_platform_names(game.get("platforms", []))
        game_record["stores"] = extract_store_names(game.get("stores", []))
        game_record["tags"] = extract_names_from_list(game.get("tags", [])[:6])  # Top 6 tags only
        
        # Calculate some useful derived fields
        game_record["release_year"] = extract_year_from_date(game.get("released"))
        game_record["popularity_score"] = calculate_popularity(game_record["rating"], game_record["rating_count"])
        game_record["has_metacritic"] = game_record["metacritic_score"] is not None
        
        processed_games.append(game_record)
    
    # Convert to DataFrame and clean it up
    df = pd.DataFrame(processed_games)
    
    # Filter columns if requested
    if focus_columns:
        available_cols: List[str] = [col for col in focus_columns if col in df.columns]
        df = df[available_cols]
    
    # Clean up the data
    df = cleanup_dataframe(df)
    
    logging.info(f"Processed {len(df)} games successfully")
    return df

def extract_names_from_list(items_list: List[Dict[str, Any]]) -> str:
    """
    Pulls the 'name' field out of a list of dictionaries.
    Works for genres, tags, developers, etc.
    """
    if not items_list or not isinstance(items_list, list):
        return ""
    
    names: List[str] = [item.get("name", "") for item in items_list if isinstance(item, dict)]
    return ", ".join(filter(None, names))

def extract_platform_names(platforms_list: List[Dict[str, Any]]) -> str:
    """
    Handles the weird nested platform structure that RAWG uses.
    """
    if not platforms_list or not isinstance(platforms_list, list):
        return ""
    
    platform_names: List[str] = []
    for platform_info in platforms_list:
        if isinstance(platform_info, dict) and "platform" in platform_info:
            platform_name: str = platform_info["platform"].get("name", "")
            if platform_name:
                platform_names.append(platform_name)
    
    return ", ".join(platform_names)

def extract_store_names(stores_list: List[Dict[str, Any]]) -> str:
    """
    Gets the store names (Steam, Epic, etc.) where you can buy the game.
    """
    if not stores_list or not isinstance(stores_list, list):
        return ""
    
    store_names: List[str] = []
    for store_info in stores_list:
        if isinstance(store_info, dict) and "store" in store_info:
            store_name: str = store_info["store"].get("name", "")
            if store_name:
                store_names.append(store_name)
    
    return ", ".join(store_names)

def extract_year_from_date(date_string: Optional[str]) -> Optional[int]:
    """
    Pulls the year out of a date string. Handles various formats gracefully.
    """
    if not date_string:
        return None
    
    try:
        # Handle YYYY-MM-DD format
        if isinstance(date_string, str) and len(date_string) >= 4:
            year_match = re.match(r"(\d{4})", date_string)
            if year_match:
                return int(year_match.group(1))
        return None
    except (ValueError, TypeError):
        return None

def calculate_popularity(rating: Union[int, float], rating_count: Union[int, float]) -> float:
    """
    Creates a popularity score that balances rating quality with rating quantity.
    A game with 4.5 stars and 1000 reviews beats one with 4.6 stars and 10 reviews.
    """
    if not rating or not rating_count:
        return 0.0
    
    # Weighted score - rating matters more but volume provides confidence
    confidence_factor = min(rating_count / 100, 10)  # Cap the volume boost
    return round(rating * (1 + confidence_factor * 0.1), 2)

def cleanup_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final cleanup pass - remove duplicates, handle missing values, optimize types.
    """
    # Remove exact duplicates
    initial_count: int = len(df)
    df.drop_duplicates(subset=["id"], inplace=True)
    
    if len(df) < initial_count:
        logging.info(f"Removed {initial_count - len(df)} duplicate games")
    
    # Drop games missing critical info
    df.dropna(subset=["name"], inplace=True)
    
    # Fill in reasonable defaults for missing ratings
    df["rating"].fillna(0, inplace=True)
    df["rating_count"].fillna(0, inplace=True)
    df["avg_playtime"].fillna(0, inplace=True)
    
    # Convert numeric columns to appropriate types
    numeric_cols: List[str] = ["rating", "rating_count", "metacritic_score", "avg_playtime", "release_year", "popularity_score"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Sort by popularity score for better user experience
    if "popularity_score" in df.columns:
        df.sort_values("popularity_score", ascending=False, inplace=True)
    
    df.reset_index(drop=True, inplace=True)
    return df

def apply_game_age_category(release_year_series: pd.Series, current_year: int) -> pd.Series:
    """
    Helper function to apply game age categorization to a pandas Series.
    """
    def safe_categorize(year: Union[int, float, None]) -> str:
        if pd.isna(year):
            return "Unknown"
        return categorize_game_age(year, current_year)
    
    return release_year_series.apply(safe_categorize).astype('string')

def enrich_with_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds some useful categorical columns for better analysis.
    """
    if df.empty:
        return df
    
    # Age categories based on release year
    current_year: int = datetime.now().year
    df["game_age"] = apply_game_age_category(df["release_year"], current_year)
    
    # Rating categories
    df["rating_tier"] = df["rating"].apply(categorize_rating)
    
    # Playtime categories  
    df["time_commitment"] = df["avg_playtime"].apply(categorize_playtime)
    
    # Platform count (how many platforms it's on)
    df["platform_count"] = df["platforms"].apply(
        lambda x: len(x.split(", ")) if isinstance(x, str) and x else 0
    )
    
    return df

def categorize_game_age(release_year: Optional[Union[int, float]], current_year: int) -> str:
    """
    Groups games into useful age categories.
    """
    if pd.isna(release_year):
        return "Unknown"
    
    age: int = current_year - int(release_year)
    if age <= 1:
        return "Brand New"
    elif age <= 3:
        return "Recent"
    elif age <= 7:
        return "Modern"
    elif age <= 15:
        return "Classic"
    else:
        return "Retro"

def categorize_rating(rating: Union[int, float]) -> str:
    """
    Converts numeric ratings into descriptive tiers.
    """
    if pd.isna(rating) or rating == 0:
        return "Unrated"
    elif rating >= 4.5:
        return "Exceptional"
    elif rating >= 4.0:
        return "Excellent"
    elif rating >= 3.5:
        return "Good"
    elif rating >= 3.0:
        return "Mixed"
    else:
        return "Poor"

def categorize_playtime(hours: Union[int, float]) -> str:
    """
    Groups games by expected time commitment.
    """
    if pd.isna(hours) or hours == 0:
        return "Unknown"
    elif hours <= 5:
        return "Quick Play"
    elif hours <= 15:
        return "Short"
    elif hours <= 40:
        return "Medium"
    elif hours <= 100:
        return "Long"
    else:
        return "Epic"

def save_processed_data(df: pd.DataFrame, filename: str = "processed_games.csv", include_timestamp: bool = True) -> bool:
    """
    Saves the cleaned data to CSV with optional timestamp.
    """
    if df.empty:
        logging.warning("No data to save - DataFrame is empty")
        return False
    
    try:
        if include_timestamp:
            timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename.replace('.csv', '')}_{timestamp}.csv"
        
        df.to_csv(filename, index=False)
        logging.info(f"Successfully saved {len(df)} games to {filename}")
        return True
        
    except Exception as e:
        logging.error(f"Failed to save data: {e}")
        return False

def create_genre_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes genre distribution for insights.
    """
    if df.empty or "genres" not in df.columns:
        return pd.DataFrame()
    
    # Split genres and count them
    all_genres = df["genres"].str.split(", ").explode()
    genre_counts = all_genres.value_counts()
    
    # Calculate some stats
    genre_stats = pd.DataFrame({
        "genre": genre_counts.index,
        "game_count": genre_counts.values,
        "percentage": (genre_counts.values.astype(float) / len(df) * 100).round(1)
    })
    
    return genre_stats

def filter_games_smartly(df: pd.DataFrame, **criteria: Any) -> pd.DataFrame:
    """
    Flexible filtering that handles various criteria naturally.
    """
    filtered_df = df.copy()
    
    # Filter by genre (partial match)
    if "genre" in criteria and isinstance(criteria["genre"], str):
        genre_filter = criteria["genre"].lower()
        filtered_df = filtered_df[filtered_df["genres"].str.lower().str.contains(genre_filter, na=False)]
    
    # Filter by rating range
    if "min_rating" in criteria and isinstance(criteria["min_rating"], (int, float)):
        filtered_df = filtered_df[filtered_df["rating"] >= criteria["min_rating"]]
    
    if "max_rating" in criteria and isinstance(criteria["max_rating"], (int, float)):
        filtered_df = filtered_df[filtered_df["rating"] <= criteria["max_rating"]]
    
    # Filter by year range
    if "min_year" in criteria and isinstance(criteria["min_year"], (int, float)):
        filtered_df = filtered_df[filtered_df["release_year"] >= criteria["min_year"]]
    
    if "max_year" in criteria and isinstance(criteria["max_year"], (int, float)):
        filtered_df = filtered_df[filtered_df["release_year"] <= criteria["max_year"]]
    
    # Filter by platform
    if "platform" in criteria and isinstance(criteria["platform"], str):
        platform_filter = criteria["platform"].lower()
        filtered_df = filtered_df[filtered_df["platforms"].str.lower().str.contains(platform_filter, na=False)]
    
    return filtered_df