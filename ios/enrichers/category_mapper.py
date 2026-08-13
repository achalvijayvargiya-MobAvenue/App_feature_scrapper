"""
ios/enrichers/category_mapper.py
=================================
Maps iTunes genres to our specific category flags.
Also implements keyword-based matching for descriptions and summaries.
"""

from typing import Any
import pandas as pd
from ios.constants import CATEGORY_COLS

# Map iTunes primaryGenreName strings to our category flags.
# Keyed by genre name (not numeric ID) since primaryGenreName is what the
# scraper now stores in the genreid column. Names below are Apple's exact
# genre labels as returned by the iTunes Lookup API.
GENRE_MAP: dict[str, list[str]] = {
    # Games (top-level + subgenres)
    "Games":               ["is_gaming_action_game"], # Base generic fallback, though usually games have subgenres
    "Action":              ["is_gaming_action_game"],
    "Adventure":           ["is_gaming_action_game"],
    "Arcade":              ["is_arcade_game"],
    "Board":                ["is_board_game"],
    "Card":                 ["is_real_money_card_and_casino_game"],
    "Casino":               ["is_real_money_card_and_casino_game"],
    "Educational":          ["is_education"],
    "Family":               ["is_casual_game"],
    "Puzzle":               ["is_trivia_and_puzzle_game"],
    "Racing":               ["is_racing_game"],
    "Role Playing":         ["is_simulation_and_role_playing_game"],
    "Simulation":           ["is_simulation_and_role_playing_game"],
    "Strategy":             ["is_strategy_game"],
    "Trivia":               ["is_trivia_and_puzzle_game"],
    "Word":                 ["is_word_game"],
    "Casual":               ["is_casual_game"],

    # Apps
    "Finance":              ["is_finance"],
    "Shopping":             ["is_shopping"],
    "Entertainment":        ["is_entertainment"],
    "Social Networking":    ["is_social"],
    "Education":            ["is_education"],
    "Utilities":            ["is_utility"],
    "Health & Fitness":     ["is_health"],
    "Travel":               ["is_travel"],
    "News":                 ["is_news"],
    "Food & Drink":         ["is_food_drink"],
    "Business":             ["is_utility"],
    "Weather":              ["is_utility"],
    "Sports":               ["is_sports_game"],
    "Reference":            ["is_utility"],
    "Productivity":         ["is_utility"],
    "Photo & Video":        ["is_utility"],
    "Navigation":           ["is_travel"],
    "Music":                ["is_entertainment"],
    "Pop":                  ["is_entertainment"], # Music sub-genre occasionally returned as primary
    "Lifestyle":            ["is_utility"],
    "Book":                 ["is_entertainment"],
    "Books":                ["is_entertainment"],
    "Medical":              ["is_health"],
    "Magazines & Newspapers": ["is_news"],
    "Developer Tools":      ["is_utility"],
    "Graphics & Design":    ["is_utility"],
}

def _wrap_single_words(kw: str) -> str:
    k = kw.strip()
    return k if " " in k else f" {k} "

# Using same keyword definitions as Android for consistency
SUMMARY_KEYWORD_MAP = {
    "is_investments": [_wrap_single_words(k) for k in ["invest", "investment", "stock market", "demat", "trading app", "mutual fund", "sip", "broker", "nifty", "sensex", "equity", "portfolio", "stocks", "shares", "intraday", "demat account"]],
    "is_crypto": [_wrap_single_words(k) for k in ["crypto", "cryptocurrency", "bitcoin", "btc", "ethereum", "blockchain", "defi", "nft", "altcoin", "wallet"]],
    "is_finance": [_wrap_single_words(k) for k in ["upi", "banking", "emi", "insurance", "netbanking", "neobank", "digital bank", "mobile banking", "payment", "wallet"]],
    "is_shopping": [_wrap_single_words(k) for k in ["shopping", "ecommerce", "e-commerce", "marketplace", "shop online", "online store", "instamart", "blinkit", "zepto"]],
    "is_entertainment": [_wrap_single_words(k) for k in ["movie", "movies", "series", "ott", "streaming", "podcast", "playlist", "lyrics", "cinema", "watch videos", "stream music", "short drama", "short dramas", "mini series", "mini drama", "romance drama", "family drama", "revenge drama", "fantasy drama", "emotional story", "emotional stories", "romantic story", "drama series", "drama episodes", "binge watch"]],
    "is_social": [_wrap_single_words(k) for k in ["chat", "messaging", "dating", "social network", "social media", "make friends", "meet people", "followers"]],
    "is_education": [_wrap_single_words(k) for k in ["education", "learning", "course", "exam", "mock test", "study", "tutorial", "academy", "syllabus", "revision"]],
    "is_utility": [_wrap_single_words(k) for k in ["file manager", "file explorer", "cleaner", "booster", "optimizer", "battery saver", "storage cleaner", "task manager", "utility", "productivity"]],
    "is_health": [_wrap_single_words(k) for k in ["fitness", "workout", "exercise", "gym", "yoga", "calorie", "step counter", "wellness", "meditation", "diet plan"]],
    "is_travel": [_wrap_single_words(k) for k in ["travel", "flight", "hotel booking", "navigation", "trip planner", "train ticket", "holiday", "vacation", "booking"]],
    "is_news": [_wrap_single_words(k) for k in ["news", "headline", "breaking news", "newspaper", "news app"]],
    "is_food_drink": [_wrap_single_words(k) for k in ["food", "restaurant", "dining", "recipe", "cuisine", "meal", "food delivery", "food ordering", "order food", "swiggy", "zomato"]],
    "is_lending": [_wrap_single_words(k) for k in ["personal loan", "instant loan", "quick loan", "payday loan", "cash advance", "borrow money", "lending", "microloan", "credit line", "fast loan", "easy loan"]],
    "is_quick_service": [_wrap_single_words(k) for k in ["quick delivery", "10 min delivery", "10 minute delivery", "instant delivery", "express delivery", "doorstep delivery", "blinkit", "zepto", "dunzo", "quick commerce"]],
    "is_ride": [_wrap_single_words(k) for k in ["cab", "taxi", "ride", "uber", "ola", "ride share", "ride sharing", "cab booking", "book cab", "auto", "bike taxi"]],
    "is_gaming_action_game": [_wrap_single_words(k) for k in ["action game", "shooter", "fps", "battle", "combat", "adventure game", "multiplayer shooter", "battle royale", "fighting game"]],
    "is_real_money_card_and_casino_game": [_wrap_single_words(k) for k in ["casino", "slots", "poker", "rummy", "teen patti", "card game", "real money", "real cash", "betting", "gambling"]],
    "is_word_game": [_wrap_single_words(k) for k in ["word game", "word puzzle", "scrabble", "crossword", "wordle", "spelling", "vocabulary", "word search"]],
    "is_trivia_and_puzzle_game": [_wrap_single_words(k) for k in ["trivia", "puzzle", "quiz", "brain teaser", "sudoku", "match 3", "puzzle game", "trivia game", "word puzzle"]],
    "is_strategy_game": [_wrap_single_words(k) for k in ["strategy", "tactical", "chess", "war game", "simulation", "strategy game", "tower defense", "empire building"]],
    "is_sports_game": [_wrap_single_words(k) for k in ["sports", "cricket", "football", "soccer", "basketball", "tennis", "sports game", "fifa", "cricket game", "sports management"]],
    "is_simulation_and_role_playing_game": [_wrap_single_words(k) for k in ["rpg", "role playing", "simulation", "simulator", "farm", "tycoon", "idle game", "character", "quest", "mmorpg"]],
    "is_racing_game": [_wrap_single_words(k) for k in ["racing", "race", "car game", "driving", "motorsport", "racing game", "kart", "asphalt"]],
    "is_casual_game": [_wrap_single_words(k) for k in ["casual game", "casual", "relaxing", "simple game", "endless", "tap", "idle", "hyper casual"]],
    "is_board_game": [_wrap_single_words(k) for k in ["board game", "ludo", "carrom", "chess", "checkers", "monopoly", "board", "tabletop"]],
    "is_arcade_game": [_wrap_single_words(k) for k in ["arcade", "retro", "classic game", "endless runner", "tap to play", "arcade game", "high score"]],
}

DESCRIPTION_PHRASE_MAP = {
    "is_investments": ["stock market", "share market", "mutual funds", "demat account", "stock trading", "equity trading", "nifty 50", "sensex", "sip investment", "invest in stocks", "trading platform", "investment portfolio", "stock broker", "intraday trading", "invest your money", "wealth management"],
    "is_crypto": ["cryptocurrency", "bitcoin wallet", "crypto wallet", "blockchain technology", "ethereum", "nft marketplace", "defi protocol", "crypto exchange", "crypto trading", "buy and sell crypto", "crypto investment"],
    "is_finance": ["bank account", "net banking", "mobile banking", "upi payment", "money transfer", "online payment", "digital payment", "bank transfer", "pay bills", "bill payment", "digital wallet", "recharge plans", "financial services"],
    "is_shopping": ["online shopping", "shop online", "buy online", "add to cart", "cash on delivery", "place an order", "order online", "product catalog", "shopping cart", "free delivery", "best deals", "exclusive offers"],
    "is_entertainment": ["watch movies", "stream videos", "watch tv shows", "watch series", "music streaming", "video streaming", "ott platform", "watch online", "movies and series", "live tv", "web series", "short drama series", "short drama app", "mini drama series", "watch short dramas", "romance drama series", "family drama series", "revenge drama", "fantasy drama series", "emotional drama", "romantic drama series", "drama for fast entertainment", "binge watch episodes", "swipe to watch", "vertical drama", "micro dramas"],
    "is_social": ["social networking app", "social network app", "dating app", "dating platform", "chat with friends", "chat with people", "video chat with friends", "real-time messaging", "messaging app", "instant messaging app", "connect with friends online", "make new friends online", "meet new people online", "follow friends", "send messages to friends"],
    "is_education": ["learn online", "online courses", "study material", "practice questions", "mock test", "exam preparation", "learning platform", "educational content", "video lectures", "study from home", "online classes", "test series"],
    "is_utility": ["file manager", "phone cleaner", "battery optimizer", "storage cleaner", "memory booster", "file explorer", "task manager", "speed booster", "junk cleaner", "ram cleaner", "cache cleaner"],
    "is_health": ["fitness tracker", "workout plan", "calorie counter", "step counter", "health tracker", "yoga poses", "meditation app", "mental health", "fitness goals", "diet plan", "weight loss", "bmi calculator"],
    "is_travel": ["book flights", "hotel booking", "flight booking", "travel booking", "travel planner", "train booking", "trip planner", "book hotel", "travel guide", "bus ticket", "holiday packages"],
    "is_news": ["breaking news", "news articles", "news headlines", "news channel", "news portal", "live news", "read the news", "news feed", "news app", "top headlines", "world news"],
    "is_food_drink": ["food delivery", "order food", "food ordering", "restaurant near you", "order from restaurants", "meal delivery", "online food", "home delivery", "food order", "order meals", "recipe app", "food and drink", "dining experience"],
    "is_lending": ["personal loan app", "instant loan approval", "loan disbursal", "apply for a loan", "get instant loan", "cash loan", "payday loan", "credit line", "microloan", "loan in minutes", "instant cash loan", "apply for personal loan", "get a personal loan"],
    "is_quick_service": ["grocery delivery app", "grocery delivery service", "order groceries online", "online grocery shopping", "quick delivery", "10 min delivery", "10 minute delivery", "at your doorstep", "doorstep delivery", "instant delivery", "express delivery", "quick commerce", "q-commerce"],
    "is_ride": ["cab booking", "book a cab", "ride sharing app", "book taxi", "taxi booking", "ride hailing", "book a ride", "cab service", "auto booking", "bike taxi", "ride with uber", "ola ride"],
    "is_gaming_action_game": ["action game", "shooter game", "fps game", "battle royale", "fighting game", "adventure game", "combat game", "multiplayer shooter", "first person shooter"],
    "is_real_money_card_and_casino_game": ["play rummy", "play poker", "teen patti", "real money games", "win real cash", "cash games", "online casino", "slot games", "card games for money", "betting app"],
    "is_word_game": ["word game", "word puzzle", "scrabble", "crossword puzzle", "word search", "spelling game", "vocabulary game", "word challenge", "guess the word"],
    "is_trivia_and_puzzle_game": ["trivia game", "puzzle game", "brain teaser", "quiz game", "match 3 game", "sudoku", "word puzzle", "trivia questions", "puzzle levels"],
    "is_strategy_game": ["strategy game", "tactical game", "chess game", "war strategy", "tower defense", "empire building", "resource management", "turn based strategy"],
    "is_sports_game": ["sports game", "cricket game", "football game", "sports management", "fifa", "cricket manager", "sports simulation", "athletic game"],
    "is_simulation_and_role_playing_game": ["role playing game", "rpg game", "simulation game", "farm simulator", "tycoon game", "idle game", "character customization", "quest game", "mmorpg"],
    "is_racing_game": ["racing game", "car racing", "driving game", "motorsport game", "kart racing", "endless racing"],
    "is_casual_game": ["casual game", "relaxing game", "simple gameplay", "tap to play", "endless runner", "hyper casual", "easy to play", "pick up and play"],
    "is_board_game": ["board game", "play ludo", "play carrom", "chess game", "board games", "tabletop game", "play monopoly"],
    "is_arcade_game": ["arcade game", "retro game", "classic arcade", "endless runner", "high score", "arcade style"],
}

_GENRE_MAP_LOWER: dict[str, list[str]] = {k.lower(): v for k, v in GENRE_MAP.items()}

def phase1_genre(genre_name: Any) -> set[str]:
    if not isinstance(genre_name, str) or not genre_name.strip():
        return set()
    return set(_GENRE_MAP_LOWER.get(genre_name.strip().lower(), []))

def _pad_text(txt: str) -> str:
    return " " + str(txt).lower().strip() + " "

def phase2a_summary(summary: Any) -> set[str]:
    if not pd.notna(summary) or not str(summary).strip():
        return set()
    txt = _pad_text(summary)
    return {cat for cat, kws in SUMMARY_KEYWORD_MAP.items() if any(kw in txt for kw in kws)}

def phase2b_description(description: Any) -> set[str]:
    if not pd.notna(description) or not str(description).strip():
        return set()
    txt = _pad_text(description)
    return {cat for cat, phrases in DESCRIPTION_PHRASE_MAP.items() if any(p in txt for p in phrases)}

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input  : DataFrame with [bundle_id, genreid (= primaryGenreName), summary, description]
    Output : DataFrame with [bundle_id] + binary category flag columns
    """
    def _map_row(genre_name: Any, summary: Any, description: Any) -> dict[str, int]:
        combined = phase1_genre(genre_name) | phase2a_summary(summary) | phase2b_description(description)
        return {col: int(col in combined) for col in CATEGORY_COLS}

    flags = df.apply(
        lambda r: _map_row(r.get("genreid"), r.get("summary"), r.get("description")),
        axis=1,
        result_type="expand",
    )
    for col in CATEGORY_COLS:
        if col not in flags.columns:
            flags[col] = 0

    return pd.concat([df[["bundle_id"]].reset_index(drop=True), flags], axis=1)
