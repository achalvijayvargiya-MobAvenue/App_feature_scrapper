"""
enrichers/category_mapper.py
============================
Three-phase Play Store app category mapper.

Phase 1  – genreId deterministic lookup.
    Fast, high-precision mapping from the Play Store genre taxonomy.
    One genre can resolve to multiple target categories.
    Genres with no clear mapping (e.g. LIFESTYLE) return an empty set.

Phase 2a – summary keyword scan  (single keywords, safe on short text).
    The `summary` field is a 1–2 sentence developer blurb — intentional,
    dense, and low in false positives. Single-word keywords work well here.

Phase 2b – description phrase scan  (multi-word phrases only).
    The `description` is a long marketing paragraph. Generic single words
    like "market", "share", "connect", "store", "order" appear in nearly
    every app description regardless of category, causing false positives.
    Only multi-word, domain-specific phrases are used here.

Final result = union(Phase 1, Phase 2a, Phase 2b).

Public API
----------
    enrich(df)                         → DataFrame with bundle_id + category flags
    phase1_genre(genre_id)             → set[str]
    phase2a_summary(summary)           → set[str]
    phase2b_description(description)   → set[str]
    phase_breakdown(df)                → diagnostic log (p1/p2a/p2b/none counts)
"""

import logging
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Target categories – order matches output_catagory.txt / test.txt schema
# ---------------------------------------------------------------------------
CATEGORY_COLS: list[str] = [
    "is_investments", "is_crypto", "is_finance", "is_shopping",
    "is_entertainment", "is_social", "is_education", "is_utility",
    "is_health", "is_travel", "is_news", "is_food_drink",
    "is_lending", "is_quick_service", "is_ride",
    "is_gaming_action_game", "is_real_money_card_and_casino_game", "is_word_game",
    "is_trivia_and_puzzle_game", "is_strategy_game", "is_sports_game",
    "is_simulation_and_role_playing_game", "is_racing_game", "is_casual_game",
    "is_board_game", "is_arcade_game",
]

# ---------------------------------------------------------------------------
# Phase 1 – genreId → target category list
#   • One genre can map to multiple categories
#   • Empty list = no deterministic mapping; phases 2a/2b are the resolver
# ---------------------------------------------------------------------------
GENRE_MAP: dict[str, list[str]] = {
    # ── Gaming (specific subcategories) ─────────────────────────────────────
    "GAME_ACTION":         ["is_gaming_action_game"],
    "GAME_ADVENTURE":      ["is_gaming_action_game"],
    "GAME_ARCADE":         ["is_arcade_game"],
    "GAME_BOARD":          ["is_board_game"],
    "GAME_CARD":           ["is_real_money_card_and_casino_game"],
    "GAME_CASINO":         ["is_real_money_card_and_casino_game"],
    "GAME_CASUAL":         ["is_casual_game"],
    "GAME_EDUCATIONAL":    ["is_education", "is_casual_game"],
    "GAME_MUSIC":          ["is_casual_game"],
    "GAME_PUZZLE":         ["is_trivia_and_puzzle_game"],
    "GAME_RACING":         ["is_racing_game"],
    "GAME_ROLE_PLAYING":   ["is_simulation_and_role_playing_game"],
    "GAME_SIMULATION":     ["is_simulation_and_role_playing_game"],
    "GAME_SPORTS":         ["is_sports_game"],
    "GAME_STRATEGY":       ["is_strategy_game"],
    "GAME_TRIVIA":         ["is_trivia_and_puzzle_game"],
    "GAME_WORD":           ["is_word_game"],
    # ── Finance ──────────────────────────────────────────────────────────────
    "FINANCE":             ["is_finance"],
    # ── Shopping ─────────────────────────────────────────────────────────────
    "SHOPPING":            ["is_shopping"],
    # ── Entertainment ────────────────────────────────────────────────────────
    "ENTERTAINMENT":       ["is_entertainment"],
    "VIDEO_PLAYERS":       ["is_entertainment"],
    "MUSIC_AND_AUDIO":     ["is_entertainment"],
    "COMICS":              ["is_entertainment"],
    "SPORTS":              ["is_sports_game", "is_entertainment"],
    "EVENTS":              ["is_entertainment"],
    # ── Social ───────────────────────────────────────────────────────────────
    "SOCIAL":              ["is_social"],
    "DATING":              ["is_social"],
    "COMMUNICATION":       ["is_social"],
    # ── Education ────────────────────────────────────────────────────────────
    "EDUCATION":           ["is_education"],
    "BOOKS_AND_REFERENCE": ["is_education"],
    "PARENTING":           ["is_education", "is_health"],
    # ── Health ───────────────────────────────────────────────────────────────
    "HEALTH_AND_FITNESS":  ["is_health"],
    "MEDICAL":             ["is_health"],
    "BEAUTY":              ["is_health"],
    # ── Travel / Ride ────────────────────────────────────────────────────────
    "TRAVEL_AND_LOCAL":    ["is_travel", "is_ride"],
    "MAPS_AND_NAVIGATION": ["is_travel", "is_ride"],
    "AUTO_AND_VEHICLES":   ["is_utility", "is_ride"],
    # ── News ─────────────────────────────────────────────────────────────────
    "NEWS_AND_MAGAZINES":  ["is_news"],
    # ── Food / Quick Service ─────────────────────────────────────────────────
    "FOOD_AND_DRINK":      ["is_food_drink"],
    # ── Utility / Tools ──────────────────────────────────────────────────────
    "TOOLS":               ["is_utility"],
    "PRODUCTIVITY":        ["is_utility"],
    "LIBRARIES_AND_DEMO":  ["is_utility"],
    "PERSONALIZATION":     ["is_utility"],
    "WEATHER":             ["is_utility"],
    "PHOTOGRAPHY":         ["is_utility"],
    "ART_AND_DESIGN":      ["is_utility"],
    "HOUSE_AND_HOME":      ["is_quick_service"],
    "BUSINESS":            ["is_utility"],
    "LIFESTYLE":           ["is_utility"],
}

# ---------------------------------------------------------------------------
# Phase 2a – SUMMARY keyword map  (single keywords, safe on short text)
#
# The summary is short (1–2 sentences) and written to describe the app's
# primary purpose. Generic words like "bank", "game", "shop" are intentional
# here and reliably indicate category.
# ---------------------------------------------------------------------------
def _wrap_single_words(kw: str) -> str:
    """Add spaces around single words to match whole words only."""
    k = kw.strip()
    if " " in k:
        return k  # multi-word phrase, keep as-is
    return f" {k} "


SUMMARY_KEYWORD_MAP: dict[str, list[str]] = {
    "is_investments": [
        _wrap_single_words(k) for k in [
        "invest", "investment", "stock market", "demat", "trading app",
        "mutual fund", "sip", "broker", "nifty", "sensex", "equity",
        "portfolio", "stocks", "shares", "intraday", "demat account",
    ]],
    "is_crypto": [
        _wrap_single_words(k) for k in [
        "crypto", "cryptocurrency", "bitcoin", "btc", "ethereum", "blockchain",
        "defi", "nft", "altcoin", "wallet",
    ]],
    "is_finance": [
        _wrap_single_words(k) for k in [
        "upi", "banking", "emi", "insurance", "netbanking", "neobank",
        "digital bank", "mobile banking", "payment", "wallet",
    ]],
    "is_shopping": [
        _wrap_single_words(k) for k in [
        "shopping", "ecommerce", "e-commerce", "marketplace",
        "shop online", "online store", "instamart", "blinkit", "zepto",
    ]],
    "is_entertainment": [
        _wrap_single_words(k) for k in [
        "movie", "movies", "series", "ott", "streaming", "podcast",
        "playlist", "lyrics", "cinema", "watch videos", "stream music",
        "short drama", "short dramas", "mini series", "mini drama",
        "romance drama", "family drama", "revenge drama", "fantasy drama",
        "emotional story", "emotional stories", "romantic story",
        "drama series", "drama episodes", "binge watch",
    ]],
    "is_social": [
        _wrap_single_words(k) for k in [
        "chat", "messaging", "dating", "social network", "social media",
        "make friends", "meet people", "followers",
    ]],
    "is_education": [
        _wrap_single_words(k) for k in [
        "education", "learning", "course", "exam", "mock test", "study",
        "tutorial", "academy", "syllabus", "revision",
    ]],
    "is_utility": [
        _wrap_single_words(k) for k in [
        "file manager", "file explorer", "cleaner", "booster", "optimizer",
        "battery saver", "storage cleaner", "task manager", "utility",
        "productivity",
    ]],
    "is_health": [
        _wrap_single_words(k) for k in [
        "fitness", "workout", "exercise", "gym", "yoga", "calorie",
        "step counter", "wellness", "meditation", "diet plan",
    ]],
    "is_travel": [
        _wrap_single_words(k) for k in [
        "travel", "flight", "hotel booking", "navigation", "trip planner",
        "train ticket", "holiday", "vacation", "booking",
    ]],
    "is_news": [
        _wrap_single_words(k) for k in [
        "news", "headline", "breaking news", "newspaper", "news app",
    ]],
    "is_food_drink": [
        _wrap_single_words(k) for k in [
        "food", "restaurant", "dining", "recipe", "cuisine", "meal",
        "food delivery", "food ordering", "order food", "swiggy", "zomato",
    ]],
    "is_lending": [
        _wrap_single_words(k) for k in [
        "personal loan", "instant loan", "quick loan", "payday loan",
        "cash advance", "borrow money", "lending", "microloan",
        "credit line", "fast loan", "easy loan",
    ]],
    "is_quick_service": [
        _wrap_single_words(k) for k in [
        "quick delivery", "10 min delivery", "10 minute delivery",
        "instant delivery", "express delivery", "doorstep delivery",
        "blinkit", "zepto", "dunzo", "quick commerce",
    ]],
    "is_ride": [
        _wrap_single_words(k) for k in [
        "cab", "taxi", "ride", "uber", "ola", "ride share", "ride sharing",
        "cab booking", "book cab", "auto", "bike taxi",
    ]],
    "is_gaming_action_game": [
        _wrap_single_words(k) for k in [
        "action game", "shooter", "fps", "battle", "combat", "adventure game",
        "multiplayer shooter", "battle royale", "fighting game",
    ]],
    "is_real_money_card_and_casino_game": [
        _wrap_single_words(k) for k in [
        "casino", "slots", "poker", "rummy", "teen patti", "card game",
        "real money", "real cash", "betting", "gambling",
    ]],
    "is_word_game": [
        _wrap_single_words(k) for k in [
        "word game", "word puzzle", "scrabble", "crossword", "wordle",
        "spelling", "vocabulary", "word search",
    ]],
    "is_trivia_and_puzzle_game": [
        _wrap_single_words(k) for k in [
        "trivia", "puzzle", "quiz", "brain teaser", "sudoku", "match 3",
        "puzzle game", "trivia game", "word puzzle",
    ]],
    "is_strategy_game": [
        _wrap_single_words(k) for k in [
        "strategy", "tactical", "chess", "war game", "simulation",
        "strategy game", "tower defense", "empire building",
    ]],
    "is_sports_game": [
        _wrap_single_words(k) for k in [
        "sports", "cricket", "football", "soccer", "basketball", "tennis",
        "sports game", "fifa", "cricket game", "sports management",
    ]],
    "is_simulation_and_role_playing_game": [
        _wrap_single_words(k) for k in [
        "rpg", "role playing", "simulation", "simulator", "farm",
        "tycoon", "idle game", "character", "quest", "mmorpg",
    ]],
    "is_racing_game": [
        _wrap_single_words(k) for k in [
        "racing", "race", "car game", "driving", "motorsport",
        "racing game", "kart", "asphalt",
    ]],
    "is_casual_game": [
        _wrap_single_words(k) for k in [
        "casual game", "casual", "relaxing", "simple game", "endless",
        "tap", "idle", "hyper casual",
    ]],
    "is_board_game": [
        _wrap_single_words(k) for k in [
        "board game", "ludo", "carrom", "chess", "checkers", "monopoly",
        "board", "tabletop",
    ]],
    "is_arcade_game": [
        _wrap_single_words(k) for k in [
        "arcade", "retro", "classic game", "endless runner", "tap to play",
        "arcade game", "high score",
    ]],
}

# ---------------------------------------------------------------------------
# Phase 2b – DESCRIPTION phrase map  (multi-word phrases only)
#
# The description is a long marketing paragraph — it contains generic words
# that are NOT reliable signals of category. Only specific multi-word phrases
# are used to prevent false positives.
#
# Rule of thumb: if the phrase could appear in a description for an UNRELATED
# app, it should NOT be in this list.
# ---------------------------------------------------------------------------
DESCRIPTION_PHRASE_MAP: dict[str, list[str]] = {
    "is_investments": [
        "stock market", "share market", "mutual funds",
        "demat account", "stock trading", "equity trading",
        "nifty 50", "sensex", "sip investment",
        "invest in stocks", "trading platform",
        "investment portfolio", "stock broker", "intraday trading",
        "invest your money", "wealth management",
    ],
    "is_crypto": [
        "cryptocurrency", "bitcoin wallet", "crypto wallet",
        "blockchain technology", "ethereum", "nft marketplace",
        "defi protocol", "crypto exchange", "crypto trading",
        "buy and sell crypto", "crypto investment",
    ],
    "is_finance": [
        "bank account", "net banking", "mobile banking", "upi payment",
        "money transfer", "online payment", "digital payment",
        "bank transfer", "pay bills", "bill payment", "digital wallet",
        "recharge plans", "financial services",
    ],
    "is_shopping": [
        "online shopping", "shop online", "buy online", "add to cart",
        "cash on delivery", "place an order", "order online",
        "product catalog", "shopping cart", "free delivery",
        "best deals", "exclusive offers",
    ],
    "is_entertainment": [
        "watch movies", "stream videos", "watch tv shows", "watch series",
        "music streaming", "video streaming", "ott platform",
        "watch online", "movies and series", "live tv", "web series",
        "short drama series", "short drama app", "mini drama series",
        "watch short dramas", "romance drama series", "family drama series",
        "revenge drama", "fantasy drama series", "emotional drama",
        "romantic drama series", "drama for fast entertainment",
        "binge watch episodes", "swipe to watch", "vertical drama","micro dramas",
    ],
    "is_social": [
        "social networking app", "social network app",
        "dating app", "dating platform",
        "chat with friends", "chat with people",
        "video chat with friends", "real-time messaging",
        "messaging app", "instant messaging app",
        "connect with friends online", "make new friends online",
        "meet new people online", "follow friends",
        "send messages to friends",
    ],
    "is_education": [
        "learn online", "online courses", "study material",
        "practice questions", "mock test", "exam preparation",
        "learning platform", "educational content", "video lectures",
        "study from home", "online classes", "test series",
    ],
    "is_utility": [
        "file manager", "phone cleaner", "battery optimizer",
        "storage cleaner", "memory booster", "file explorer",
        "task manager", "speed booster", "junk cleaner",
        "ram cleaner", "cache cleaner",
    ],
    "is_health": [
        "fitness tracker", "workout plan", "calorie counter",
        "step counter", "health tracker", "yoga poses",
        "meditation app", "mental health", "fitness goals",
        "diet plan", "weight loss", "bmi calculator",
    ],
    "is_travel": [
        "book flights", "hotel booking", "flight booking",
        "travel booking", "travel planner",
        "train booking", "trip planner", "book hotel",
        "travel guide", "bus ticket", "holiday packages",
    ],
    "is_news": [
        "breaking news", "news articles", "news headlines",
        "news channel", "news portal", "live news",
        "read the news", "news feed", "news app",
        "top headlines", "world news",
    ],
    "is_food_drink": [
        "food delivery", "order food", "food ordering",
        "restaurant near you", "order from restaurants",
        "meal delivery", "online food", "home delivery",
        "food order", "order meals", "recipe app",
        "food and drink", "dining experience",
    ],
    "is_lending": [
        "personal loan app", "instant loan approval", "loan disbursal",
        "apply for a loan", "get instant loan", "cash loan",
        "payday loan", "credit line", "microloan",
        "loan in minutes", "instant cash loan",
        "apply for personal loan", "get a personal loan",
    ],
    "is_quick_service": [
        "grocery delivery app", "grocery delivery service",
        "order groceries online", "online grocery shopping",
        "quick delivery", "10 min delivery", "10 minute delivery",
        "at your doorstep", "doorstep delivery",
        "instant delivery", "express delivery",
        "quick commerce", "q-commerce",
    ],
    "is_ride": [
        "cab booking", "book a cab", "ride sharing app",
        "book taxi", "taxi booking", "ride hailing",
        "book a ride", "cab service", "auto booking",
        "bike taxi", "ride with uber", "ola ride",
    ],
    "is_gaming_action_game": [
        "action game", "shooter game", "fps game", "battle royale",
        "fighting game", "adventure game", "combat game",
        "multiplayer shooter", "first person shooter",
    ],
    "is_real_money_card_and_casino_game": [
        "play rummy", "play poker", "teen patti", "real money games",
        "win real cash", "cash games", "online casino",
        "slot games", "card games for money", "betting app",
    ],
    "is_word_game": [
        "word game", "word puzzle", "scrabble", "crossword puzzle",
        "word search", "spelling game", "vocabulary game",
        "word challenge", "guess the word",
    ],
    "is_trivia_and_puzzle_game": [
        "trivia game", "puzzle game", "brain teaser", "quiz game",
        "match 3 game", "sudoku", "word puzzle",
        "trivia questions", "puzzle levels",
    ],
    "is_strategy_game": [
        "strategy game", "tactical game", "chess game",
        "war strategy", "tower defense", "empire building",
        "resource management", "turn based strategy",
    ],
    "is_sports_game": [
        "sports game", "cricket game", "football game",
        "sports management", "fifa", "cricket manager",
        "sports simulation", "athletic game",
    ],
    "is_simulation_and_role_playing_game": [
        "role playing game", "rpg game", "simulation game",
        "farm simulator", "tycoon game", "idle game",
        "character customization", "quest game", "mmorpg",
    ],
    "is_racing_game": [
        "racing game", "car racing", "driving game",
        "motorsport game", "kart racing", "endless racing",
    ],
    "is_casual_game": [
        "casual game", "relaxing game", "simple gameplay",
        "tap to play", "endless runner", "hyper casual",
        "easy to play", "pick up and play",
    ],
    "is_board_game": [
        "board game", "play ludo", "play carrom", "chess game",
        "board games", "tabletop game", "play monopoly",
    ],
    "is_arcade_game": [
        "arcade game", "retro game", "classic arcade",
        "endless runner", "high score", "arcade style",
    ],
}


# ---------------------------------------------------------------------------
# Phase functions  (independently callable for testing / debugging)
# ---------------------------------------------------------------------------

def phase1_genre(genre_id: Any) -> set[str]:
    """Phase 1: genreId → category set. Empty set when unknown/unmapped."""
    if not isinstance(genre_id, str) or not genre_id.strip():
        return set()
    return set(GENRE_MAP.get(genre_id.strip().upper(), []))


def _pad_text(txt: str) -> str:
    """Pad text with spaces so keywords at start/end match (e.g. ' invest ' in ' invest today ')."""
    return " " + str(txt).lower().strip() + " "


def phase2a_summary(summary: Any) -> set[str]:
    """
    Phase 2a: single-keyword scan on summary text only.
    Single words are space-wrapped to match whole words only.
    Text is padded so keywords at start/end match.
    """
    if not pd.notna(summary) or not str(summary).strip():
        return set()
    txt = _pad_text(summary)
    return {cat for cat, kws in SUMMARY_KEYWORD_MAP.items() if any(kw in txt for kw in kws)}


def phase2b_description(description: Any) -> set[str]:
    """
    Phase 2b: multi-word phrase scan on description text only.
    Phrases only — prevents generic words from causing false positives.
    Text is padded so phrases at start/end match.
    """
    if not pd.notna(description) or not str(description).strip():
        return set()
    txt = _pad_text(description)
    return {cat for cat, phrases in DESCRIPTION_PHRASE_MAP.items() if any(p in txt for p in phrases)}


# ---------------------------------------------------------------------------
# Row-level combiner
# ---------------------------------------------------------------------------

def _map_row(genre_id: Any, summary: Any, description: Any) -> dict[str, int]:
    """
    Combine all three phases and return a binary flag dict.

        Phase 1  – genreId lookup        (deterministic, fast)
        Phase 2a – summary keywords      (single words, short safe text)
        Phase 2b – description phrases   (multi-word only, long text)
        Final    – union of all three
    """
    combined = phase1_genre(genre_id) | phase2a_summary(summary) | phase2b_description(description)
    return {col: int(col in combined) for col in CATEGORY_COLS}


# ---------------------------------------------------------------------------
# Diagnostics helper
# ---------------------------------------------------------------------------

def phase_breakdown(df: pd.DataFrame) -> None:
    """
    Log per-category contribution from each phase:
        p1     – genreId only
        p2a    – summary keywords only
        p2b    – description phrases only
        multi  – caught by 2+ phases (high-confidence hits)
        none   – not caught by any phase
    """
    genre_col = "genreId" if "genreId" in df.columns else "genreid"

    p1  = df.apply(lambda r: phase1_genre(r.get(genre_col)), axis=1)
    p2a = df.apply(lambda r: phase2a_summary(r.get("summary")), axis=1)
    p2b = df.apply(lambda r: phase2b_description(r.get("description")), axis=1)

    log.info("Phase breakdown (%d rows)", len(df))
    log.info("  %-22s  %8s  %8s  %8s  %8s  %8s",
             "category", "p1", "p2a_sum", "p2b_desc", "multi", "none")
    log.info("  " + "-" * 75)
    for cat in CATEGORY_COLS:
        in_p1  = p1.apply(lambda s: cat in s)
        in_p2a = p2a.apply(lambda s: cat in s)
        in_p2b = p2b.apply(lambda s: cat in s)
        sources = in_p1.astype(int) + in_p2a.astype(int) + in_p2b.astype(int)
        log.info(
            "  %-22s  %8d  %8d  %8d  %8d  %8d",
            cat,
            int(in_p1.sum()),
            int(in_p2a.sum()),
            int(in_p2b.sum()),
            int((sources >= 2).sum()),
            int((sources == 0).sum()),
        )
    log.info("  " + "-" * 75)


# ---------------------------------------------------------------------------
# Public enricher API
# ---------------------------------------------------------------------------

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input  : DataFrame with [bundle_id, genreId/genreid, summary, description]
    Output : DataFrame with [bundle_id] + binary category flag columns

    Phases:
        1   genreId lookup          — deterministic, high precision
        2a  summary keywords        — single keywords on short intentional text
        2b  description phrases     — phrases only on long marketing text
    """
    genre_col = "genreId" if "genreId" in df.columns else "genreid"
    log.info(
        "category_mapper: %d rows | phase1=genreId  phase2a=summary-keywords  phase2b=description-phrases",
        len(df),
    )

    flags = df.apply(
        lambda r: _map_row(r.get(genre_col), r.get("summary"), r.get("description")),
        axis=1,
        result_type="expand",
    )

    result = pd.concat([df[["bundle_id"]].reset_index(drop=True), flags], axis=1)

    total  = len(result)
    no_cat = int((result[CATEGORY_COLS].sum(axis=1) == 0).sum())
    log.info(
        "category_mapper: done — %d assigned (%.1f%%)  %d unassigned (%.1f%%)",
        total - no_cat, (total - no_cat) / total * 100,
        no_cat,          no_cat          / total * 100,
    )
    return result
