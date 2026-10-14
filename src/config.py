"""Paths, column lists and experiment settings shared by every script."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
GDRIVE_CACHE_DIR = ROOT / ".cache" / "airbnb_gdrive"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"

# An explicit AIRBNB_GDRIVE_FOLDER or --data-dir can override this source.
GDRIVE_FOLDER_URL = os.environ.get(
    "AIRBNB_GDRIVE_FOLDER",
    "https://drive.google.com/drive/folders/1jX0ZgSC8IiFIQvxlUd3gXerPbnIGZyfD",
)

SEED = 42
SEEDS = (42, 7, 2024)        # repeated splits for mean ± std in the main table
PRICE_CAP = 500              # paper: drop listings above 500 per night (~1-3 %)
SPLIT = (0.7, 0.2, 0.1)      # train : dev : test, as in the paper

# File names accepted for each city (first one found is used)
CITY_FILES = {
    "berlin": ["berlin.csv", "listings_detail_berlin.csv"],
    "paris": ["paris.csv", "listings_pari26.csv"],
    "nyc": ["nyc.csv", "listings_nyc26.csv"],
}
CITIES = tuple(CITY_FILES)

# City centres, used for the city-independent "distance to centre" feature
CITY_CENTRE = {
    "berlin": (52.5200, 13.4050),   # Mitte
    "paris": (48.8566, 2.3522),     # Hotel de Ville
    "nyc": (40.7580, -73.9855),     # Times Square
}

# --- C: continuous ----------------------------------------------------------
CONTINUOUS = [
    "accommodates", "bathrooms", "bedrooms", "beds",
    "minimum_nights", "maximum_nights",
    "availability_30", "availability_60", "availability_90", "availability_365",
    "number_of_reviews", "number_of_reviews_ltm", "number_of_reviews_l30d", "reviews_per_month",
    "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness",
    "review_scores_checkin", "review_scores_communication", "review_scores_location",
    "review_scores_value",
    "host_listings_count", "calculated_host_listings_count",
    "host_response_rate", "host_acceptance_rate",
]
LAT_LON = ["latitude", "longitude"]          # only meaningful inside one city

# --- O: categorical (one-hot) + list columns (multi-hot) --------------------
CATEGORICAL = [
    "room_type", "property_type", "neighbourhood_cleansed", "neighbourhood_group_cleansed",
    "host_response_time", "host_is_superhost", "host_has_profile_pic",
    "host_identity_verified", "has_availability", "instant_bookable",
]
# neighbourhoods are city-specific, so they are left out of the cross-city features
CITY_SPECIFIC = {"neighbourhood_cleansed", "neighbourhood_group_cleansed"}
MIN_CATEGORY_COUNT = 10      # rarer categories are merged into one "infrequent" column
AMENITY_MIN_SHARE = 0.01     # keep amenities present in >= 1 % of training listings

# --- T: text (TF-IDF unigrams+bigrams -> 50-d SVD per column) ---------------
TEXT = ["name", "description", "neighborhood_overview", "host_about"]
TFIDF_MAX_FEATURES = 5000
SVD_DIM = 50

# --- D: dates (days since 2008-01-01) ---------------------------------------
DATES = ["host_since", "first_review", "last_review"]
DATE_ORIGIN = "2008-01-01"
