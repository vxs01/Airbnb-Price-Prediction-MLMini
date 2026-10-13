"""Tiny fake Inside-Airbnb files (same column formats) so the pipeline can be tested offline."""
import json

import numpy as np
import pandas as pd

AMENITIES = ["Wifi", "Kitchen", "TV", "Washer", "Elevator", "Air conditioning", "Heating",
             "Dishwasher", "Pool", "Gym", "Hair dryer", "Self check-in"]
WORDS = ["cozy", "bright", "spacious", "luxury", "studio", "loft", "quiet", "central", "view",
         "metro", "garden", "modern", "small", "family", "terrace"]


def make_city(city, n=800, seed=0):
    rng = np.random.default_rng(seed + len(city))
    lat0, lon0 = {"berlin": (52.52, 13.405), "paris": (48.857, 2.352), "nyc": (40.758, -73.986)}[city]
    base = {"berlin": 4.5, "paris": 5.1, "nyc": 4.9}[city]
    room = rng.choice(["Entire home/apt", "Private room", "Shared room"], n, p=[.6, .35, .05])
    acc = rng.integers(1, 8, n)
    lat, lon = lat0 + rng.normal(0, .04, n), lon0 + rng.normal(0, .05, n)
    lux = rng.random(n) < .25
    amen = [list(rng.choice(AMENITIES, rng.integers(2, 10), replace=False)) for _ in range(n)]
    logp = (base + .15 * acc + .4 * (room == "Entire home/apt") - 6 * np.hypot(lat - lat0, lon - lon0)
            + .4 * lux + .1 * np.array(["Pool" in a for a in amen]) + rng.normal(0, .25, n))
    words = lambda i: " ".join(rng.choice(WORDS, 8)) + (" luxury penthouse" if lux[i] else " budget")
    rating = np.where(rng.random(n) < .2, np.nan, np.round(rng.normal(4.7, .2, n), 2))
    d = pd.DataFrame({
        "name": [words(i) for i in range(n)], "description": [words(i) for i in range(n)],
        "neighborhood_overview": [words(i) for i in range(n)], "host_about": [words(i) for i in range(n)],
        "host_since": pd.Timestamp("2012-01-01") + pd.to_timedelta(rng.integers(0, 4000, n), "D"),
        "host_response_time": rng.choice(["within an hour", "within a day", None], n),
        "host_response_rate": [f"{x}%" for x in rng.integers(50, 101, n)],
        "host_is_superhost": rng.choice(["t", "f"], n), "host_has_profile_pic": "t",
        "host_identity_verified": rng.choice(["t", "f"], n),
        "host_verifications": [str(["email", "phone"][: rng.integers(1, 3)]) for _ in range(n)],
        "host_listings_count": rng.integers(1, 30, n),
        "neighbourhood_cleansed": [f"{city}-{k}" for k in rng.integers(0, 12, n)],
        "neighbourhood_group_cleansed": [f"{city}-g{k}" for k in rng.integers(0, 4, n)],
        "latitude": lat, "longitude": lon, "property_type": rng.choice(["Entire rental unit", "Room"], n),
        "room_type": room, "accommodates": acc, "bathrooms": np.nan,
        "bathrooms_text": [f"{b} baths" for b in rng.choice([1, 1.5, 2], n)],
        "bedrooms": np.clip(acc // 2, 1, None), "beds": acc,
        "amenities": [json.dumps(a) for a in amen],
        "price": [f"${np.exp(v):,.2f}" if rng.random() > .1 else np.nan for v in logp],
        "minimum_nights": rng.integers(1, 7, n), "maximum_nights": 365,
        "has_availability": "t", "availability_30": rng.integers(0, 30, n),
        "availability_365": rng.integers(0, 365, n), "number_of_reviews": rng.poisson(20, n),
        "first_review": "2019-05-01", "last_review": "2025-08-01",
        "review_scores_rating": rating, "review_scores_location": rating,
        "instant_bookable": rng.choice(["t", "f"], n), "calculated_host_listings_count": 1,
        "reviews_per_month": rng.gamma(2, .8, n),
    })
    d["host_since"] = d["host_since"].dt.strftime("%Y-%m-%d")
    if city == "nyc":                # the real NYC file has several empty columns
        d[["host_since", "instant_bookable", "host_verifications"]] = np.nan
    return d


def write_all(folder, n=800):
    for c in ("berlin", "paris", "nyc"):
        make_city(c, n).to_csv(f"{folder}/{c}.csv", index=False)
