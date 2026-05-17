"""
generate_dataset.py
Intelligent Travel Recommendation System - Canada
MSc Data Science Project

Generates synthetic but realistic Canadian tourism datasets
modelled on Statistics Canada TSRC microdata structure.
"""

import pandas as pd
import numpy as np
import json, os, random
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

# â”€â”€ 1. Destinations catalogue â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
DESTINATIONS = [
    {"id": "D001", "name": "Banff National Park",          "province": "Alberta",            "lat": 51.496, "lon": -115.928, "tags": ["nature","hiking","skiing","wildlife","photography"], "avg_cost_day": 180, "avg_rating": 4.8, "seasons": ["Winter","Spring","Summer","Fall"], "type": "National Park"},
    {"id": "D002", "name": "Whistler Blackcomb",            "province": "British Columbia",   "lat": 50.116, "lon": -122.957, "tags": ["skiing","adventure","nature","food","nightlife"],    "avg_cost_day": 320, "avg_rating": 4.7, "seasons": ["Winter","Summer"],              "type": "Resort"},
    {"id": "D003", "name": "Old Quebec City",               "province": "Quebec",             "lat": 46.814, "lon": -71.208,  "tags": ["history","culture","food","photography","architecture"], "avg_cost_day": 150, "avg_rating": 4.6, "seasons": ["Spring","Summer","Fall"],  "type": "City"},
    {"id": "D004", "name": "Algonquin Provincial Park",     "province": "Ontario",            "lat": 45.531, "lon": -78.354,  "tags": ["nature","wildlife","hiking","photography","camping"],  "avg_cost_day": 90,  "avg_rating": 4.5, "seasons": ["Spring","Summer","Fall"],  "type": "Provincial Park"},
    {"id": "D005", "name": "Vancouver Stanley Park",        "province": "British Columbia",   "lat": 49.301, "lon": -123.148, "tags": ["nature","urban","photography","family","beaches"],     "avg_cost_day": 130, "avg_rating": 4.7, "seasons": ["Spring","Summer","Fall"],  "type": "City"},
    {"id": "D006", "name": "Cavendish Beach PEI",           "province": "Prince Edward Island","lat": 46.498, "lon": -63.396, "tags": ["beaches","nature","family","food","history"],         "avg_cost_day": 120, "avg_rating": 4.4, "seasons": ["Summer"],                  "type": "Beach"},
    {"id": "D007", "name": "Niagara Falls",                 "province": "Ontario",            "lat": 43.096, "lon": -79.071,  "tags": ["nature","photography","family","adventure"],          "avg_cost_day": 140, "avg_rating": 4.5, "seasons": ["Spring","Summer","Fall"],  "type": "Natural Wonder"},
    {"id": "D008", "name": "Churchill Manitoba (Polar Bears)","province": "Manitoba",         "lat": 58.767, "lon": -94.165,  "tags": ["wildlife","nature","photography","adventure"],        "avg_cost_day": 380, "avg_rating": 4.9, "seasons": ["Fall","Winter"],           "type": "Wildlife Destination"},
    {"id": "D009", "name": "Cape Breton Highlands",         "province": "Nova Scotia",        "lat": 46.723, "lon": -60.607,  "tags": ["hiking","nature","culture","photography","scenic"],   "avg_cost_day": 100, "avg_rating": 4.6, "seasons": ["Summer","Fall"],           "type": "National Park"},
    {"id": "D010", "name": "Montreal Food & Culture",       "province": "Quebec",             "lat": 45.508, "lon": -73.588,  "tags": ["food","culture","urban","history","nightlife"],       "avg_cost_day": 160, "avg_rating": 4.7, "seasons": ["Spring","Summer","Fall"],  "type": "City"},
    {"id": "D011", "name": "Jasper Dark Sky Preserve",      "province": "Alberta",            "lat": 52.873, "lon": -117.954, "tags": ["nature","photography","hiking","adventure","astronomy"],"avg_cost_day": 170,"avg_rating": 4.8, "seasons": ["Summer","Fall","Winter"],  "type": "National Park"},
    {"id": "D012", "name": "Tofino Surf & Rainforest",      "province": "British Columbia",   "lat": 49.153, "lon": -125.908, "tags": ["beaches","adventure","nature","wildlife","surfing"],  "avg_cost_day": 200, "avg_rating": 4.6, "seasons": ["Summer","Fall"],           "type": "Coastal"},
    {"id": "D013", "name": "Toronto Waterfront & Museums",  "province": "Ontario",            "lat": 43.638, "lon": -79.382,  "tags": ["urban","culture","food","family","history"],          "avg_cost_day": 190, "avg_rating": 4.4, "seasons": ["Spring","Summer","Fall"],  "type": "City"},
    {"id": "D014", "name": "Icefields Parkway Drive",       "province": "Alberta",            "lat": 52.217, "lon": -117.206, "tags": ["nature","photography","hiking","scenic","adventure"], "avg_cost_day": 140, "avg_rating": 4.9, "seasons": ["Summer","Fall"],           "type": "Scenic Route"},
    {"id": "D015", "name": "Fundy National Park",           "province": "New Brunswick",      "lat": 45.590, "lon": -64.980,  "tags": ["nature","hiking","wildlife","beaches","family"],      "avg_cost_day": 95,  "avg_rating": 4.5, "seasons": ["Spring","Summer","Fall"],  "type": "National Park"},
    {"id": "D016", "name": "Kelowna Wine Country",          "province": "British Columbia",   "lat": 49.888, "lon": -119.496, "tags": ["food","culture","nature","scenic","photography"],     "avg_cost_day": 220, "avg_rating": 4.6, "seasons": ["Summer","Fall"],           "type": "Wine Region"},
    {"id": "D017", "name": "Ottawa Parliament Hill",        "province": "Ontario",            "lat": 45.424, "lon": -75.700,  "tags": ["history","culture","urban","family","architecture"],  "avg_cost_day": 130, "avg_rating": 4.4, "seasons": ["Spring","Summer","Fall"],  "type": "City"},
    {"id": "D018", "name": "Yukon Northern Lights",         "province": "Yukon",              "lat": 60.721, "lon": -135.057, "tags": ["nature","photography","adventure","wildlife"],        "avg_cost_day": 280, "avg_rating": 4.9, "seasons": ["Winter","Fall"],           "type": "Remote Wilderness"},
    {"id": "D019", "name": "Gros Morne National Park",     "province": "Newfoundland",       "lat": 49.597, "lon": -57.779,  "tags": ["hiking","nature","photography","culture","wildlife"],  "avg_cost_day": 110, "avg_rating": 4.7, "seasons": ["Summer","Fall"],           "type": "National Park"},
    {"id": "D020", "name": "Calgary Stampede & Foothills", "province": "Alberta",            "lat": 51.044, "lon": -114.062,  "tags": ["culture","food","adventure","history","family"],     "avg_cost_day": 160, "avg_rating": 4.5, "seasons": ["Summer"],                  "type": "City/Event"},

# --- Auto-generate additional synthetic destinations to reach 100 ---

    {"id": "D021", "name": "Bay of Fundy", "province": "New Brunswick", "lat": 45.254, "lon": -64.514, "tags": ["nature","scenic","wildlife","photography"], "avg_cost_day": 110, "avg_rating": 4.7, "seasons": ["Spring","Summer","Fall"], "type": "Natural Wonder"},
    {"id": "D022", "name": "Lake Louise", "province": "Alberta", "lat": 51.425, "lon": -116.177, "tags": ["nature","hiking","skiing","photography"], "avg_cost_day": 200, "avg_rating": 4.8, "seasons": ["Winter","Summer","Fall"], "type": "Lake"},
    {"id": "D023", "name": "St. John's Signal Hill", "province": "Newfoundland", "lat": 47.572, "lon": -52.681, "tags": ["history","scenic","photography","hiking"], "avg_cost_day": 90, "avg_rating": 4.5, "seasons": ["Spring","Summer","Fall"], "type": "Historic Site"},
    {"id": "D024", "name": "Whitehorse Yukon River", "province": "Yukon", "lat": 60.721, "lon": -135.056, "tags": ["nature","adventure","wildlife","photography"], "avg_cost_day": 170, "avg_rating": 4.6, "seasons": ["Summer","Fall"], "type": "River"},
    {"id": "D025", "name": "Saskatoon Meewasin Valley", "province": "Saskatchewan", "lat": 52.133, "lon": -106.670, "tags": ["nature","hiking","scenic","photography"], "avg_cost_day": 80, "avg_rating": 4.4, "seasons": ["Spring","Summer","Fall"], "type": "Park"},
    {"id": "D026", "name": "Magdalen Islands", "province": "Quebec", "lat": 47.383, "lon": -61.866, "tags": ["beaches","nature","wildlife","photography"], "avg_cost_day": 140, "avg_rating": 4.7, "seasons": ["Summer"], "type": "Islands"},
    {"id": "D027", "name": "Bruce Peninsula National Park", "province": "Ontario", "lat": 45.229, "lon": -81.525, "tags": ["nature","hiking","photography","camping"], "avg_cost_day": 120, "avg_rating": 4.8, "seasons": ["Spring","Summer","Fall"], "type": "National Park"},
    {"id": "D028", "name": "Mont Tremblant", "province": "Quebec", "lat": 46.210, "lon": -74.584, "tags": ["skiing","nature","adventure","photography"], "avg_cost_day": 210, "avg_rating": 4.6, "seasons": ["Winter","Summer","Fall"], "type": "Resort"},
    {"id": "D029", "name": "Prince Albert National Park", "province": "Saskatchewan", "lat": 53.571, "lon": -106.465, "tags": ["nature","wildlife","hiking","photography"], "avg_cost_day": 100, "avg_rating": 4.5, "seasons": ["Spring","Summer","Fall"], "type": "National Park"},

]

# --- Auto-generate additional synthetic destinations to reach 50 ---
provinces = ["Ontario","British Columbia","Quebec","Alberta","Nova Scotia","Manitoba","New Brunswick","Prince Edward Island","Saskatchewan","Yukon","Newfoundland"]
seasons_all = ["Winter","Spring","Summer","Fall"]
tags_all = ["nature","hiking","skiing","wildlife","photography","adventure","food","nightlife","culture","history","urban","family","beaches"]
types_all = ["National Park","Resort","City","Provincial Park","Beach","Natural Wonder","Wildlife Destination","Wine Region","Scenic Route","City/Event","Historic Site","River","Islands","Coastal","Remote Wilderness","Park"]

# Ensure at least one destination for every province, season, and tag
idx = 31
for prov in provinces:
    for season in seasons_all:
        DESTINATIONS.append({
            "id": f"D{idx:03d}",
            "name": f"{prov} {season} Adventure",
            "province": prov,
            "lat": round(random.uniform(43.0, 62.0), 3),
            "lon": round(random.uniform(-140.0, -52.0), 3),
            "tags": random.sample(tags_all, k=4),
            "avg_cost_day": random.randint(50, 500),
            "avg_rating": round(random.uniform(4.3, 4.9), 1),
            "seasons": [season],
            "type": random.choice(types_all),
        })
        idx += 1
# Fill up to 50 with diverse destinations
while len(DESTINATIONS) < 50:
    DESTINATIONS.append({
        "id": f"D{idx:03d}",
        "name": f"Diverse Destination {idx}",
        "province": random.choice(provinces),
        "lat": round(random.uniform(43.0, 62.0), 3),
        "lon": round(random.uniform(-140.0, -52.0), 3),
        "tags": random.sample(tags_all, k=5),
        "avg_cost_day": random.randint(50, 500),
        "avg_rating": round(random.uniform(4.3, 4.9), 1),
        "seasons": random.sample(seasons_all, k=random.randint(1,4)),
        "type": random.choice(types_all),
    })
    idx += 1

# â”€â”€ 2. User profiles â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
INTEREST_COMBOS = [
    ["nature","hiking","photography"],
    ["skiing","adventure","food"],
    ["culture","history","urban"],
    ["wildlife","photography","nature"],
    ["beaches","family","food"],
    ["adventure","hiking","skiing"],
    ["food","culture","nightlife"],
    ["nature","camping","wildlife"],
    ["photography","scenic","nature"],
    ["family","urban","history"],
]

def generate_users(n=500):
    rows = []
    for i in range(n):
        combo = random.choice(INTEREST_COMBOS)
        extra = random.sample(["photography","nature","food","adventure","culture","wildlife","family","history","urban","beaches","skiing"], k=random.randint(0,2))
        interests = list(set(combo + extra))
        rows.append({
            "user_id":    f"U{i+1:04d}",
            "age":        int(np.clip(np.random.normal(38, 12), 18, 75)),
            "budget_day": int(np.clip(np.random.normal(180, 80), 50, 500)),
            "duration":   int(np.clip(np.random.normal(7, 3), 1, 30)),
            "province_home": random.choice(["Ontario","British Columbia","Quebec","Alberta","Nova Scotia","Manitoba"]),
            "interests":  "|".join(interests),
            "pref_season": random.choice(["Winter","Spring","Summer","Fall",""]),
            "solo_or_group": random.choice(["Solo","Couple","Family","Group"]),
        })
    return pd.DataFrame(rows)

# â”€â”€ 3. Ratings matrix (user Ã— destination) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def generate_ratings(users_df, destinations, sparsity=0.12):
    rows = []
    for _, u in users_df.iterrows():
        u_interests = set(u["interests"].split("|"))
        for d in destinations:
            if random.random() > sparsity:
                continue
            tag_overlap = len(u_interests & set(d["tags"])) / max(len(d["tags"]), 1)
            budget_fit  = 1 - abs(u["budget_day"] - d["avg_cost_day"]) / 500
            season_fit  = 1 if (not u["pref_season"] or u["pref_season"] in d["seasons"]) else 0.3
            base        = d["avg_rating"] * 0.5 + tag_overlap * 1.5 + budget_fit * 0.8 + season_fit * 0.7
            rating      = float(np.clip(base + np.random.normal(0, 0.3), 1, 5))
            rows.append({"user_id": u["user_id"], "dest_id": d["id"], "rating": round(rating, 1),
                         "visit_date": (datetime(2022,1,1) + timedelta(days=random.randint(0,730))).strftime("%Y-%m-%d")})
    return pd.DataFrame(rows)

# â”€â”€ 4. Weather features â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def generate_weather():
    rows = []
    for d in DESTINATIONS:
        for month in range(1,13):
            season = ["Winter","Winter","Spring","Spring","Spring","Summer","Summer","Summer","Fall","Fall","Fall","Winter"][month-1]
            lat_factor = (d["lat"] - 43) / 20
            temp = 20 - lat_factor*15 + [(-15),(-12),(-5),(5),(13),(18),(22),(21),(14),(7),(-2),(-10)][month-1]
            rows.append({"dest_id": d["id"], "month": month, "season": season,
                         "avg_temp_c": round(temp + np.random.normal(0,1), 1),
                         "precip_mm":  round(max(0, 60 + np.random.normal(0,20)), 1),
                         "visitor_index": round(max(0.1, ([0.3,0.3,0.5,0.7,0.8,1.0,1.0,0.9,0.8,0.6,0.3,0.4][month-1]) + np.random.normal(0,0.05)), 2)})
    return pd.DataFrame(rows)


HOTEL_PREFIXES = ["Grand", "Summit", "Northern", "Maple", "Lakeview", "Wildflower", "Aurora", "Harbour", "Cedar", "Glacier"]
HOTEL_SUFFIXES = ["Hotel", "Lodge", "Suites", "Inn", "Retreat", "Resort", "House", "Stay", "Residences", "Mews"]
ATTRACTION_TEMPLATES = {
    "nature": [("Scenic Lookout", "scenic"), ("Nature Trail", "nature"), ("Valley Viewpoint", "scenic")],
    "hiking": [("Ridge Trail", "hiking"), ("Summit Walk", "hiking"), ("Forest Loop", "hiking")],
    "skiing": [("Alpine Run", "skiing"), ("Snow Adventure Zone", "skiing"), ("Peak Lift Experience", "skiing")],
    "wildlife": [("Wildlife Sanctuary", "wildlife"), ("Animal Viewing Point", "wildlife"), ("Eco Discovery Walk", "nature")],
    "photography": [("Photo Point", "scenic"), ("Sunrise Deck", "scenic"), ("Panorama Platform", "photography")],
    "adventure": [("Adventure Course", "adventure"), ("Guided Excursion Hub", "tour"), ("Explorer Basecamp", "adventure")],
    "food": [("Market Hall", "food"), ("Tasting Plaza", "food"), ("Culinary Studio", "food")],
    "nightlife": [("Night Market", "nightlife"), ("Live Music Venue", "nightlife"), ("Evening Lights Walk", "nightlife")],
    "culture": [("Cultural Centre", "cultural"), ("Arts Pavilion", "cultural"), ("Heritage Gallery", "museum")],
    "history": [("Historic Quarter", "historic"), ("Heritage Museum", "museum"), ("Founders Walk", "historic")],
    "urban": [("City Square", "urban"), ("Waterfront Promenade", "urban"), ("Downtown Hub", "shopping")],
    "family": [("Family Fun Park", "family"), ("Discovery Centre", "family"), ("Interactive Zone", "family")],
    "beaches": [("Beachfront Boardwalk", "beach"), ("Coastal Access Point", "beach"), ("Sunset Shore", "beach")],
}
FALLBACK_ATTRACTIONS = [("Visitor Centre", "landmark"), ("Local Museum", "museum"), ("Community Park", "park")]
MEAL_CUISINES = {
    "nature": ["campfire grill", "canadian", "cafe"],
    "hiking": ["trail cafe", "snacks", "canadian"],
    "skiing": ["alpine grill", "bistro", "casual"],
    "wildlife": ["northern cuisine", "cafe", "casual"],
    "photography": ["brunch", "cafe", "desserts"],
    "adventure": ["bbq", "gastropub", "casual"],
    "food": ["fine dining", "bistro", "international"],
    "nightlife": ["pub", "gourmet", "late-night"],
    "culture": ["french", "bistro", "fusion"],
    "history": ["heritage cuisine", "tea house", "bistro"],
    "urban": ["street food", "international", "casual"],
    "family": ["family dining", "pizza", "grill"],
    "beaches": ["seafood", "grill", "beach cafe"],
}
FALLBACK_CUISINES = ["casual", "cafe", "grill"]
MEAL_PREFIXES = ["North", "Cedar", "Harbour", "Summit", "Wild", "Golden", "Fireside", "Blue", "Trailhead", "Maple"]
MEAL_SUFFIXES = ["Kitchen", "Table", "Bistro", "Cafe", "Grill", "Eatery", "House", "Oven", "Room", "Bar"]


def build_name_parts(destination_name):
    words = [word for word in destination_name.replace("&", " ").replace("-", " ").split() if word.isalpha()]
    if not words:
        return "Travel", "Place"
    lead = words[0]
    tail = words[-1] if len(words) > 1 else lead
    return lead, tail


def row_target(index, total_items, total_rows):
    base = total_rows // total_items
    remainder = total_rows % total_items
    return base + (1 if index < remainder else 0)


def generate_hotels(destinations, total_rows=250):
    rows = []
    for index, d in enumerate(destinations):
        lead, tail = build_name_parts(d["name"])
        count = row_target(index, len(destinations), total_rows)
        nightly_base = max(45, int(d["avg_cost_day"] * 0.75))
        for variant in range(count):
            prefix = HOTEL_PREFIXES[(index + variant) % len(HOTEL_PREFIXES)]
            suffix = HOTEL_SUFFIXES[(index * 2 + variant) % len(HOTEL_SUFFIXES)]
            rows.append({
                "destination_id": d["id"],
                "name": f"{prefix} {lead} {suffix}" if variant % 2 == 0 else f"{tail} {suffix}",
                "cost": int(np.clip(nightly_base + variant * 18 + (index % 5) * 12, 40, 650)),
                "lat": round(d["lat"] + ((variant - count / 2) * 0.012), 3),
                "lng": round(d["lon"] + ((count / 2 - variant) * 0.012), 3),
            })
    return pd.DataFrame(rows)


def generate_attractions(destinations, total_rows=250):
    rows = []
    for index, d in enumerate(destinations):
        lead, _ = build_name_parts(d["name"])
        count = row_target(index, len(destinations), total_rows)
        attraction_pool = []
        for tag in d["tags"]:
            attraction_pool.extend(ATTRACTION_TEMPLATES.get(tag, []))
        if not attraction_pool:
            attraction_pool = FALLBACK_ATTRACTIONS
        for variant in range(count):
            template_name, attraction_type = attraction_pool[variant % len(attraction_pool)]
            rows.append({
                "destination_id": d["id"],
                "name": f"{lead} {template_name}",
                "cost": int(max(0, (variant % 4) * 10 + (index % 3) * 5)),
                "type": attraction_type,
                "lat": round(d["lat"] + ((variant - count / 2) * 0.01), 3),
                "lng": round(d["lon"] + ((variant - count / 2) * 0.009), 3),
            })
    return pd.DataFrame(rows)


def generate_meals(destinations, total_rows=250):
    rows = []
    for index, d in enumerate(destinations):
        lead, _ = build_name_parts(d["name"])
        count = row_target(index, len(destinations), total_rows)
        cuisine_pool = []
        for tag in d["tags"]:
            cuisine_pool.extend(MEAL_CUISINES.get(tag, []))
        if not cuisine_pool:
            cuisine_pool = FALLBACK_CUISINES
        for variant in range(count):
            prefix = MEAL_PREFIXES[(index + variant) % len(MEAL_PREFIXES)]
            suffix = MEAL_SUFFIXES[(index * 3 + variant) % len(MEAL_SUFFIXES)]
            rows.append({
                "destination_id": d["id"],
                "name": f"{prefix} {lead} {suffix}",
                "cost": int(np.clip(18 + (index % 6) * 4 + variant * 6, 12, 120)),
                "type": cuisine_pool[variant % len(cuisine_pool)],
                "lat": round(d["lat"] + ((count / 2 - variant) * 0.008), 3),
                "lng": round(d["lon"] + ((variant - count / 2) * 0.008), 3),
            })
    return pd.DataFrame(rows)

# â”€â”€ Run and save â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    dest_df  = pd.DataFrame(DESTINATIONS)
    users_df = generate_users(500)
    ratings  = generate_ratings(users_df, DESTINATIONS, sparsity=0.15)
    weather  = generate_weather()
    hotels   = generate_hotels(DESTINATIONS, total_rows=250)
    attractions = generate_attractions(DESTINATIONS, total_rows=250)
    meals = generate_meals(DESTINATIONS, total_rows=250)

    dest_df.to_csv("data/destinations.csv", index=False)
    users_df.to_csv("data/users.csv", index=False)
    ratings.to_csv("data/ratings.csv", index=False)
    weather.to_csv("data/weather.csv", index=False)
    hotels.to_csv("data/hotels.csv", index=False)
    attractions.to_csv("data/attractions.csv", index=False)
    meals.to_csv("data/meals.csv", index=False)

    print(f"Destinations : {len(dest_df)}")
    print(f"Users        : {len(users_df)}")
    print(f"Ratings      : {len(ratings)} (sparsity {1-len(ratings)/(len(users_df)*len(dest_df)):.2%})")
    print(f"Weather rows : {len(weather)}")
    print(f"Hotels       : {len(hotels)}")
    print(f"Attractions  : {len(attractions)}")
    print(f"Meals        : {len(meals)}")
    print("All datasets saved to data/")
