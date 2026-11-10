"""
Script to generate sample datasets for testing:
1. sample_reviews.csv: Raw reviews with gmap_id, rating, and text (Case A)
2. sample_aggregated.csv: Pre-aggregated business features (Case B)
"""
import random
import pandas as pd
import numpy as np

# Seed for reproducibility
random.seed(42)
np.random.seed(42)

# --- 1. Raw Reviews Dataset ---
businesses = [
    # Top-tier consistently high rating, large volume
    {"prefix": "elite_dining", "count": 20, "rating_mean": 4.8, "rating_std": 0.3, "reviews_per_biz": (30, 60), "avg_len": 180},
    # Polarized / volatile businesses (mixed reviews)
    {"prefix": "budget_service", "count": 25, "rating_mean": 3.1, "rating_std": 1.4, "reviews_per_biz": (20, 50), "avg_len": 110},
    # Emerging small businesses (few reviews, high rating)
    {"prefix": "local_niche", "count": 30, "rating_mean": 4.6, "rating_std": 0.4, "reviews_per_biz": (2, 8), "avg_len": 90},
    # Struggling businesses (low rating, complaints)
    {"prefix": "neglected_utility", "count": 25, "rating_mean": 2.2, "rating_std": 0.9, "reviews_per_biz": (15, 40), "avg_len": 220},
]

sample_texts = {
    "positive": [
        "Incredible quality and friendly staff! Will definitely come back regularly.",
        "Top-notch service, punctual, polite, and very reasonably priced.",
        "Outstanding experience from start to finish. Highly recommended to everyone in town.",
        "Exceeded all expectations. Fast turnaround and transparent pricing.",
        "A hidden gem in the city! Clean facilities, courteous staff, and wonderful atmosphere."
    ],
    "neutral": [
        "Decent service overall. Got the job done, but nothing particularly stood out.",
        "Average experience. Wait times were a bit long, but staff was polite.",
        "Acceptable quality for the price paid. Might consider them again.",
        "Satisfactory. Met basic requirements without exceeding expectations."
    ],
    "negative": [
        "Extremely disappointing. Unresponsive customer support and unexpected delays.",
        "Poor quality of service. Staff was unprofessional and dismissive of concerns.",
        "Terrible experience. Overcharged for shoddy work and refusal to resolve the issue.",
        "Would not recommend. Long queues, messy work, and unhelpful management.",
        "Completely dissatisfied with the outcome. Avoid if you value your time and money."
    ]
}

reviews = []
for group in businesses:
    for b_idx in range(group["count"]):
        biz_id = f"0x880_{group['prefix']}_{b_idx+1:03d}"
        num_reviews = random.randint(group["reviews_per_biz"][0], group["reviews_per_biz"][1])
        ratings = np.clip(np.random.normal(group["rating_mean"], group["rating_std"], num_reviews), 1.0, 5.0)
        ratings = np.round(ratings)

        for r_val in ratings:
            if r_val >= 4:
                pool = sample_texts["positive"]
            elif r_val == 3:
                pool = sample_texts["neutral"]
            else:
                pool = sample_texts["negative"]
            text = random.choice(pool)
            # Add some variation to length
            if random.random() > 0.5:
                text += " " + random.choice(pool)
            
            reviews.append({
                "gmap_id": biz_id,
                "rating": float(r_val),
                "text": text
            })

df_reviews = pd.DataFrame(reviews)
df_reviews.to_csv("sample_data/sample_reviews.csv", index=False)
print(f"Generated sample_reviews.csv with {len(df_reviews)} reviews across {df_reviews['gmap_id'].nunique()} businesses.")

# --- 2. Pre-Aggregated Tabular Dataset (Case B) ---
# Grouped features directly
agg_data = []
for biz_id, g in df_reviews.groupby("gmap_id"):
    agg_data.append({
        "business_id": biz_id,
        "mean_rating": round(g["rating"].mean(), 3),
        "std_rating": round(g["rating"].std(ddof=0) if len(g) > 1 else 0.0, 3),
        "log_review_count": round(np.log1p(len(g)), 3),
        "avg_review_length": round(g["text"].str.len().mean(), 1)
    })

df_agg = pd.DataFrame(agg_data)
df_agg.to_csv("sample_data/sample_aggregated.csv", index=False)
print(f"Generated sample_aggregated.csv with {len(df_agg)} business profiles.")
