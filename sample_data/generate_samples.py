"""
Script to generate sample datasets for testing:
1. sample_reviews.csv: Raw reviews with gmap_id, business_name, business_type, district, address, latitude, longitude, rating, text (Case A)
2. sample_aggregated.csv: Pre-aggregated business features with location and entity mapping (Case B)
"""
import random
import pandas as pd
import numpy as np

# Seed for reproducibility
random.seed(42)
np.random.seed(42)

# Specific real-world business entity presets
business_presets = {
    "elite_dining": [
        ("Boulder Peak Artisan Bistro", "Commercial Dining & Bistro", "Pearl Street Promenade", "1104 Pearl St", 40.0178, -105.2810),
        ("Pearl Street Craft Roastery", "Specialty Coffee Roasters", "Pearl Street Promenade", "1325 Pearl St", 40.0183, -105.2785),
        ("Chautauqua Grill & Wine Bar", "Commercial Dining & Bistro", "South Foothills District", "900 Baseline Rd", 39.9985, -105.2820),
        ("Flatirons Fine Dining Room", "Commercial Dining & Bistro", "Downtown Commercial Core", "1048 Walnut St", 40.0165, -105.2825),
        ("Alpine Reserve Kitchen", "Commercial Dining & Bistro", "Pearl Street Promenade", "1421 Pearl St", 40.0187, -105.2768),
        ("Sumac Mediterranean Table", "Commercial Dining & Bistro", "Downtown Commercial Core", "1900 Broadway", 40.0152, -105.2792),
        ("Bramble Craft Gastropub", "Commercial Dining & Bistro", "Downtown Commercial Core", "1970 13th St", 40.0161, -105.2780),
        ("Flatirons Dental & Aesthetics", "Health, Dental & Wellness", "Downtown Commercial Core", "1800 15th St", 40.0169, -105.2755),
        ("Highland Artisan Bakery", "Commercial Dining & Bistro", "Pearl Street Promenade", "1212 Pearl St", 40.0180, -105.2800),
        ("Summit Wellness Spa & Retreat", "Health, Dental & Wellness", "South Foothills District", "1002 9th St", 39.9995, -105.2815),
        ("Pike Organic Kitchen", "Commercial Dining & Bistro", "Downtown Commercial Core", "1505 Pearl St", 40.0190, -105.2750),
        ("Vanguard Legal & Wealth Advisory", "Professional Services", "Downtown Commercial Core", "1720 14th St", 40.0166, -105.2769),
        ("Front Range Medical Clinic", "Health, Dental & Wellness", "Downtown Commercial Core", "2100 Broadway", 40.0145, -105.2788),
        ("Colorado Craft Oyster Bar", "Commercial Dining & Bistro", "Pearl Street Promenade", "1128 Pearl St", 40.0179, -105.2808),
        ("Eldorado Spring Wellness Hub", "Health, Dental & Wellness", "South Foothills District", "1250 Canyon Blvd", 40.0148, -105.2820),
        ("Arapahoe Gourmet Charcuterie", "Commercial Dining & Bistro", "East Tech Commerce Hub", "3200 Arapahoe Ave", 40.0149, -105.2510),
        ("Sanitas Family Orthodontics", "Health, Dental & Wellness", "Downtown Commercial Core", "1650 Canyon Blvd", 40.0155, -105.2730),
        ("Valmont Artisan Chocolatier", "Boutique Retail & Food", "Downtown Commercial Core", "1750 15th St", 40.0168, -105.2758),
        ("Crestview Private Wealth", "Professional Services", "Downtown Commercial Core", "1881 9th St", 40.0150, -105.2835),
        ("Silver Fern Botanical Apothecary", "Health, Dental & Wellness", "Pearl Street Promenade", "1440 Pearl St", 40.0188, -105.2762),
    ],
    "budget_service": [
        ("QuickFix Auto Collision & Repair", "Automotive & Fleet Care", "Downtown Commercial Core", "1705 15th St", 40.0162, -105.2750),
        ("Metro Express Plumbing & Drain", "HVAC, Plumbing & Electrical", "North Industrial Corridor", "3100 28th St", 40.0315, -105.2590),
        ("Front Range Budget Towing", "Automotive & Fleet Care", "North Industrial Corridor", "4750 Valmont Rd", 40.0345, -105.2395),
        ("Discount Appliance Repair Pros", "Home Services & Contracting", "Downtown Commercial Core", "2050 26th St", 40.0215, -105.2610),
        ("Reliable 24/7 HVAC Solutions", "HVAC, Plumbing & Electrical", "North Industrial Corridor", "3300 Sterling Cir", 40.0360, -105.2440),
        ("Citywide Commercial Cleaners", "Commercial Cleaning & Janitorial", "Downtown Commercial Core", "1910 28th St", 40.0195, -105.2595),
        ("Discount Brake & Tire Depot", "Automotive & Fleet Care", "North Industrial Corridor", "3800 Arapahoe Ave", 40.0150, -105.2440),
        ("Rapid Rooter Drain Doctors", "HVAC, Plumbing & Electrical", "East Tech Commerce Hub", "4840 Pearl Pkwy", 40.0260, -105.2380),
        ("Affordable Electric Pros", "HVAC, Plumbing & Electrical", "Downtown Commercial Core", "2400 Central Ave", 40.0270, -105.2490),
        ("Valley Transmission Specialist", "Automotive & Fleet Care", "North Industrial Corridor", "4690 Valmont Rd", 40.0340, -105.2410),
        ("Budget City Locksmiths", "Home Services & Contracting", "Downtown Commercial Core", "1820 30th St", 40.0185, -105.2570),
        ("Speedy Auto Glass Express", "Automotive & Fleet Care", "North Industrial Corridor", "3055 28th St", 40.0310, -105.2598),
        ("Midtown Handy Services", "Home Services & Contracting", "Downtown Commercial Core", "2250 26th St", 40.0230, -105.2612),
        ("All-Star Carpet Cleaning", "Commercial Cleaning & Janitorial", "East Tech Commerce Hub", "5300 Arapahoe Ave", 40.0155, -105.2280),
        ("Boulder Valley Muffler Shop", "Automotive & Fleet Care", "North Industrial Corridor", "3220 28th St", 40.0328, -105.2588),
        ("Downtown Emergency Plumbers", "HVAC, Plumbing & Electrical", "Downtown Commercial Core", "1950 14th St", 40.0160, -105.2770),
        ("Colorado Express Moving Co", "Commercial Logistics & Storage", "East Tech Commerce Hub", "4900 Pearl Pkwy", 40.0265, -105.2360),
        ("FastTrack Oil & Lube Express", "Automotive & Fleet Care", "Downtown Commercial Core", "2500 28th St", 40.0255, -105.2590),
        ("Ace Drywall & Home Painters", "Home Services & Contracting", "North Industrial Corridor", "3500 Sterling Cir", 40.0365, -105.2435),
        ("Apex Water Heater Repair", "HVAC, Plumbing & Electrical", "North Industrial Corridor", "2850 Iris Ave", 40.0350, -105.2680),
        ("Economy Pest Control Group", "Home Services & Contracting", "Downtown Commercial Core", "2300 30th St", 40.0235, -105.2568),
        ("Metro Diesel & Fleet Repair", "Automotive & Fleet Care", "North Industrial Corridor", "4720 Valmont Rd", 40.0342, -105.2405),
        ("Boulder Valley Lock & Key", "Home Services & Contracting", "Downtown Commercial Core", "1905 15th St", 40.0158, -105.2752),
        ("Precision Window & Door Fix", "Home Services & Contracting", "North Industrial Corridor", "3400 28th St", 40.0340, -105.2580),
        ("Rapid Clean Pressure Washing", "Commercial Cleaning & Janitorial", "Downtown Commercial Core", "2100 26th St", 40.0220, -105.2615),
    ],
    "local_niche": [
        ("Flatirons Holistic Physical Therapy", "Health, Dental & Wellness", "South Foothills District", "2750 Broadway", 40.0075, -105.2760),
        ("Apex Botanical Apothecary", "Boutique Retail & Food", "Pearl Street Promenade", "1500 Pearl St", 40.0191, -105.2748),
        ("Boulder Boutique Pilates Lab", "Personal Fitness & Pilates", "Downtown Commercial Core", "2020 11th St", 40.0170, -105.2818),
        ("High Country Leather Crafts", "Boutique Retail & Food", "Pearl Street Promenade", "1230 Pearl St", 40.0181, -105.2798),
        ("Sanitas Acupuncture & Wellness", "Health, Dental & Wellness", "South Foothills District", "3100 Broadway", 40.0040, -105.2750),
        ("Summit Ceramic Studio", "Artisan Studio & Crafts", "Downtown Commercial Core", "1630 14th St", 40.0163, -105.2771),
        ("Front Range Ortho Massage", "Health, Dental & Wellness", "South Foothills District", "2800 Baseline Rd", 39.9990, -105.2610),
        ("Pearl Street Hat & Haberdashery", "Boutique Retail & Food", "Pearl Street Promenade", "1310 Pearl St", 40.0182, -105.2789),
        ("Alpine Custom Bicycles", "Boutique Retail & Food", "Downtown Commercial Core", "2425 Canyon Blvd", 40.0150, -105.2630),
        ("Flatirons Canine Day Spa", "Specialty Pet Care & Clinic", "South Foothills District", "1955 28th St", 40.0200, -105.2585),
        ("Boulder Herb & Spice Co", "Boutique Retail & Food", "Pearl Street Promenade", "1410 Pearl St", 40.0186, -105.2770),
        ("Canyon Yoga & Sound Sanctuary", "Personal Fitness & Pilates", "South Foothills District", "1720 Canyon Blvd", 40.0152, -105.2720),
        ("Silver Lake Bookshop & Press", "Boutique Retail & Food", "Pearl Street Promenade", "1140 Pearl St", 40.0180, -105.2805),
        ("Piedmont Custom Goldsmiths", "Boutique Retail & Food", "Downtown Commercial Core", "1815 13th St", 40.0159, -105.2781),
        ("Foothills Acoustic Guitar Workshop", "Artisan Studio & Crafts", "South Foothills District", "2950 Baseline Rd", 39.9992, -105.2580),
        ("Eldorado Organic Micro-Bakery", "Commercial Dining & Bistro", "South Foothills District", "3200 Broadway", 40.0030, -105.2745),
        ("Boulder Valley Bonsai Garden", "Boutique Retail & Food", "South Foothills District", "2550 9th St", 40.0080, -105.2825),
        ("Sanitas Vintage Vinyl & Hi-Fi", "Boutique Retail & Food", "Downtown Commercial Core", "1920 13th St", 40.0160, -105.2782),
        ("Highland Equine & Pet Rehab", "Specialty Pet Care & Clinic", "South Foothills District", "3450 Baseline Rd", 39.9995, -105.2510),
        ("Flatirons Optical & Eyewear", "Health, Dental & Wellness", "Downtown Commercial Core", "1710 15th St", 40.0167, -105.2756),
        ("Apex Tailoring & Bespoke Suits", "Boutique Retail & Food", "Downtown Commercial Core", "1640 15th St", 40.0164, -105.2758),
        ("Chautauqua Sound Recording Lab", "Artisan Studio & Crafts", "South Foothills District", "1200 Baseline Rd", 39.9987, -105.2780),
        ("Alpine Leatherwork & Bootmakers", "Boutique Retail & Food", "Pearl Street Promenade", "1430 Pearl St", 40.0187, -105.2765),
        ("Summit Tea & Botanicals", "Boutique Retail & Food", "Pearl Street Promenade", "1380 Pearl St", 40.0185, -105.2775),
        ("Boulder Ridge Natural Perfumery", "Boutique Retail & Food", "Downtown Commercial Core", "1850 14th St", 40.0168, -105.2767),
        ("Green Mountain Woodcraft Co", "Artisan Studio & Crafts", "South Foothills District", "3600 Table Mesa Dr", 39.9860, -105.2500),
        ("Sanitas Specialty Skincare Bar", "Health, Dental & Wellness", "Downtown Commercial Core", "1915 15th St", 40.0159, -105.2754),
        ("Front Range Micro-Creamery", "Commercial Dining & Bistro", "Pearl Street Promenade", "1215 Pearl St", 40.0181, -105.2802),
        ("Flatirons Custom Framing & Art", "Artisan Studio & Crafts", "Downtown Commercial Core", "1725 15th St", 40.0167, -105.2754),
        ("Boulder Peak Barbers & Shave", "Health, Dental & Wellness", "Pearl Street Promenade", "1115 Pearl St", 40.0179, -105.2812),
    ],
    "neglected_utility": [
        ("Valley Industrial Equipment Yard", "Commercial Logistics & Storage", "North Industrial Corridor", "4800 Valmont Rd", 40.0350, -105.2380),
        ("Eastside Storage & Parcel Hub", "Commercial Logistics & Storage", "East Tech Commerce Hub", "5450 Arapahoe Ave", 40.0158, -105.2250),
        ("Midtown Scrap Metals & Salvage", "Waste & Salvage Management", "North Industrial Corridor", "3600 28th St", 40.0355, -105.2575),
        ("Legacy Parking Management Group", "Commercial Parking Facilities", "Downtown Commercial Core", "1900 11th St", 40.0165, -105.2815),
        ("Pioneer Waste & Roll-Off Bins", "Waste & Salvage Management", "North Industrial Corridor", "4900 63rd St", 40.0390, -105.2150),
        ("Continental Forklift & Heavy Rental", "Commercial Logistics & Storage", "North Industrial Corridor", "3750 Sterling Cir", 40.0370, -105.2420),
        ("Front Range Rail Cargo Depot", "Commercial Logistics & Storage", "East Tech Commerce Hub", "5600 Arapahoe Ave", 40.0160, -105.2210),
        ("Boulder Commercial Pallet Exchange", "Commercial Logistics & Storage", "North Industrial Corridor", "3850 28th St", 40.0375, -105.2560),
        ("Summit Concrete Batch Yard", "Commercial Logistics & Storage", "North Industrial Corridor", "4650 Valmont Rd", 40.0335, -105.2425),
        ("Apex Industrial Chemical Supply", "Waste & Salvage Management", "East Tech Commerce Hub", "5100 Pearl Pkwy", 40.0270, -105.2330),
        ("Midway Heavy Tow & Impound", "Commercial Logistics & Storage", "North Industrial Corridor", "4950 Valmont Rd", 40.0360, -105.2350),
        ("Boulder Auto Wrecking Yard", "Waste & Salvage Management", "North Industrial Corridor", "5000 63rd St", 40.0400, -105.2140),
        ("Flatirons Truck Parking Plaza", "Commercial Parking Facilities", "East Tech Commerce Hub", "5800 Arapahoe Ave", 40.0162, -105.2170),
        ("Rocky Mountain Aggregate Co", "Commercial Logistics & Storage", "North Industrial Corridor", "3900 Sterling Cir", 40.0380, -105.2410),
        ("Universal Scrap & Recycling Yard", "Waste & Salvage Management", "East Tech Commerce Hub", "5250 Pearl Pkwy", 40.0275, -105.2310),
        ("Centennial Heavy Equipment Repair", "Commercial Logistics & Storage", "North Industrial Corridor", "4550 Valmont Rd", 40.0330, -105.2445),
        ("Pioneer Demolition Waste Depot", "Waste & Salvage Management", "North Industrial Corridor", "5100 63rd St", 40.0410, -105.2130),
        ("Front Range Bulk Fuel Transfer", "Commercial Logistics & Storage", "East Tech Commerce Hub", "6000 Arapahoe Ave", 40.0165, -105.2130),
        ("Metro Municipal Impound Lot", "Commercial Parking Facilities", "North Industrial Corridor", "4600 28th St", 40.0420, -105.2550),
        ("Boulder Valley Asphalt Depot", "Commercial Logistics & Storage", "North Industrial Corridor", "4450 Valmont Rd", 40.0325, -105.2460),
        ("Apex Scrap Steel & Processing", "Waste & Salvage Management", "East Tech Commerce Hub", "5400 Pearl Pkwy", 40.0280, -105.2280),
        ("Summit Industrial Tank Storage", "Commercial Logistics & Storage", "East Tech Commerce Hub", "5700 Arapahoe Ave", 40.0161, -105.2190),
        ("Colorado Heavy Freight Terminal", "Commercial Logistics & Storage", "East Tech Commerce Hub", "6100 Arapahoe Ave", 40.0167, -105.2110),
        ("Continental Metal Reclamation", "Waste & Salvage Management", "North Industrial Corridor", "5200 63rd St", 40.0420, -105.2120),
        ("Legacy Municipal Transfer Station", "Waste & Salvage Management", "North Industrial Corridor", "4500 28th St", 40.0415, -105.2555),
    ]
}

businesses_groups = [
    {"prefix": "elite_dining", "rating_mean": 4.8, "rating_std": 0.3, "reviews_per_biz": (30, 60), "avg_len": 180},
    {"prefix": "budget_service", "rating_mean": 3.1, "rating_std": 1.4, "reviews_per_biz": (20, 50), "avg_len": 110},
    {"prefix": "local_niche", "rating_mean": 4.6, "rating_std": 0.4, "reviews_per_biz": (2, 8), "avg_len": 90},
    {"prefix": "neglected_utility", "rating_mean": 2.2, "rating_std": 0.9, "reviews_per_biz": (15, 40), "avg_len": 220},
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
entity_metadata = {}

for group in businesses_groups:
    prefix = group["prefix"]
    preset_list = business_presets[prefix]
    
    for b_idx, (b_name, b_type, b_dist, b_addr, b_lat, b_lng) in enumerate(preset_list):
        biz_id = f"0x880_{prefix}_{b_idx+1:03d}"
        entity_metadata[biz_id] = {
            "business_name": b_name,
            "business_type": b_type,
            "district": b_dist,
            "address": b_addr,
            "latitude": b_lat,
            "longitude": b_lng
        }
        
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
            if random.random() > 0.5:
                text += " " + random.choice(pool)
            
            reviews.append({
                "gmap_id": biz_id,
                "business_name": b_name,
                "business_type": b_type,
                "district": b_dist,
                "address": b_addr,
                "latitude": b_lat,
                "longitude": b_lng,
                "rating": float(r_val),
                "text": text
            })

df_reviews = pd.DataFrame(reviews)
df_reviews.to_csv("sample_data/sample_reviews.csv", index=False)
print(f"Generated sample_reviews.csv with {len(df_reviews)} reviews across {df_reviews['gmap_id'].nunique()} businesses with location & entity fields.")

# --- 2. Pre-Aggregated Tabular Dataset (Case B) ---
agg_data = []
for biz_id, g in df_reviews.groupby("gmap_id"):
    meta = entity_metadata[biz_id]
    agg_data.append({
        "business_id": biz_id,
        "business_name": meta["business_name"],
        "business_type": meta["business_type"],
        "district": meta["district"],
        "address": meta["address"],
        "latitude": meta["latitude"],
        "longitude": meta["longitude"],
        "mean_rating": round(g["rating"].mean(), 3),
        "std_rating": round(g["rating"].std(ddof=0) if len(g) > 1 else 0.0, 3),
        "log_review_count": round(np.log1p(len(g)), 3),
        "avg_review_length": round(g["text"].str.len().mean(), 1)
    })

df_agg = pd.DataFrame(agg_data)
df_agg.to_csv("sample_data/sample_aggregated.csv", index=False)
print(f"Generated sample_aggregated.csv with {len(df_agg)} business profiles with full entity and location metadata.")
