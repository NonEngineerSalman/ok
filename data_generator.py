"""
Bangladesh Flood Disaster Management Interactive Tool
Dataset Generator: Generates 390,499 records representing Bangladeshi administrative divisions,
flood risk metrics, elevation, population, and shelters.
Splits data programmatically: 40% train (156,200), 30% test (117,149), 30% validate (117,150).
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd

# Set random seed for reproducible realistic generation
np.random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

DATASET_CSV = os.path.join(DATA_DIR, 'dataset.csv')
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
TEST_CSV = os.path.join(DATA_DIR, 'test.csv')
VALIDATE_CSV = os.path.join(DATA_DIR, 'validate.csv')
PICKLE_FILE = os.path.join(DATA_DIR, 'processed_data.pkl')

TOTAL_RECORDS = 390499
TRAIN_COUNT = 156200     # 40.00%
TEST_COUNT = 117149      # 30.00%
VALIDATE_COUNT = 117150  # 30.00% (156200 + 117149 + 117150 = 390499)

# Comprehensive Bangladeshi Administrative Hierarchy (Zillas, Upazilas, Thanas with Bengali names & geo-coordinates)
DISTRICT_PROFILES = [
    # Sylhet Division (Extreme Flash Flood / Haor Basin)
    {
        "zilla": "Sunamganj (সুনামগঞ্জ)", "division": "Sylhet", "base_lat": 25.0658, "base_lon": 91.3950,
        "flood_propensity": 0.88, "base_elevation": 4.5,
        "upazilas": [
            ("Tahirpur (তাহিরপুর)", "Tahirpur Thana"),
            ("Bishwamvarpur (বিশ্বম্ভরপুর)", "Bishwamvarpur Thana"),
            ("Dharmapasha (ধর্মপাশা)", "Dharmapasha Thana"),
            ("Chhatak (ছাতক)", "Chhatak Thana"),
            ("Jagannathpur (জগন্নাথপুর)", "Jagannathpur Thana"),
            ("Shalla (শাল্লা)", "Shalla Thana"),
            ("Derai (দেরাই)", "Derai Thana"),
            ("Dowarabazar (দোয়ারাবাজার)", "Dowarabazar Thana"),
            ("Sunamganj Sadar (সুনামগঞ্জ সদর)", "Sadar Thana")
        ]
    },
    {
        "zilla": "Sylhet (সিলেট)", "division": "Sylhet", "base_lat": 24.8949, "base_lon": 91.8687,
        "flood_propensity": 0.82, "base_elevation": 8.0,
        "upazilas": [
            ("Companiganj (কোম্পানীগঞ্জ)", "Companiganj Thana"),
            ("Gowainghat (গোয়াইনঘাট)", "Gowainghat Thana"),
            ("Jaintiapur (জৈন্তাপুর)", "Jaintiapur Thana"),
            ("Kanaighat (কানাইঘাট)", "Kanaighat Thana"),
            ("Zakiganj (জকিগঞ্জ)", "Zakiganj Thana"),
            ("Beanibazar (বিয়ানীবাজার)", "Beanibazar Thana"),
            ("Golapganj (গোলাপগঞ্জ)", "Golapganj Thana"),
            ("Sylhet Sadar (সিলেট সদর)", "Kotwali Thana"),
            ("Fenchuganj (ফেঞ্চুগঞ্জ)", "Fenchuganj Thana")
        ]
    },
    {
        "zilla": "Kurigram (কুড়িগ্রাম)", "division": "Rangpur", "base_lat": 25.8072, "base_lon": 89.6295,
        "flood_propensity": 0.89, "base_elevation": 24.0,
        "upazilas": [
            ("Ulipur (উলিপুর)", "Ulipur Thana"),
            ("Chilmari (চিলমারী)", "Chilmari Thana"),
            ("Roumari (রৌমারী)", "Roumari Thana"),
            ("Char Rajibpur (রাজিবপুর)", "Rajibpur Thana"),
            ("Nageshwari (নাগেশ্বরী)", "Nageshwari Thana"),
            ("Bhurungamari (ভুরুঙ্গামারী)", "Bhurungamari Thana"),
            ("Kurigram Sadar (কুড়িগ্রাম সদর)", "Kurigram Thana"),
            ("Phulbari (ফুলবাড়ী)", "Phulbari Thana"),
            ("Rajarhat (রাজারহাট)", "Rajarhat Thana")
        ]
    },
    {
        "zilla": "Feni (ফেনী)", "division": "Chattogram", "base_lat": 23.0159, "base_lon": 91.3976,
        "flood_propensity": 0.85, "base_elevation": 6.5,
        "upazilas": [
            ("Parshuram (পরশুরাম)", "Parshuram Thana"),
            ("Phulgazi (ফুলগাজী)", "Phulgazi Thana"),
            ("Chhagalnaiya (ছাগলনাইয়া)", "Chhagalnaiya Thana"),
            ("Feni Sadar (ফেনী সদর)", "Feni Sadar Thana"),
            ("Daganbhuiyan (দাগনভূঞা)", "Daganbhuiyan Thana"),
            ("Sonagazi (সোনাগাজী)", "Sonagazi Thana")
        ]
    },
    {
        "zilla": "Sirajganj (সিরাজগঞ্জ)", "division": "Rajshahi", "base_lat": 24.4534, "base_lon": 89.7008,
        "flood_propensity": 0.84, "base_elevation": 14.0,
        "upazilas": [
            ("Kazipur (কাজীপুর)", "Kazipur Thana"),
            ("Belkuchi (বেলকুচি)", "Belkuchi Thana"),
            ("Chauhali (চৌহালী)", "Chauhali Thana"),
            ("Shahjadpur (শাহজাদপুর)", "Shahjadpur Thana"),
            ("Tarash (তাড়াশ)", "Tarash Thana"),
            ("Ullapara (উল্লাপাড়া)", "Ullapara Thana"),
            ("Sirajganj Sadar (সিরাজগঞ্জ সদর)", "Sadar Thana"),
            ("Kamarkhanda (কামারখন্দ)", "Kamarkhanda Thana")
        ]
    },
    {
        "zilla": "Bogura (বগুড়া)", "division": "Rajshahi", "base_lat": 24.8465, "base_lon": 89.3777,
        "flood_propensity": 0.68, "base_elevation": 20.0,
        "upazilas": [
            ("Sariakandi (সারিয়াকান্দি)", "Sariakandi Thana"),
            ("Dhunat (ধুনট)", "Dhunat Thana"),
            ("Sonatala (সোনাতলা)", "Sonatala Thana"),
            ("Bogura Sadar (বগুড়া সদর)", "Bogura Thana"),
            ("Shibganj (শিবগঞ্জ)", "Shibganj Thana"),
            ("Gabtali (গাবতলী)", "Gabtali Thana"),
            ("Sherpur (শেরপুর)", "Sherpur Thana")
        ]
    },
    {
        "zilla": "Gaibandha (গাইবান্ধা)", "division": "Rangpur", "base_lat": 25.3288, "base_lon": 89.5430,
        "flood_propensity": 0.83, "base_elevation": 22.0,
        "upazilas": [
            ("Fulchhari (ফুলছড়ি)", "Fulchhari Thana"),
            ("Saghata (সাঘাটা)", "Saghata Thana"),
            ("Sundarganj (সুন্দরগঞ্জ)", "Sundarganj Thana"),
            ("Gaibandha Sadar (গাইবান্ধা সদর)", "Sadar Thana"),
            ("Gobindaganj (গোবিন্দগঞ্জ)", "Gobindaganj Thana")
        ]
    },
    {
        "zilla": "Noakhali (নোয়াখালী)", "division": "Chattogram", "base_lat": 22.8696, "base_lon": 91.0995,
        "flood_propensity": 0.79, "base_elevation": 5.0,
        "upazilas": [
            ("Begumganj (বেগমগঞ্জ)", "Begumganj Thana"),
            ("Companiganj (কোম্পানীগঞ্জ)", "Companiganj Thana"),
            ("Senbagh (সেনবাগ)", "Senbagh Thana"),
            ("Hatiya (হাতিয়া)", "Hatiya Thana"),
            ("Subarnachar (সুবর্ণচর)", "Subarnachar Thana"),
            ("Noakhali Sadar (নোয়াখালী সদর)", "Sudharam Thana")
        ]
    },
    {
        "zilla": "Netrokona (নেত্রকোণা)", "division": "Mymensingh", "base_lat": 24.8709, "base_lon": 90.7279,
        "flood_propensity": 0.80, "base_elevation": 11.0,
        "upazilas": [
            ("Khaliajuri (খালিয়াজুড়ি)", "Khaliajuri Thana"),
            ("Madan (মদন)", "Madan Thana"),
            ("Mohanganj (মোহনগঞ্জ)", "Mohanganj Thana"),
            ("Kalmakanda (কলমাকান্দা)", "Kalmakanda Thana"),
            ("Netrokona Sadar (নেত্রকোণা সদর)", "Sadar Thana")
        ]
    },
    {
        "zilla": "Jamalpur (জামালপুর)", "division": "Mymensingh", "base_lat": 24.9375, "base_lon": 89.9378,
        "flood_propensity": 0.77, "base_elevation": 17.0,
        "upazilas": [
            ("Dewanganj (দেওয়ানগঞ্জ)", "Dewanganj Thana"),
            ("Islampur (ইসলামপুর)", "Islampur Thana"),
            ("Madarganj (মাদারগঞ্জ)", "Madarganj Thana"),
            ("Melandaha (মেলান্দহ)", "Melandaha Thana"),
            ("Sarishabari (সরিষাবাড়ী)", "Sarishabari Thana")
        ]
    },
    {
        "zilla": "Habiganj (হবিগঞ্জ)", "division": "Sylhet", "base_lat": 24.3749, "base_lon": 91.4155,
        "flood_propensity": 0.73, "base_elevation": 8.5,
        "upazilas": [
            ("Ajmiriganj (আজমিরীগঞ্জ)", "Ajmiriganj Thana"),
            ("Baniachong (বানিয়াচং)", "Baniachong Thana"),
            ("Lakhai (লাখাই)", "Lakhai Thana"),
            ("Nabiganj (নবীগঞ্জ)", "Nabiganj Thana"),
            ("Habiganj Sadar (হবিগঞ্জ সদর)", "Sadar Thana")
        ]
    },
    {
        "zilla": "Moulvibazar (মৌলভীবাজার)", "division": "Sylhet", "base_lat": 24.4829, "base_lon": 91.7774,
        "flood_propensity": 0.71, "base_elevation": 12.0,
        "upazilas": [
            ("Kulaura (কুলাউড়া)", "Kulaura Thana"),
            ("Rajnagar (রাজনগর)", "Rajnagar Thana"),
            ("Kamalganj (কমলগঞ্জ)", "Kamalganj Thana"),
            ("Barlekha (বড়লেখা)", "Barlekha Thana"),
            ("Moulvibazar Sadar (মৌলভীবাজার সদর)", "Sadar Thana")
        ]
    },
    {
        "zilla": "Dhaka (ঢাকা)", "division": "Dhaka", "base_lat": 23.8103, "base_lon": 90.4125,
        "flood_propensity": 0.42, "base_elevation": 9.5,
        "upazilas": [
            ("Dhamrai (ধামরাই)", "Dhamrai Thana"),
            ("Dohar (দোহার)", "Dohar Thana"),
            ("Keraniganj (কেরানীগঞ্জ)", "Keraniganj Thana"),
            ("Nawabganj (নবাবগঞ্জ)", "Nawabganj Thana"),
            ("Savar (সাভার)", "Savar Thana")
        ]
    },
    {
        "zilla": "Munshiganj (মুন্সীগঞ্জ)", "division": "Dhaka", "base_lat": 23.5422, "base_lon": 90.5305,
        "flood_propensity": 0.75, "base_elevation": 6.0,
        "upazilas": [
            ("Lohajang (লৌহজং)", "Lohajang Thana"),
            ("Sreenagar (শ্রীনগর)", "Sreenagar Thana"),
            ("Sirajdikhan (সিরাজদিখান)", "Sirajdikhan Thana"),
            ("Munshiganj Sadar (মুন্সীগঞ্জ সদর)", "Sadar Thana")
        ]
    },
    {
        "zilla": "Chattogram (চট্টগ্রাম)", "division": "Chattogram", "base_lat": 22.3569, "base_lon": 91.7832,
        "flood_propensity": 0.65, "base_elevation": 15.0,
        "upazilas": [
            ("Hathazari (হাটহাজারী)", "Hathazari Thana"),
            ("Raozan (রাউজান)", "Raozan Thana"),
            ("Fatikchhari (ফটিকছড়ি)", "Fatikchhari Thana"),
            ("Boalkhali (বোয়ালখালী)", "Boalkhali Thana"),
            ("Patiya (পটিয়া)", "Patiya Thana")
        ]
    },
    {
        "zilla": "Cox's Bazar (কক্সবাজার)", "division": "Chattogram", "base_lat": 21.4272, "base_lon": 92.0058,
        "flood_propensity": 0.69, "base_elevation": 6.0,
        "upazilas": [
            ("Chakaria (চকোরিয়া)", "Chakaria Thana"),
            ("Pekua (পেকুয়া)", "Pekua Thana"),
            ("Ramu (রামু)", "Ramu Thana"),
            ("Teknaf (টেকনাফ)", "Teknaf Thana"),
            ("Cox's Bazar Sadar (কক্সবাজার সদর)", "Sadar Thana")
        ]
    }
]

VILLAGE_PREFIXES = [
    "Kandi", "Char", "Uttar", "Dakshin", "Purba", "Paschim", "Madhyam", "Boro", "Chhoto",
    "Shimul", "Rampur", "Fatehpur", "Bhanga", "Gopalpur", "Joypur", "Alampur", "Sonapur",
    "Ujangaon", "Nayagram", "Bishanpur", "Kashipur", "Shantinagar", "Gobindapur", "Durgapur"
]

VILLAGE_SUFFIXES = [
    "gaon", "pur", "nagar", "kandi", "para", "bari", "danga", "tali", "mor", "bazar", "char", "hati"
]

def generate_area_names(count):
    """Generates realistic Bangladeshi village / settlement names."""
    p_indices = np.random.randint(0, len(VILLAGE_PREFIXES), size=count)
    s_indices = np.random.randint(0, len(VILLAGE_SUFFIXES), size=count)
    names = [f"{VILLAGE_PREFIXES[p]} {VILLAGE_SUFFIXES[s].capitalize()}" for p, s in zip(p_indices, s_indices)]
    return names

def create_dataset():
    """Generates the full dataset (390,499 records) and splits it."""
    print(f"[*] Starting dataset generation for {TOTAL_RECORDS} records...")
    start_time = time.time()

    # Pre-calculate counts per district profile
    num_profiles = len(DISTRICT_PROFILES)
    base_count = TOTAL_RECORDS // num_profiles
    remainder = TOTAL_RECORDS % num_profiles

    records_per_profile = [base_count + (1 if i < remainder else 0) for i in range(num_profiles)]

    dfs = []
    current_id = 1

    for idx, profile in enumerate(DISTRICT_PROFILES):
        p_count = records_per_profile[idx]
        upazilas = profile["upazilas"]
        u_count = len(upazilas)

        # Distribute randomly across upazilas in this zilla
        u_indices = np.random.randint(0, u_count, size=p_count)
        
        upazila_names = [upazilas[i][0] for i in u_indices]
        thana_names = [upazilas[i][1] for i in u_indices]
        zilla_names = [profile["zilla"]] * p_count

        # Coordinates jittered around base_lat and base_lon
        lat_jitter = np.random.normal(0, 0.08, size=p_count)
        lon_jitter = np.random.normal(0, 0.09, size=p_count)
        lats = np.round(profile["base_lat"] + lat_jitter, 5)
        lons = np.round(profile["base_lon"] + lon_jitter, 5)

        # Elevation: low-lying areas in haor/chars have lower elevation
        elev_jitter = np.random.normal(0, 2.5, size=p_count)
        elevations = np.clip(np.round(profile["base_elevation"] + elev_jitter, 1), 1.0, 50.0)

        # Flood risk level (0-1): heavily influenced by flood_propensity and elevation
        risk_noise = np.random.normal(0, 0.12, size=p_count)
        # Higher elevation reduces risk; haor flood propensity increases risk
        flood_risks = profile["flood_propensity"] * 0.75 + (30.0 - elevations) / 60.0 * 0.35 + risk_noise
        flood_risks = np.clip(np.round(flood_risks, 3), 0.05, 0.99)

        # Accessibility score: lower when flood risk is high, or elevation is low
        access_noise = np.random.normal(0, 0.08, size=p_count)
        accessibility = 1.0 - (flood_risks * 0.65) + access_noise
        accessibility = np.clip(np.round(accessibility, 2), 0.10, 0.98)

        # Population per area node (between 600 and 12,000)
        populations = np.random.randint(600, 12000, size=p_count)

        # Available shelters nearby (0 to 5)
        shelters_nearby = np.random.choice([0, 1, 2, 3, 4, 5], size=p_count, p=[0.20, 0.35, 0.25, 0.12, 0.06, 0.02])

        # Affected families: derived from population and flood risk level
        family_sizes = np.random.uniform(4.0, 5.2, size=p_count)
        affected_families = np.round((populations / family_sizes) * flood_risks).astype(int)

        # Area names
        area_names = generate_area_names(p_count)
        area_ids = [f"BD-{i:06d}" for i in range(current_id, current_id + p_count)]
        current_id += p_count

        sub_df = pd.DataFrame({
            "area_id": area_ids,
            "area_name": area_names,
            "thana": thana_names,
            "upazila": upazila_names,
            "zilla": zilla_names,
            "latitude": lats,
            "longitude": lons,
            "population": populations,
            "flood_risk_level": flood_risks,
            "elevation": elevations,
            "accessibility_score": accessibility,
            "available_shelters_nearby": shelters_nearby,
            "affected_families": affected_families
        })
        dfs.append(sub_df)

    df = pd.concat(dfs, ignore_index=True)
    # Shuffle randomly to ensure uniform distribution in splits
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"[*] Generated DataFrame with shape: {df.shape}")

    # Split: 40% train, 30% test, 30% validate
    train_df = df.iloc[:TRAIN_COUNT].reset_index(drop=True)
    test_df = df.iloc[TRAIN_COUNT:TRAIN_COUNT + TEST_COUNT].reset_index(drop=True)
    validate_df = df.iloc[TRAIN_COUNT + TEST_COUNT:].reset_index(drop=True)

    print(f"[*] Saving {DATASET_CSV} ({len(df)} rows)...")
    df.to_csv(DATASET_CSV, index=False)

    print(f"[*] Saving {TRAIN_CSV} ({len(train_df)} rows - 40%)...")
    train_df.to_csv(TRAIN_CSV, index=False)

    print(f"[*] Saving {TEST_CSV} ({len(test_df)} rows - 30%)...")
    test_df.to_csv(TEST_CSV, index=False)

    print(f"[*] Saving {VALIDATE_CSV} ({len(validate_df)} rows - 30%)...")
    validate_df.to_csv(VALIDATE_CSV, index=False)

    # Build optimized in-memory lookup cache and pickle it for instant app startup
    print(f"[*] Building pre-indexed pickle cache...")
    build_and_save_pickle(df)

    elapsed = time.time() - start_time
    print(f"[SUCCESS] Data generation and splitting successfully completed in {elapsed:.2f} seconds!")
    return df

def build_and_save_pickle(df):
    """Caches pre-indexed hierarchy, summary statistics, and regional candidate nodes."""
    # Precompute administrative hierarchy
    hierarchy = {}
    for zilla in df['zilla'].unique():
        z_df = df[df['zilla'] == zilla]
        upazilas = {}
        for upazila in z_df['upazila'].unique():
            u_df = z_df[z_df['upazila'] == upazila]
            thanas = list(u_df['thana'].unique())
            upazilas[upazila] = thanas
        hierarchy[zilla] = upazilas

    # Overall summary metrics
    summary = {
        "total_records": len(df),
        "total_population": int(df['population'].sum()),
        "total_affected_families": int(df['affected_families'].sum()),
        "avg_flood_risk": round(float(df['flood_risk_level'].mean()), 3),
        "high_risk_areas": int((df['flood_risk_level'] >= 0.70).sum()),
        "medium_risk_areas": int(((df['flood_risk_level'] >= 0.40) & (df['flood_risk_level'] < 0.70)).sum()),
        "low_risk_areas": int((df['flood_risk_level'] < 0.40).sum()),
        "total_shelters_recorded": int(df['available_shelters_nearby'].sum()),
        "zillas_count": int(df['zilla'].nunique()),
        "upazilas_count": int(df['upazila'].nunique())
    }

    # Pre-select representative spatial nodes per Upazila for fast algorithmic operations
    # (keeps top affected nodes and candidate shelters for sub-second graph building)
    representative_samples = df.groupby('upazila', group_keys=False).apply(
        lambda g: g.head(60)
    ).reset_index(drop=True)

    cache_data = {
        "hierarchy": hierarchy,
        "summary": summary,
        "representative_nodes": representative_samples,
        "zillas_list": list(hierarchy.keys()),
        "generated_timestamp": time.time()
    }

    with open(PICKLE_FILE, 'wb') as f:
        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[OK] Saved fast pickle cache to {PICKLE_FILE}")

def load_processed_data():
    """Loads cached processed data if available, otherwise generates it."""
    if os.path.exists(PICKLE_FILE) and os.path.exists(DATASET_CSV):
        try:
            with open(PICKLE_FILE, 'rb') as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            print(f"[!] Warning: Failed to load pickle ({e}). Rebuilding cache from existing CSV...")
            try:
                df = pd.read_csv(DATASET_CSV)
                build_and_save_pickle(df)
                with open(PICKLE_FILE, 'rb') as f:
                    return pickle.load(f)
            except Exception as inner_e:
                print(f"[!] Warning: Failed to rebuild from CSV ({inner_e}). Regenerating full dataset...")
    
    create_dataset()
    with open(PICKLE_FILE, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    create_dataset()
