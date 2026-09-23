"""
ResellHub synthetic marketplace data generator
AML typology: STRUCTURING of seller payouts

Platform rule assumed for this project:
    Any single payout of $10,000 or more triggers enhanced review and reporting.
    Structuring = deliberately splitting payouts to stay under that line.

Outputs, written to ./data/raw/ :
    sellers.csv, buyers.csv, orders.csv, payouts.csv   -> visible to every approach
    labels.csv                                          -> ANSWER KEY, never shown to any approach
    seller_split.csv                                    -> dev / test assignment, stratified by group

All data is synthetic. Run:  python generate_data.py
"""

import os
import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

START = pd.Timestamp("2026-06-01")          # a Monday
DAYS = 90
END = START + pd.Timedelta(days=DAYS)
FEE = 0.12                                    # platform fee, seller receives 88% of item price
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")

# ---------------------------------------------------------------------------
# Marketplace shape: eBay and Depop style mix
# ---------------------------------------------------------------------------
CATEGORIES = {
    #  name                        share  price_lo  price_hi  mean_orders_90d
    "fashion_vintage":          (0.45,    15,      150,     45),
    "sneakers_streetwear":      (0.20,    80,      600,     30),
    "electronics":              (0.15,   100,     2000,     20),
    "cards_collectibles":       (0.10,    20,     5000,     30),
    "watches_jewelry_luxury":   (0.10,   500,    15000,      6),
}
CAT_NAMES = list(CATEGORIES)
CAT_SHARES = np.array([CATEGORIES[c][0] for c in CAT_NAMES])
HIGH_VALUE_CATS = ["electronics", "cards_collectibles", "watches_jewelry_luxury"]

COUNTRIES = ["US", "GB", "CA", "DE", "AU", "FR"]
COUNTRY_P = [0.80, 0.06, 0.05, 0.04, 0.03, 0.02]

HOUR_P = np.array([1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 6, 7, 7, 6, 6, 6, 7, 8, 9, 9, 8, 5, 3], float)
HOUR_P /= HOUR_P.sum()

GROUP_SIZES = {
    "easy": 20,
    "medium": 20,
    "hard": 20,
    "decoy_highvalue": 40,
    "decoy_business": 30,
    "decoy_seasonal": 30,
    "normal": 1840,
}
STRUCTURING_GROUPS = {"easy", "medium", "hard"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_bank_counter = [0]


def new_bank_ids(n):
    ids = []
    for _ in range(n):
        _bank_counter[0] += 1
        ids.append(f"BA{_bank_counter[0]:05d}")
    return ids


def sample_price(lo, hi, a=1.3, b=3.0):
    """Skewed toward the low end of the range, like real listings."""
    u = rng.beta(a, b)
    return round(float(lo * (hi / lo) ** u), 2)


def shipping_fee(price):
    return round(min(60.0, float(rng.uniform(4, 12)) + price * 0.01), 2)


def ts_on_day(day_float):
    """Timestamp on a given day offset, with a realistic hour of day."""
    day = int(np.floor(day_float))
    hour = int(rng.choice(24, p=HOUR_P))
    minute = int(rng.integers(0, 60))
    second = int(rng.integers(0, 60))
    return START + pd.Timedelta(days=day, hours=hour, minutes=minute, seconds=second)


def payment_method(ring=False):
    if ring:
        return str(rng.choice(["card", "wallet", "bank"], p=[0.35, 0.45, 0.20]))
    return str(rng.choice(["card", "wallet", "bank"], p=[0.70, 0.25, 0.05]))


def payout_status():
    return str(rng.choice(["paid", "failed", "reversed"], p=[0.985, 0.010, 0.005]))


# ---------------------------------------------------------------------------
# Buyers
# ---------------------------------------------------------------------------
N_POOL_BUYERS = 14200
buyers = []   # dicts with internal key
for i in range(N_POOL_BUYERS):
    age_days = int(rng.integers(1, 365 * 8))
    buyers.append({
        "key": i,
        "signup_date": (START - pd.Timedelta(days=age_days)).date(),
        "country": str(rng.choice(COUNTRIES, p=COUNTRY_P)),
    })
# heavy tailed popularity so some buyers are repeat customers
pool_weights = rng.pareto(1.5, N_POOL_BUYERS) + 1
pool_weights /= pool_weights.sum()


def pick_pool_buyer():
    return int(rng.choice(N_POOL_BUYERS, p=pool_weights))


def make_ring(size, activity_start_day, aged_share=0.0):
    """Colluding buyer accounts used to push fake sales to a structurer."""
    keys = []
    for _ in range(size):
        k = len(buyers)
        if rng.random() < aged_share:
            age = int(rng.integers(365, 365 * 4))                       # bought or aged account
            signup = START + pd.Timedelta(days=activity_start_day) - pd.Timedelta(days=age)
        else:
            lead = int(rng.integers(1, 30))                              # freshly created
            signup = START + pd.Timedelta(days=activity_start_day - lead)
        buyers.append({"key": k, "signup_date": signup.date(),
                       "country": str(rng.choice(COUNTRIES, p=[0.7, 0.06, 0.06, 0.06, 0.06, 0.06]))})
        keys.append(k)
    return keys


# ---------------------------------------------------------------------------
# Order and payout builders
# ---------------------------------------------------------------------------
orders = []    # dicts
payouts = []   # dicts


def add_order(s_idx, buyer_key, ts, cat, price, seller_country, ring=False,
              status=None, funded=False):
    if status is None:
        status = str(rng.choice(["completed", "refunded", "cancelled"], p=[0.95, 0.03, 0.02]))
    b_country = seller_country if rng.random() < 0.9 else str(rng.choice(COUNTRIES, p=COUNTRY_P))
    rec = {
        "s_idx": s_idx, "buyer_key": buyer_key, "order_ts": ts, "category": cat,
        "item_price": round(price, 2), "shipping_fee": shipping_fee(price),
        "payment_method": payment_method(ring), "buyer_country": b_country,
        "order_status": status, "_on_demand_funded": funded,
    }
    orders.append(rec)
    return rec


def add_payout(s_idx, ts, amount, ptype, bank, status=None):
    payouts.append({
        "s_idx": s_idx, "payout_ts": ts, "amount": round(float(amount), 2),
        "payout_type": ptype, "bank_account_id": bank,
        "payout_status": status if status else payout_status(),
    })


def organic_orders(s_idx, seller, mean_orders, lo, hi, day_lo=0, day_hi=DAYS, a=1.3, b=3.0):
    n = int(rng.negative_binomial(2, 2 / (2 + mean_orders))) if mean_orders > 0 else 0
    first_day = max(day_lo, seller["_signup_offset"] + 1)
    if first_day >= day_hi:
        return
    for _ in range(n):
        d = rng.uniform(first_day, day_hi)
        add_order(s_idx, pick_pool_buyer(), ts_on_day(d), seller["primary_category"],
                  sample_price(lo, hi, a, b), seller["country"])


def scheduled_payouts(s_idx, seller, on_demand_prob=0.0):
    """Weekly automatic payouts for completed sales not already paid on demand."""
    weekly = {}
    for o in orders_by_seller.get(s_idx, []):
        if o["order_status"] != "completed" or o["_on_demand_funded"]:
            continue
        w = (o["order_ts"] - START).days // 7
        weekly[w] = weekly.get(w, 0.0) + o["item_price"] * (1 - FEE)
    banks = seller["_banks"]
    for w, net in sorted(weekly.items()):
        pay_ts = START + pd.Timedelta(days=7 * (w + 1), hours=9)
        if pay_ts >= END or net <= 0:
            continue
        bank = banks[0] if (len(banks) == 1 or rng.random() < 0.8) else str(rng.choice(banks[1:]))
        if on_demand_prob and rng.random() < on_demand_prob:
            # seller pulled the money early, sometime late in the sale week
            od_day = 7 * w + rng.uniform(3, 6.99)
            add_payout(s_idx, ts_on_day(od_day), net, "on_demand", bank)
        else:
            add_payout(s_idx, pay_ts, net, "scheduled", bank)


def funded_on_demand_payout(s_idx, seller, pay_day, amount, ring, n_orders_range,
                            bank, lead_days=(0, 2)):
    """Fake sales from a buyer ring, then an on-demand cash-out of exactly that money."""
    gross = amount / (1 - FEE)
    m = int(rng.integers(n_orders_range[0], n_orders_range[1] + 1))
    shares = rng.dirichlet(np.ones(m) * 3)
    earliest = seller["_signup_offset"] + 1
    for sh in shares:
        od = max(earliest, pay_day - rng.uniform(*lead_days))
        buyer = int(rng.choice(ring))
        add_order(s_idx, buyer, ts_on_day(od), seller["primary_category"], gross * sh,
                  seller["country"], ring=True, status="completed", funded=True)
    pay_ts = ts_on_day(pay_day)
    add_payout(s_idx, pay_ts, amount, "on_demand", bank, status="paid")


# ---------------------------------------------------------------------------
# Sellers
# ---------------------------------------------------------------------------
sellers = []
for group, n in GROUP_SIZES.items():
    for _ in range(n):
        sellers.append({"group": group})

orders_by_seller = {}


def index_orders():
    orders_by_seller.clear()
    for o in orders:
        orders_by_seller.setdefault(o["s_idx"], []).append(o)


def set_profile(s, age_lo, age_hi, account_type, cat, kyc_p, n_banks, country=None):
    age = int(rng.integers(age_lo, age_hi + 1))
    s["signup_date"] = (END - pd.Timedelta(days=age)).date()
    s["_signup_offset"] = DAYS - age              # day offset of signup vs START, can be negative
    s["account_type"] = account_type
    s["primary_category"] = cat
    s["kyc_verified"] = bool(rng.random() < kyc_p)
    s["num_bank_accounts"] = n_banks
    s["_banks"] = new_bank_ids(n_banks)
    s["country"] = country if country else str(rng.choice(COUNTRIES, p=COUNTRY_P))


structurer_plans = []   # (s_idx, plan) handled after organic payouts are computed
seasonal_plans = []
business_idx = []

for idx, s in enumerate(sellers):
    g = s["group"]

    if g == "normal":
        cat = str(rng.choice(CAT_NAMES, p=CAT_SHARES))
        acct = "business" if rng.random() < 0.08 else "individual"
        n_banks = int(rng.choice([1, 2, 3], p=[0.85, 0.12, 0.03]))
        set_profile(s, 20, 365 * 7, acct, cat, 0.92, n_banks)
        lo, hi, mean = CATEGORIES[cat][1], CATEGORIES[cat][2], CATEGORIES[cat][3]
        organic_orders(idx, s, mean, lo, hi)
        # about 12% of ordinary sellers like to cash out early, some very often
        s["_on_demand_prob"] = float(rng.uniform(0.2, 0.8)) if rng.random() < 0.12 else 0.0

    elif g == "decoy_highvalue":
        cat = str(rng.choice(["watches_jewelry_luxury", "electronics", "cards_collectibles"],
                             p=[0.7, 0.15, 0.15]))
        set_profile(s, 365, 365 * 8, "individual" if rng.random() < 0.7 else "business",
                    cat, 1.0, 1 if rng.random() < 0.8 else 2, country="US")
        organic_orders(idx, s, 14, 1500, 15000, a=1.5, b=2.5)
        s["_on_demand_prob"] = 0.5

    elif g == "decoy_business":
        cat = str(rng.choice(["electronics", "sneakers_streetwear", "watches_jewelry_luxury"],
                             p=[0.5, 0.3, 0.2]))
        set_profile(s, 365, 365 * 6, "business", cat, 1.0, 1 if rng.random() < 0.6 else 2,
                    country="US")
        business_idx.append(idx)
        s["_on_demand_prob"] = 0.0
        # every week the business turns over roughly the same amount, just under 10k net
        base = rng.uniform(8200, 9300)
        lo = {"electronics": 200, "sneakers_streetwear": 150, "watches_jewelry_luxury": 900}[cat]
        hi = {"electronics": 1500, "sneakers_streetwear": 600, "watches_jewelry_luxury": 4000}[cat]
        for w in range(DAYS // 7):
            target_gross = float(np.clip(base + rng.normal(0, 250), 7800, 9700)) / (1 - FEE)
            total = 0.0
            while total < target_gross:
                p = sample_price(lo, hi)
                remaining = target_gross - total
                if remaining < lo * 1.5:
                    p = remaining
                d = 7 * w + rng.uniform(0, 6.99)
                add_order(idx, pick_pool_buyer(), ts_on_day(d), cat, p, s["country"],
                          status="completed")
                total += p

    elif g == "decoy_seasonal":
        cat = str(rng.choice(["fashion_vintage", "cards_collectibles", "sneakers_streetwear"],
                             p=[0.4, 0.35, 0.25]))
        set_profile(s, 365, 365 * 6, "individual", cat, 1.0, 1, country="US")
        lo, hi = CATEGORIES[cat][1], CATEGORIES[cat][2]
        organic_orders(idx, s, 20, lo, hi)
        s["_on_demand_prob"] = 0.0
        seasonal_plans.append(idx)

    else:  # structurers
        # structurers keep their genuine side sales modest so no weekly payout reaches 10k
        if g == "easy":
            cat = str(rng.choice(HIGH_VALUE_CATS, p=[0.4, 0.2, 0.4]))
            set_profile(s, 12, 60, "individual", cat, 0.6, 1, country="US")
            organic_orders(idx, s, 3, CATEGORIES[cat][1], min(CATEGORIES[cat][2], 1200))
        elif g == "medium":
            cat = str(rng.choice(HIGH_VALUE_CATS, p=[0.4, 0.2, 0.4]))
            set_profile(s, 30, 200, "individual", cat, 0.8, int(rng.integers(1, 3)), country="US")
            organic_orders(idx, s, 15, CATEGORIES[cat][1], min(CATEGORIES[cat][2], 1200))
        else:  # hard
            cat = str(rng.choice(["fashion_vintage"] + HIGH_VALUE_CATS, p=[0.4, 0.2, 0.2, 0.2]))
            set_profile(s, 150, 900, "individual", cat, 1.0, int(rng.integers(2, 4)))
            organic_orders(idx, s, 40, CATEGORIES[cat][1], min(CATEGORIES[cat][2], 900))
        s["_on_demand_prob"] = 0.0
        structurer_plans.append(idx)

# ---- seasonal bursts: one legit big sales event, half cash out on demand ----
for idx in seasonal_plans:
    s = sellers[idx]
    start = float(rng.uniform(10, 75))
    dur = float(rng.uniform(3, 6))
    n = int(rng.integers(15, 41))
    total = float(rng.uniform(18000, 35000))
    shares = rng.dirichlet(np.ones(n) * 2)
    on_demand_mode = rng.random() < 0.5
    burst = []
    for sh in shares:
        d = start + rng.uniform(0, dur)
        burst.append(add_order(idx, pick_pool_buyer(), ts_on_day(d), s["primary_category"],
                               total * sh, s["country"], status="completed",
                               funded=on_demand_mode))
    if on_demand_mode:
        by_day = {}
        for o in burst:
            day = (o["order_ts"] - START).days
            by_day[day] = by_day.get(day, 0.0) + o["item_price"] * (1 - FEE)
        for day, net in sorted(by_day.items()):
            if net > 500:
                add_payout(idx, ts_on_day(day + rng.uniform(0.3, 0.9)), net, "on_demand",
                           s["_banks"][0], status="paid")

# ---- scheduled payouts for everyone's organic sales ----
index_orders()
for idx, s in enumerate(sellers):
    scheduled_payouts(idx, s, s.get("_on_demand_prob", 0.0))

# ---- structuring activity: fake sales from rings, then on-demand payouts under 10k ----
for idx in structurer_plans:
    s = sellers[idx]
    g = s["group"]
    first_active = max(s["_signup_offset"] + 2, 2)
    banks = s["_banks"]

    if g == "easy":
        ring = make_ring(int(rng.integers(3, 7)), first_active)
        n_bursts = int(rng.integers(1, 3))
        for _ in range(n_bursts):
            burst_start = float(rng.uniform(first_active, 82))
            k = int(rng.integers(3, 7))
            days = np.sort(burst_start + rng.uniform(0, 6.5, k))
            for d in days:
                amt = float(rng.uniform(9000, 9999)) + float(rng.integers(0, 100)) / 100
                funded_on_demand_payout(idx, s, d, amt, ring, (1, 3), banks[0])

    elif g == "medium":
        ring = make_ring(int(rng.integers(5, 11)), first_active, aged_share=0.2)
        n = int(rng.integers(5, 10))
        d = float(rng.uniform(first_active, first_active + 10))
        for i in range(n):
            if d >= DAYS - 1:
                break
            amt = float(rng.uniform(7500, 9900)) + float(rng.integers(0, 100)) / 100
            bank = banks[i % len(banks)]
            funded_on_demand_payout(idx, s, d, amt, ring, (1, 3), bank)
            d += float(rng.uniform(3, 8))

    else:  # hard
        ring = make_ring(int(rng.integers(12, 26)), first_active, aged_share=0.5)
        n = int(rng.integers(8, 15))
        d = float(rng.uniform(first_active, first_active + 6))
        for i in range(n):
            if d >= DAYS - 1:
                break
            amt = float(rng.uniform(4000, 8000)) + float(rng.integers(0, 100)) / 100
            bank = banks[int(rng.integers(0, len(banks)))]
            funded_on_demand_payout(idx, s, d, amt, ring, (2, 6), bank, lead_days=(1, 5))
            d += float(rng.uniform(3, 10))

# ---------------------------------------------------------------------------
# Assign shuffled IDs so no ID carries information about the group
# ---------------------------------------------------------------------------
seller_perm = rng.permutation(len(sellers))
seller_id_of = {idx: f"S{seller_perm[idx] + 1:05d}" for idx in range(len(sellers))}

buyer_perm = rng.permutation(len(buyers))
buyer_id_of = {b["key"]: f"B{buyer_perm[i] + 1:05d}" for i, b in enumerate(buyers)}

sellers_df = pd.DataFrame([{
    "seller_id": seller_id_of[i],
    "signup_date": s["signup_date"],
    "account_type": s["account_type"],
    "country": s["country"],
    "primary_category": s["primary_category"],
    "kyc_verified": s["kyc_verified"],
    "num_bank_accounts": s["num_bank_accounts"],
} for i, s in enumerate(sellers)]).sort_values("seller_id").reset_index(drop=True)

labels_df = pd.DataFrame([{
    "seller_id": seller_id_of[i],
    "is_structuring": int(s["group"] in STRUCTURING_GROUPS),
    "seller_group": s["group"],
} for i, s in enumerate(sellers)]).sort_values("seller_id").reset_index(drop=True)

buyers_df = pd.DataFrame([{
    "buyer_id": buyer_id_of[b["key"]],
    "signup_date": b["signup_date"],
    "country": b["country"],
} for b in buyers]).sort_values("buyer_id").reset_index(drop=True)

orders_df = pd.DataFrame(orders)
orders_df = orders_df[orders_df["order_ts"] < END].sort_values("order_ts").reset_index(drop=True)
orders_df.insert(0, "order_id", [f"O{i + 1:06d}" for i in range(len(orders_df))])
orders_df["seller_id"] = orders_df["s_idx"].map(seller_id_of)
orders_df["buyer_id"] = orders_df["buyer_key"].map(buyer_id_of)
orders_df = orders_df[["order_id", "seller_id", "buyer_id", "order_ts", "category", "item_price",
                       "shipping_fee", "payment_method", "buyer_country", "order_status"]]

payouts_df = pd.DataFrame(payouts)
payouts_df = payouts_df[payouts_df["payout_ts"] < END].sort_values("payout_ts").reset_index(drop=True)
payouts_df.insert(0, "payout_id", [f"P{i + 1:06d}" for i in range(len(payouts_df))])
payouts_df["seller_id"] = payouts_df["s_idx"].map(seller_id_of)
payouts_df = payouts_df[["payout_id", "seller_id", "payout_ts", "amount", "payout_type",
                         "bank_account_id", "payout_status"]]

# ---------------------------------------------------------------------------
# Dev / test split, stratified by group so each half has 30 structurers and 50 decoys
# ---------------------------------------------------------------------------
split_rows = []
for grp, sub in labels_df.groupby("seller_group"):
    ids = sub["seller_id"].to_numpy().copy()
    rng.shuffle(ids)
    half = len(ids) // 2
    split_rows += [{"seller_id": x, "split": "dev"} for x in ids[:half]]
    split_rows += [{"seller_id": x, "split": "test"} for x in ids[half:]]
split_df = pd.DataFrame(split_rows).sort_values("seller_id").reset_index(drop=True)

# ---------------------------------------------------------------------------
# Inject about 2% dirty rows so there is a real cleaning step
# ---------------------------------------------------------------------------
TS_FMT = "%Y-%m-%d %H:%M:%S"
ALT_FMT = "%m/%d/%Y %H:%M"


def dirty_timestamps(df, col, frac):
    out = df[col].dt.strftime(TS_FMT)
    mask = rng.random(len(df)) < frac
    out[mask] = df.loc[mask, col].dt.strftime(ALT_FMT)
    return out


# orders
o = orders_df.copy()
o["order_ts"] = dirty_timestamps(o, "order_ts", 0.005)
null_mask = rng.random(len(o)) < 0.003
o.loc[null_mask, "item_price"] = np.nan
neg_mask = (rng.random(len(o)) < 0.002) & ~null_mask
o.loc[neg_mask, "item_price"] = -o.loc[neg_mask, "item_price"]
case_mask = rng.random(len(o)) < 0.003
o.loc[case_mask, "category"] = o.loc[case_mask, "category"].str.upper() + " "
dups = o.sample(frac=0.008, random_state=SEED)
o = pd.concat([o, dups]).sort_values("order_id", kind="stable").reset_index(drop=True)

# payouts
p = payouts_df.copy()
p["payout_ts"] = dirty_timestamps(p, "payout_ts", 0.005)
pnull = rng.random(len(p)) < 0.003
p.loc[pnull, "amount"] = np.nan
pdups = p.sample(frac=0.008, random_state=SEED + 1)
p = pd.concat([p, pdups]).sort_values("payout_id", kind="stable").reset_index(drop=True)

# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)
sellers_df.to_csv(os.path.join(OUT_DIR, "sellers.csv"), index=False)
buyers_df.to_csv(os.path.join(OUT_DIR, "buyers.csv"), index=False)
o.to_csv(os.path.join(OUT_DIR, "orders.csv"), index=False)
p.to_csv(os.path.join(OUT_DIR, "payouts.csv"), index=False)
labels_df.to_csv(os.path.join(OUT_DIR, "labels.csv"), index=False)
split_df.to_csv(os.path.join(OUT_DIR, "seller_split.csv"), index=False)

print("Wrote to", OUT_DIR)
for name, df in [("sellers", sellers_df), ("buyers", buyers_df), ("orders", o),
                 ("payouts", p), ("labels", labels_df), ("seller_split", split_df)]:
    print(f"  {name:13s} {len(df):>7,} rows  {df.shape[1]} cols")
