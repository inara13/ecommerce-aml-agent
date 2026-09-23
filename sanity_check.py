"""
Sanity check for the generated data.
Cleans the raw files, then summarizes key signals by seller_group so we can confirm:
  - easy structurers are obvious, hard ones blend in
  - decoys trip naive rules
  - no structurer ever takes a payout of 10k or more
Uses labels ONLY to verify the design. No detection approach ever sees them.
"""
import os
import pandas as pd

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
END = pd.Timestamp("2026-08-30")

orders = pd.read_csv(os.path.join(RAW, "orders.csv"))
payouts = pd.read_csv(os.path.join(RAW, "payouts.csv"))
sellers = pd.read_csv(os.path.join(RAW, "sellers.csv"))
buyers = pd.read_csv(os.path.join(RAW, "buyers.csv"))
labels = pd.read_csv(os.path.join(RAW, "labels.csv"))

print("=== Dirty data found before cleaning ===")
print(f"orders : dup rows {orders.duplicated().sum()}, null price {orders.item_price.isna().sum()}, "
      f"negative price {(orders.item_price < 0).sum()}, "
      f"odd category {(~orders.category.isin(orders.category.str.strip().str.lower().unique()) | (orders.category != orders.category.str.strip().str.lower())).sum()}")
print(f"payouts: dup rows {payouts.duplicated().sum()}, null amount {payouts.amount.isna().sum()}")

# ---- clean ----
orders = orders.drop_duplicates()
orders["order_ts"] = pd.to_datetime(orders["order_ts"], format="mixed")
orders["category"] = orders["category"].str.strip().str.lower()
orders = orders[orders.item_price.notna() & (orders.item_price > 0)]
payouts = payouts.drop_duplicates()
payouts["payout_ts"] = pd.to_datetime(payouts["payout_ts"], format="mixed")
payouts = payouts[payouts.amount.notna()]
paid = payouts[payouts.payout_status == "paid"]

# ---- per seller signals ----
near = paid[(paid.amount >= 8000) & (paid.amount < 10000)]


def max_in_7d(df):
    t = df.sort_values("payout_ts").payout_ts.reset_index(drop=True)
    best, j = 0, 0
    for i in range(len(t)):
        while t[i] - t[j] > pd.Timedelta(days=7):
            j += 1
        best = max(best, i - j + 1)
    return best


f = pd.DataFrame({"seller_id": sellers.seller_id})
f = f.merge(near.groupby("seller_id").size().rename("payouts_8k_to_10k"), on="seller_id", how="left")
f = f.merge(near.groupby("seller_id").apply(max_in_7d, include_groups=False).rename("max_near_in_7d"),
            on="seller_id", how="left")
f = f.merge(paid.groupby("seller_id").amount.max().rename("max_payout"), on="seller_id", how="left")
f = f.merge(paid.groupby("seller_id").bank_account_id.nunique().rename("banks_used"), on="seller_id", how="left")
f = f.merge((paid.payout_type == "on_demand").groupby(paid.seller_id).sum().rename("on_demand"),
            on="seller_id", how="left")
comp = orders[orders.order_status == "completed"]
by_buyer = comp.groupby(["seller_id", "buyer_id"]).item_price.sum()
top3 = by_buyer.groupby(level=0).apply(lambda s: s.nlargest(3).sum() / s.sum()).rename("top3_buyer_share")
f = f.merge(top3, on="seller_id", how="left")
sellers["account_age_days"] = (END - pd.to_datetime(sellers.signup_date)).dt.days
f = f.merge(sellers[["seller_id", "account_age_days"]], on="seller_id")
f = f.fillna(0).merge(labels, on="seller_id")

order = ["easy", "medium", "hard", "decoy_highvalue", "decoy_business", "decoy_seasonal", "normal"]
summary = f.groupby("seller_group")[["payouts_8k_to_10k", "max_near_in_7d", "max_payout", "banks_used",
                                      "on_demand", "top3_buyer_share", "account_age_days"]].median().loc[order]
pd.set_option("display.width", 200)
print("\n=== Median signals by group ===")
print(summary.round(2).to_string())

print("\n=== Structurers with any payout of 10k or more (should be 0) ===")
print(int(((f.is_structuring == 1) & (f.max_payout >= 10000)).sum()))

print("\n=== Naive rule test: flag if 3+ payouts between 8k and 10k ===")
f["naive_flag"] = f.payouts_8k_to_10k >= 3
print(f.groupby("seller_group").naive_flag.agg(["sum", "count"]).loc[order].to_string())
tp = ((f.naive_flag) & (f.is_structuring == 1)).sum()
fp = ((f.naive_flag) & (f.is_structuring == 0)).sum()
print(f"recall {tp / f.is_structuring.sum():.0%}   alerts {tp + fp}   alert FP rate {fp / max(1, tp + fp):.0%}")

print("\n=== Overlap check: can one simple signal isolate the hard tier? ===")
for name, cond in [("3+ bank accounts used", f.banks_used >= 3),
                   ("8+ on-demand payouts", f.on_demand >= 8)]:
    hit = f[cond]
    print(f"{name:24s} hard caught {int((hit.seller_group == 'hard').sum())}/20, "
          f"innocent sellers also flagged {int((hit.is_structuring == 0).sum())}")
