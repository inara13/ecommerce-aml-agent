# ResellHub: AML Structuring Detection Benchmark

Synthetic data for comparing three ways to detect **payout structuring** on an eBay and Depop style marketplace:

1. **Data Analyst Agent:** an LLM studies the dev half and writes SQL rules; the rules make the decision
2. **LLM decider:** an LLM judges each seller directly
3. **Jev:** TypeSafe's decision model returns a structuring probability per seller

All data is synthetic. ResellHub is a made-up platform.

## The typology

Platform rule for this project: **any single payout of $10,000 or more triggers enhanced review and reporting.**
Structuring means deliberately splitting payouts to stay under that line. Here, structurers push fake sales through colluding buyer accounts, then cash out with on-demand payouts under $10k.

## Files, in `data/raw/`

| File | Rows | Visible to approaches? |
|---|---|---|
| sellers.csv | 2,000 | Yes |
| buyers.csv | ~14,800 | Yes |
| orders.csv | ~74,000 | Yes |
| payouts.csv | ~18,450 | Yes |
| seller_split.csv | 2,000 | Yes, dev or test |
| **labels.csv** | 2,000 | **No. Answer key, used only for scoring** |

Period: 2026-06-01 to 2026-08-29, 90 days. Platform fee: 12%.

### Columns

- **sellers:** seller_id, signup_date, account_type, country, primary_category, kyc_verified, num_bank_accounts
- **buyers:** buyer_id, signup_date, country
- **orders:** order_id, seller_id, buyer_id, order_ts, category, item_price, shipping_fee, payment_method, buyer_country, order_status
- **payouts:** payout_id, seller_id, payout_ts, amount, payout_type, bank_account_id, payout_status
- **labels:** seller_id, is_structuring, seller_group
- **seller_split:** seller_id, split

## Seller groups

| Group | Count | Structuring | Behavior |
|---|---|---|---|
| easy | 20 | Yes | New account, 3 to 6 on-demand payouts of $9,000 to $9,999 inside 7 days, tiny buyer ring |
| medium | 20 | Yes | $7,500 to $9,900 payouts spread 3 to 8 days apart, bigger ring, some real sales |
| hard | 20 | Yes | $4,000 to $8,000 payouts across 2 to 3 bank accounts, older account, large ring mixed with lots of real sales |
| decoy_highvalue | 40 | No | Legit watch, jewelry and electronics sellers with real $1.5k to $15k sales |
| decoy_business | 30 | No | Businesses with steady weekly payouts of about $8k to $9.7k for the whole period |
| decoy_seasonal | 30 | No | Small sellers with one legit big sales burst, half cash out on demand |
| normal | 1,840 | No | Everyday sellers; about 12% like on-demand payouts, a few use 3 bank accounts |

Split: stratified 50/50, so each half has **30 structurers and 50 decoys**.

## Dirty data, about 2% on purpose

- orders: duplicate rows, null item_price, negative item_price, uppercase or padded category, mixed timestamp formats
- payouts: duplicate rows, null amount, mixed timestamp formats

## Leakage protections

- Seller and buyer IDs are randomly shuffled, so IDs carry no group information
- Order and payout IDs follow time order only
- labels.csv must never be joined into features, prompts or Jev state

## Reproduce

```
python generate_data.py     # seed 42, writes data/raw/
python sanity_check.py      # cleans data and verifies the design
```

Baseline from the sanity check: a naive rule "3+ payouts between $8k and $10k" gets **67% recall** with a **44% alert false positive rate**. It misses every hard structurer and flags every steady business.
