# Data

The raw files are large, so they are not in Git. Put these three
[Inside Airbnb](https://insideairbnb.com/get-the-data/) `listings.csv.gz` (detailed listings) files here, unzipped:

| City   | Scrape     | File name used here          | Rows  | With price ≤ 500 |
|--------|------------|------------------------------|------:|-----------------:|
| Berlin | Sep 2025   | `listings_detail_berlin.csv` | 14,274 | 9,133 |
| Paris  | Mar 2026   | `listings_pari26.csv`        | 38,075 | 30,971 |
| NYC    | Jun 2026   | `listings_nyc26.csv`         | 30,259 | 19,148 |

`berlin.csv`, `paris.csv` and `nyc.csv` are accepted too (see `src/config.py`).

The 2025 Paris and New York scrapes have an empty `price` column, so they can't be used.
Prices are in local currency (EUR for Berlin/Paris, USD for NYC).
