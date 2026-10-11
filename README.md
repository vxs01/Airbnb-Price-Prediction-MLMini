# Predicting Airbnb Listing Price Across Different Cities

UE24CS352A Machine Learning Mini Project. Predicts Airbnb listing prices for NYC, Paris and Berlin, and tests cross-city transfer (NYC to Berlin, Paris to Berlin, NYC+Paris to Berlin).

## How to run
1. Put the 3 data CSVs (listings_nyc26.csv, listings_pari26.csv, listings_detail_berlin.csv) in Google Drive at MyDrive/airbnb_project/data/raw/
2. Open notebooks/airbnb_price_prediction.ipynb in Google Colab (GPU preferred).
3. Run the cells in order. The "Cross City Prediction" section is at the end.

## Results
Cross-city (neural network, tested on Berlin, never used for training):

| Experiment | Features | Berlin MSE | Berlin R2 |
|---|---|---|---|
| NYC -> Berlin | 258 | 2850.29 | -8243.98 |
| Paris -> Berlin | 234 | 30.87 | -88.30 |
| NYC+Paris -> Berlin | 271 | 0.315 | 0.088 |

Training on several cities transfers much better than training on a single city.

## Team
Sakaray Vishal - PES1UG24AM241

Sreedharala Praneetha - PES1UG24AM281
