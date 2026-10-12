# Predicting Airbnb Listing Price Across Different Cities

UE24CS352A Machine Learning mini project. We reproduce the CS229 report "Predicting Airbnb Listing Price Across Different Cities". The project predicts Airbnb listing prices for Berlin, Paris and NYC with five models per city, and then tests cross-city transfer (NYC to Berlin, Paris to Berlin, NYC+Paris to Berlin).

## Files
- notebooks/airbnb_price_prediction.ipynb: the full project (single-city baselines for Berlin, Paris and NYC, then the "Cross city Prediction" section)
- results/cross_city_results.csv: cross-city results table
- docs/: project report (PDF)

## How to run
1. Put these three data files in Google Drive at MyDrive/airbnb_project/data/raw/ (Inside Airbnb listing files):
   - listings_detail_berlin.csv
   - listings_pari26.csv
   - listings_nyc26.csv
2. Open notebooks/airbnb_price_prediction.ipynb in Google Colab. A GPU runtime (T4) is optional but faster.
3. Run the cells from top to bottom (Runtime > Run all). The first cell mounts Google Drive and asks for permission.
4. The "Cross city Prediction" section at the end can also be run on its own after a restart. Its first cell mounts Drive and it reads the three CSV files from Drive.

## Results
Single-city (log price, test set): XGBoost was the best model in all three cities (R2 about 0.71 for Berlin, 0.72 for Paris and 0.81 for NYC). Single-city numbers vary slightly from run to run.

Cross-city (neural network, tested on all of Berlin, which is never used for training):

| Trained on | Features | Berlin MSE | Berlin R2 |
|---|---|---|---|
| NYC | 258 | 2850.29 | -8243.98 |
| Paris | 234 | 30.87 | -88.30 |
| NYC + Paris | 271 | 0.315 | 0.088 |

Training on several cities transfers much better than training on a single city.

## Team
- Sakaray Vishal - PES1UG24AM241
- Sreedharala Praneetha - PES1UG24AM281
