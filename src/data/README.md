# Data Directory

This directory contains all data used in the project.

## Structure

- `raw/`: Original, immutable data
  - `transactions/`: E-commerce transaction data
  - `behavior/`: Customer behavior data (clickstreams)
  - `surveys/`: Market research survey data
  - `competitors/`: Competitor pricing information

- `processed/`: Cleaned and processed data
  - `features/`: Extracted features for modeling
  - `train/`: Training datasets
  - `validation/`: Validation datasets
  - `test/`: Test datasets

- `external/`: Data from third-party sources
  - `market_reports/`: Industry market reports
  - `economic_indicators/`: Economic data that might affect pricing