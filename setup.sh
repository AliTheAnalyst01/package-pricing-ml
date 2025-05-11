#!/bin/bash

# Create directories for data if they don't exist
mkdir -p results
mkdir -p experiments

# Copy the latest data files to the deployment
cp results/enhanced_product_pricing_*.csv results/
cp results/enhanced_bundle_pricing_*.csv results/
cp experiments/price_test_*_results.csv experiments/

echo "Setup completed successfully!"