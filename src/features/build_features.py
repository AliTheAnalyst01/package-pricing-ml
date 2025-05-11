"""
Module for feature engineering and preparing data for modeling.
"""
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("feature_engineering")

def get_latest_file(directory, prefix):
    """Get the most recent file with a given prefix in a directory."""
    files = [f for f in os.listdir(directory) if f.startswith(prefix)]
    if not files:
        return None
    return os.path.join(directory, sorted(files)[-1])

def load_data():
    """Load all data sources."""
    logger.info("Loading data...")
    
    # Load product and bundle reference data
    products = pd.read_csv('data/external/products.csv')
    bundles = pd.read_csv('data/external/bundles.csv')
    
    # Load transaction data
    transactions_file = get_latest_file('data/raw/transactions', 'transactions_')
    transactions = pd.read_csv(transactions_file)
    transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])
    
    # Load transaction items
    items_file = get_latest_file('data/raw/transactions', 'transaction_items_')
    items = pd.read_csv(items_file)
    
    # Load behavior data
    behavior_file = get_latest_file('data/raw/behavior', 'behavior_')
    behavior = pd.read_csv(behavior_file)
    behavior['date'] = pd.to_datetime(behavior['date'])
    
    # Load willingness to pay data
    wtp_file = get_latest_file('data/raw/surveys', 'willingness_to_pay_')
    wtp = pd.read_csv(wtp_file)
    
    # Try to load competitor data if available
    try:
        competitor_file = get_latest_file('data/raw/competitors', 'competitor_prices_')
        competitors = pd.read_csv(competitor_file) if competitor_file else None
    except:
        competitors = None
        logger.warning("No competitor data available")
    
    return {
        'products': products,
        'bundles': bundles,
        'transactions': transactions,
        'items': items,
        'behavior': behavior,
        'wtp': wtp,
        'competitors': competitors
    }

def create_product_features(data_dict):
    """Create product-level features for modeling."""
    logger.info("Creating product features...")
    
    products = data_dict['products'].copy()
    items = data_dict['items'].copy()
    behavior = data_dict['behavior'].copy()
    wtp = data_dict['wtp'].copy()
    
    # 1. Calculate sales metrics per product
    product_sales = items.groupby('product_id').agg({
        'transaction_id': 'count',
        'quantity': 'sum',
        'unit_price': 'mean'
    }).reset_index()
    
    product_sales.rename(columns={
        'transaction_id': 'num_transactions',
        'quantity': 'total_quantity_sold',
        'unit_price': 'avg_selling_price'
    }, inplace=True)
    
    # 2. Calculate behavior metrics per product
    product_behavior = behavior.groupby('product_id').agg({
        'page_views': 'sum',
        'time_on_page': 'mean',
        'clicks': 'sum',
        'add_to_cart': 'sum',
        'remove_from_cart': 'sum'
    }).reset_index()
    
    # Calculate engagement metrics
    product_behavior['cart_retention'] = 1 - (product_behavior['remove_from_cart'] / product_behavior['add_to_cart']).fillna(0)
    product_behavior['engagement_score'] = (
        product_behavior['time_on_page'] * 0.3 + 
        product_behavior['clicks'] * 0.3 + 
        product_behavior['cart_retention'] * 0.4
    )
    
    # 3. Willingness to pay data per product
    # Extract WTP columns
    wtp_columns = [col for col in wtp.columns if col.startswith('wtp_')]
    
    # Create a clean format with product_id as a column rather than in column name
    if wtp_columns:
        wtp_data = []
        
        for column in wtp_columns:
            product_id = int(column.replace('wtp_', ''))
            
            # Get WTP by segment
            segment_wtp = wtp.groupby('customer_segment')[column].mean().reset_index()
            segment_wtp['product_id'] = product_id
            segment_wtp.rename(columns={column: 'mean_wtp'}, inplace=True)
            
            wtp_data.append(segment_wtp)
        
        product_wtp = pd.concat(wtp_data)
        
        # Pivot to have segments as columns
        product_wtp = product_wtp.pivot(
            index='product_id', 
            columns='customer_segment', 
            values='mean_wtp'
        ).reset_index()
        
        product_wtp.columns.name = None
        
        # Rename columns to be more descriptive
        product_wtp.rename(columns={
            'Budget': 'budget_wtp',
            'Mainstream': 'mainstream_wtp',
            'Premium': 'premium_wtp'
        }, inplace=True)
    else:
        product_wtp = pd.DataFrame({'product_id': products['product_id'].unique()})
    
    # 4. Combine all product features
    product_features = products.merge(product_sales, on='product_id', how='left')
    product_features = product_features.merge(product_behavior, on='product_id', how='left')
    product_features = product_features.merge(product_wtp, on='product_id', how='left')
    
    # 5. Fill missing values
    product_features.fillna({
        'num_transactions': 0,
        'total_quantity_sold': 0,
        'page_views': 0,
        'clicks': 0,
        'add_to_cart': 0,
        'remove_from_cart': 0,
        'cart_retention': 0,
        'engagement_score': 0
    }, inplace=True)
    
    # 6. Create price sensitivity features
    if 'base_price' in product_features.columns and 'avg_selling_price' in product_features.columns:
        product_features['discount_level'] = 1 - (product_features['avg_selling_price'] / product_features['base_price'])
    
    # 7. Add competitor pricing features if available
    if data_dict['competitors'] is not None:
        competitors = data_dict['competitors']
        competitor_prices = competitors.groupby('product_id')['price'].mean().reset_index()
        competitor_prices.rename(columns={'price': 'avg_competitor_price'}, inplace=True)
        
        product_features = product_features.merge(competitor_prices, on='product_id', how='left')
        
        # Calculate price positioning
        product_features['price_vs_competitor'] = product_features['base_price'] / product_features['avg_competitor_price']
    
    # Save processed data
    output_dir = 'data/processed'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filename = os.path.join(output_dir, f'product_features_{datetime.now().strftime("%Y%m%d")}.csv')
    product_features.to_csv(filename, index=False)
    
    logger.info(f"Product features saved to {filename}")
    return product_features

def create_bundle_features(data_dict):
    """Create bundle-level features for modeling."""
    logger.info("Creating bundle features...")
    
    bundles = data_dict['bundles'].copy()
    transactions = data_dict['transactions'].copy()
    
    # Extract bundle IDs
    bundle_ids = bundles['bundle_id'].unique()
    
    # Get bundle sales data
    bundle_sales = transactions[transactions['is_bundle'] == True].copy()
    
    # Calculate bundle metrics
    if not bundle_sales.empty:
        bundle_metrics = bundle_sales.groupby('bundle_id').agg({
            'transaction_id': 'count',
            'total_amount': 'mean',
            'discount_amount': 'mean'
        }).reset_index()
        
        bundle_metrics.rename(columns={
            'transaction_id': 'num_transactions',
            'total_amount': 'avg_revenue',
            'discount_amount': 'avg_discount'
        }, inplace=True)
        
        # Calculate discount percentage
        bundle_metrics['discount_percent'] = bundle_metrics['avg_discount'] / (bundle_metrics['avg_revenue'] + bundle_metrics['avg_discount'])
        
        # Merge with bundle data
        bundle_features = bundles.merge(bundle_metrics, on='bundle_id', how='left')
    else:
        bundle_features = bundles.copy()
        bundle_features['num_transactions'] = 0
        bundle_features['avg_revenue'] = 0
        bundle_features['avg_discount'] = 0
        bundle_features['discount_percent'] = 0
    
    # Fill missing values
    bundle_features.fillna({
        'num_transactions': 0,
        'avg_revenue': 0,
        'avg_discount': 0,
        'discount_percent': 0
    }, inplace=True)
    
    # Save processed data
    output_dir = 'data/processed'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filename = os.path.join(output_dir, f'bundle_features_{datetime.now().strftime("%Y%m%d")}.csv')
    bundle_features.to_csv(filename, index=False)
    
    logger.info(f"Bundle features saved to {filename}")
    return bundle_features

def main():
    """Main function to run feature engineering pipeline."""
    logger.info("Starting feature engineering process...")
    
    # Load all data
    data_dict = load_data()
    
    # Create features
    product_features = create_product_features(data_dict)
    bundle_features = create_bundle_features(data_dict)
    
    logger.info("Feature engineering complete!")
    
    return {
        'product_features': product_features,
        'bundle_features': bundle_features
    }

if __name__ == "__main__":
    main()