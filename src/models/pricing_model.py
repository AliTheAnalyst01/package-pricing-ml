"""
Machine learning models for package pricing optimization.
"""
import pandas as pd
import numpy as np
import os
import pickle
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("pricing_model")

def get_latest_file(directory, prefix):
    """Get the most recent file with a given prefix in a directory."""
    files = [f for f in os.listdir(directory) if f.startswith(prefix)]
    if not files:
        return None
    return os.path.join(directory, sorted(files)[-1])

def load_features():
    """Load the latest processed features."""
    logger.info("Loading processed features...")
    
    # Find latest product features file
    product_features_file = get_latest_file('data/processed', 'product_features_')
    if not product_features_file:
        logger.error("No product features file found!")
        return None
    
    # Find latest bundle features file
    bundle_features_file = get_latest_file('data/processed', 'bundle_features_')
    if not bundle_features_file:
        logger.error("No bundle features file found!")
        return None
    
    # Load the data
    product_features = pd.read_csv(product_features_file)
    bundle_features = pd.read_csv(bundle_features_file)
    
    return {
        'product_features': product_features,
        'bundle_features': bundle_features
    }

def build_product_pricing_model(product_features):
    """
    Build a model to predict optimal product pricing.
    
    For this simple model, we'll predict the average selling price
    based on product attributes and customer behavior.
    """
    logger.info("Building product pricing model...")
    
    # Basic validation
    if product_features is None or product_features.empty:
        logger.error("No valid product features data!")
        return None
    
    # Define features to use
    # Adjust these based on your actual data columns
    feature_cols = [
        'category', 'engagement_score', 'page_views', 'time_on_page',
        'cart_retention', 'total_quantity_sold'
    ]
    
    categorical_cols = ['category']
    
    # Check which columns actually exist
    available_features = [col for col in feature_cols if col in product_features.columns]
    
    if not available_features:
        logger.error("No valid features available!")
        return None
    
    # Prepare categorical features
    if categorical_cols:
        # One-hot encode categorical features
        product_data = pd.get_dummies(product_features, columns=categorical_cols, drop_first=True)
    else:
        product_data = product_features.copy()
    
    # Re-check which columns exist after transformation
    feature_cols = [col for col in product_data.columns 
                   if col.startswith(tuple(categorical_cols)) or col in available_features]
    
    # Define target variable - avg_selling_price or base_price
    if 'avg_selling_price' in product_data.columns:
        target_col = 'avg_selling_price'
    elif 'base_price' in product_data.columns:
        target_col = 'base_price'
    else:
        logger.error("No valid target column found!")
        return None
    
    # Remove rows with missing target
    product_data = product_data.dropna(subset=[target_col])
    
    if product_data.empty:
        logger.error("No valid data after dropping missing targets!")
        return None
    
    # Remove any NaN values in features
    product_data = product_data.dropna(subset=feature_cols)
    
    if product_data.empty or len(product_data) < 5:
        logger.error("Not enough data for modeling after cleaning!")
        return None
    
    # Split data into features and target
    X = product_data[feature_cols]
    y = product_data[target_col]
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    logger.info(f"Product pricing model performance: MSE={mse:.2f}, R²={r2:.2f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    logger.info("Top 5 features for product pricing:")
    for idx, row in feature_importance.head(5).iterrows():
        logger.info(f"  {row['Feature']}: {row['Importance']:.4f}")
    
    # Save model
    model_dir = 'models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    model_file = os.path.join(model_dir, f'product_pricing_model_{datetime.now().strftime("%Y%m%d")}.pkl')
    with open(model_file, 'wb') as f:
        pickle.dump({
            'model': model,
            'features': feature_cols,
            'target': target_col,
            'performance': {
                'mse': mse,
                'r2': r2
            },
            'feature_importance': feature_importance.to_dict()
        }, f)
    
    logger.info(f"Product pricing model saved to {model_file}")
    
    return {
        'model': model,
        'features': feature_cols,
        'importance': feature_importance,
        'performance': {
            'mse': mse,
            'r2': r2
        }
    }

def build_bundle_pricing_model(bundle_features, product_features):
    """
    Build a model to predict optimal bundle pricing.
    
    For this simple model, we'll predict the optimal bundle discount
    based on the bundle composition and product attributes.
    """
    logger.info("Building bundle pricing model...")
    
    # Basic validation
    if bundle_features is None or bundle_features.empty:
        logger.error("No valid bundle features data!")
        return None
    
    if product_features is None or product_features.empty:
        logger.error("No valid product features data!")
        return None
    
    # Process bundle composition
    # Extract product IDs from the bundle
    bundle_data = bundle_features.copy()
    
    # For each bundle, calculate average metrics of included products
    merged_data = []
    
    for idx, bundle in bundle_data.iterrows():
        # Skip if product_ids is missing
        if 'product_ids' not in bundle or pd.isna(bundle['product_ids']):
            continue
        
        try:
            # Get the product IDs in this bundle
            product_ids = [int(pid.strip()) for pid in str(bundle['product_ids']).split(',')]
            
            # Get features for these products
            bundle_products = product_features[product_features['product_id'].isin(product_ids)]
            
            if bundle_products.empty:
                continue
            
            # Calculate average metrics
            avg_metrics = {
                'bundle_id': bundle['bundle_id'],
                'num_products': len(product_ids)
            }
            
            # Add bundle-specific features
            for col in bundle.index:
                if col not in ['product_ids', 'bundle_id']:
                    avg_metrics[f'bundle_{col}'] = bundle[col]
            
            # Calculate average product metrics
            for col in product_features.columns:
                if col not in ['product_id', 'product_name', 'description', 'image_url']:
                    if pd.api.types.is_numeric_dtype(product_features[col]):
                        avg_metrics[f'avg_product_{col}'] = bundle_products[col].mean()
            
            merged_data.append(avg_metrics)
        except Exception as e:
            logger.warning(f"Error processing bundle {bundle['bundle_id']}: {e}")
            continue
    
    if not merged_data:
        logger.error("No valid bundle data after merging with products!")
        return None
    
    # Create DataFrame
    bundle_data = pd.DataFrame(merged_data)
    
    # Define features and target
    if 'discount_percent' in bundle_data.columns:
        target_col = 'discount_percent'
    elif 'avg_discount' in bundle_data.columns:
        target_col = 'avg_discount'
    else:
        logger.error("No valid target column found!")
        return None
    
    # Filter out non-numeric or redundant columns
    feature_cols = [col for col in bundle_data.columns 
                   if col not in [target_col, 'bundle_id'] 
                   and pd.api.types.is_numeric_dtype(bundle_data[col])
                   and not bundle_data[col].isna().all()]
    
    # Make sure we have enough data
    if len(bundle_data) < 5:
        logger.warning("Not enough data for bundle pricing model. Using simple rule-based approach.")
        
        # Return a simple rule-based model
        avg_discount = bundle_data[target_col].mean() if not bundle_data[target_col].isna().all() else 0.15
        
        return {
            'model_type': 'rule_based',
            'avg_discount': avg_discount
        }
    
    # Remove rows with missing target
    bundle_data = bundle_data.dropna(subset=[target_col])
    
    # Remove any NaN values in features
    bundle_data = bundle_data.dropna(subset=feature_cols)
    
    # Split data into features and target
    X = bundle_data[feature_cols]
    y = bundle_data[target_col]
    
    # Train Linear Regression model (simpler due to likely small data size)
    model = LinearRegression()
    model.fit(X, y)
    
    # Evaluate model
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    logger.info(f"Bundle pricing model performance: MSE={mse:.4f}, R²={r2:.4f}")
    
    # Calculate coefficient importance
    coef_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    logger.info("Top 5 features for bundle pricing:")
    for idx, row in coef_importance.head(5).iterrows():
        logger.info(f"  {row['Feature']}: {row['Coefficient']:.4f}")
    
    # Save model
    model_dir = 'models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    model_file = os.path.join(model_dir, f'bundle_pricing_model_{datetime.now().strftime("%Y%m%d")}.pkl')
    with open(model_file, 'wb') as f:
        pickle.dump({
            'model': model,
            'features': feature_cols,
            'target': target_col,
            'performance': {
                'mse': mse,
                'r2': r2
            },
            'coefficients': coef_importance.to_dict()
        }, f)
    
    logger.info(f"Bundle pricing model saved to {model_file}")
    
    return {
        'model': model,
        'features': feature_cols,
        'coefficients': coef_importance,
        'performance': {
            'mse': mse,
            'r2': r2
        }
    }

def main():
    """Main function to run pricing model pipeline."""
    logger.info("Starting pricing model development...")
    
    # Load features
    features = load_features()
    if not features:
        logger.error("Failed to load features. Aborting.")
        return
    
    # Build product pricing model
    product_model = build_product_pricing_model(features['product_features'])
    
    # Build bundle pricing model
    bundle_model = build_bundle_pricing_model(features['bundle_features'], features['product_features'])
    
    logger.info("Pricing model development complete!")
    
    return {
        'product_model': product_model,
        'bundle_model': bundle_model
    }

if __name__ == "__main__":
    main()