"""
Enhanced pricing model with more sophisticated optimization (fixed version).
"""
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("enhanced_pricing")

def get_latest_file(directory, prefix):
    """Get the most recent file with a given prefix in a directory."""
    files = [f for f in os.listdir(directory) if f.startswith(prefix)]
    if not files:
        return None
    return os.path.join(directory, sorted(files)[-1])

def load_data():
    """Load all necessary data for enhanced pricing."""
    logger.info("Loading data for enhanced pricing model...")
    
    # Load raw product data
    product_file = 'data/external/products.csv'
    if not os.path.exists(product_file):
        logger.error(f"Product data file {product_file} not found!")
        return None
    
    products = pd.read_csv(product_file)
    
    # Load transaction data
    transaction_items_file = get_latest_file('data/raw/transactions', 'transaction_items_')
    if transaction_items_file:
        transactions = pd.read_csv(transaction_items_file)
    else:
        logger.warning("No transaction data found.")
        transactions = None
    
    # Load behavior data
    behavior_file = get_latest_file('data/raw/behavior', 'behavior_')
    if behavior_file:
        behavior = pd.read_csv(behavior_file)
    else:
        logger.warning("No behavior data found.")
        behavior = None
    
    # Load previous test results if available
    test_results_file = get_latest_file('experiments', 'price_test')
    if test_results_file and test_results_file.endswith('_results.csv'):
        test_results = pd.read_csv(test_results_file)
        logger.info(f"Loaded test results from {test_results_file}")
    else:
        test_results = None
    
    return {
        'products': products,
        'transactions': transactions,
        'behavior': behavior,
        'test_results': test_results
    }

def calculate_more_accurate_elasticity(data_dict):
    """
    Calculate price elasticity with improved accuracy.
    
    Uses a combination of transaction data and A/B test results when available.
    Falls back to product-specific elasticity estimates when necessary.
    """
    logger.info("Calculating refined price elasticity estimates...")
    
    products = data_dict['products']
    
    # Start with elasticity defaults
    # Check if category exists in products
    if 'category' in products.columns:
        # Group by category if it exists
        category_elasticity = {}
        # Check the existing categories and assign elasticity values
        for category in products['category'].unique():
            if 'electronic' in str(category).lower():
                category_elasticity[category] = -1.2  # Less elastic
            elif 'clothing' in str(category).lower() and 'men' in str(category).lower():
                category_elasticity[category] = -1.8
            elif 'clothing' in str(category).lower() and 'women' in str(category).lower():
                category_elasticity[category] = -2.0
            elif 'jewelery' in str(category).lower() or 'jewelry' in str(category).lower():
                category_elasticity[category] = -1.5
            else:
                category_elasticity[category] = -1.5  # Default
    
    # Initialize elasticity with defaults
    product_elasticity = []
    
    for idx, product in products.iterrows():
        product_id = product['product_id']
        
        # Default elasticity - either by category or general
        if 'category' in product and product['category'] in category_elasticity:
            default_elasticity = category_elasticity[product['category']]
        else:
            # Assign based on price point if category not available
            if 'price' in product or 'base_price' in product:
                price = product.get('price', product.get('base_price', 0))
                if price > 200:  # High-priced items tend to be more elastic
                    default_elasticity = -1.8
                elif price > 50:  # Mid-priced
                    default_elasticity = -1.5
                else:  # Low-priced items often less elastic
                    default_elasticity = -1.2
            else:
                default_elasticity = -1.5  # General default
        
        elasticity_data = {
            'product_id': product_id,
            'product_name': product['product_name'] if 'product_name' in product else f"Product {product_id}",
            'elasticity': default_elasticity,
            'elasticity_source': 'default'
        }
        
        # Add category if it exists
        if 'category' in product:
            elasticity_data['category'] = product['category']
        
        # Check if we have test results for this product
        if data_dict['test_results'] is not None:
            test_product = data_dict['test_results'][data_dict['test_results']['product_id'] == product_id]
            
            if not test_product.empty:
                # Calculate elasticity from test data
                test_product = test_product.iloc[0]
                
                if 'test_price' in test_product and 'control_price' in test_product:
                    price_ratio = test_product['test_price'] / test_product['control_price']
                    
                    if 'test_conversion_rate' in test_product and 'control_conversion_rate' in test_product:
                        conversion_ratio = test_product['test_conversion_rate'] / test_product['control_conversion_rate']
                        
                        if price_ratio != 1:  # Avoid division by zero
                            measured_elasticity = (conversion_ratio - 1) / (price_ratio - 1)
                            
                            # Apply bounds for sanity
                            if -5 <= measured_elasticity <= 0:
                                elasticity_data['elasticity'] = measured_elasticity
                                elasticity_data['elasticity_source'] = 'test_results'
        
        # If we have transaction data, use it to refine elasticity
        if data_dict['transactions'] is not None:
            product_transactions = data_dict['transactions'][data_dict['transactions']['product_id'] == product_id]
            
            if len(product_transactions) >= 5:  # Need sufficient transactions
                # Group by price and calculate quantity
                if 'unit_price' in product_transactions.columns and 'quantity' in product_transactions.columns:
                    price_qty = product_transactions.groupby('unit_price')['quantity'].sum().reset_index()
                    
                    if len(price_qty) >= 2:  # Need at least two price points
                        # Calculate percent changes
                        price_qty = price_qty.sort_values('unit_price')
                        price_qty['price_pct_change'] = price_qty['unit_price'].pct_change()
                        price_qty['qty_pct_change'] = price_qty['quantity'].pct_change()
                        
                        # Calculate elasticity for each price change
                        price_qty['point_elasticity'] = price_qty['qty_pct_change'] / price_qty['price_pct_change']
                        
                        # Average elasticity (excluding infinites and NaNs)
                        valid_elasticity = price_qty['point_elasticity'].replace([np.inf, -np.inf], np.nan).dropna()
                        
                        if len(valid_elasticity) > 0:
                            transaction_elasticity = valid_elasticity.mean()
                            
                            # Use if within reasonable bounds
                            if -5 <= transaction_elasticity <= 0:
                                elasticity_data['elasticity'] = transaction_elasticity
                                elasticity_data['elasticity_source'] = 'transactions'
        
        # Consider customer behavior data to adjust elasticity
        if data_dict['behavior'] is not None:
            product_behavior = data_dict['behavior'][data_dict['behavior']['product_id'] == product_id]
            
            if not product_behavior.empty:
                # Calculate engagement metrics if columns exist
                engagement = product_behavior['time_on_page'].mean() if 'time_on_page' in product_behavior else 0
                add_to_cart = product_behavior['add_to_cart'].sum() if 'add_to_cart' in product_behavior else 0
                views = product_behavior['page_views'].sum() if 'page_views' in product_behavior else 0
                
                # Calculate add-to-cart rate
                cart_rate = add_to_cart / views if views > 0 else 0
                
                # Adjust elasticity based on engagement and cart rate
                # Higher engagement and cart rate imply lower elasticity (less price sensitive)
                if engagement > 60 and cart_rate > 0.1:  # High engagement
                    elasticity_adjustment = 0.3  # Less elastic
                elif engagement < 20 or cart_rate < 0.02:  # Low engagement
                    elasticity_adjustment = -0.3  # More elastic
                else:
                    elasticity_adjustment = 0
                
                # Apply adjustment
                elasticity_data['elasticity'] += elasticity_adjustment
                
                # Ensure it stays in reasonable bounds
                elasticity_data['elasticity'] = max(-5, min(-0.5, elasticity_data['elasticity']))
        
        product_elasticity.append(elasticity_data)
    
    # Create DataFrame
    elasticity_df = pd.DataFrame(product_elasticity)
    
    return elasticity_df

def optimize_prices(data_dict, elasticity_df):
    """
    Generate optimized prices using refined elasticity estimates.
    
    Uses profit optimization with a more robust approach.
    """
    logger.info("Generating optimized prices...")
    
    products = data_dict['products']
    
    # Merge product data with elasticity
    product_df = products.merge(elasticity_df[['product_id', 'elasticity', 'elasticity_source']], 
                              on='product_id', how='left')
    
    # Replace NaN values in elasticity column
    product_df['elasticity'] = product_df['elasticity'].fillna(-1.5)
    
    # Calculate cost estimates
    if 'price' in product_df.columns:
        price_col = 'price'
    elif 'base_price' in product_df.columns:
        price_col = 'base_price'
    else:
        logger.error("No price column found in product data")
        return None
    
    # Assume standard cost margin
    product_df['estimated_cost'] = product_df[price_col] * 0.5
    
    # Calculate optimal price using elasticity-based formula
    # For profit maximization: Optimal Price = Cost / (1 + 1/elasticity)
    product_df['optimal_price'] = product_df.apply(
        lambda row: row['estimated_cost'] / (1 + 1/row['elasticity']) 
                    if row['elasticity'] < -1 else row[price_col] * 1.1,
        axis=1
    )
    
    # Apply more conservative price change limits based on A/B test results
    if data_dict['test_results'] is not None and len(data_dict['test_results']) > 0:
        # Check if recommendation column exists
        if 'recommendation' in data_dict['test_results'].columns:
            test_success_rate = sum(data_dict['test_results']['recommendation'] == 'Implement new price') / len(data_dict['test_results'])
            
            if test_success_rate < 0.2:  # Low success rate
                max_increase = 0.15  # Limit to 15% increase
                max_decrease = 0.10  # Limit to 10% decrease
                logger.info(f"Using conservative price change limits due to low test success rate")
            else:
                max_increase = 0.25  # Allow up to 25% increase
                max_decrease = 0.20  # Allow up to 20% decrease
        else:
            # Without valid test data, be conservative
            max_increase = 0.15
            max_decrease = 0.10
    else:
        # Without test data, be conservative
        max_increase = 0.15
        max_decrease = 0.10
    
    # Apply constraints
    product_df['min_price'] = product_df[price_col] * (1 - max_decrease)
    product_df['max_price'] = product_df[price_col] * (1 + max_increase)
    product_df['optimal_price'] = product_df.apply(
        lambda row: min(max(row['optimal_price'], row['min_price']), row['max_price']),
        axis=1
    )
    
    # Calculate expected metrics
    product_df['price_change_pct'] = (product_df['optimal_price'] / product_df[price_col] - 1) * 100
    product_df['expected_quantity_change_pct'] = product_df['price_change_pct'] * product_df['elasticity'] / 100
    
    # Calculate profit metrics
    product_df['current_margin'] = product_df[price_col] - product_df['estimated_cost']
    product_df['optimal_margin'] = product_df['optimal_price'] - product_df['estimated_cost']
    product_df['current_profit'] = product_df['current_margin'] * 100  # Assuming baseline quantity of 100
    
    # Calculate expected quantity at new price (baseline 100)
    product_df['expected_quantity'] = 100 * (1 + product_df['expected_quantity_change_pct'] / 100)
    product_df['expected_profit'] = product_df['optimal_margin'] * product_df['expected_quantity']
    product_df['profit_change_pct'] = (product_df['expected_profit'] / product_df['current_profit'] - 1) * 100
    
    # Only recommend changes that improve profit
    product_df['recommend_change'] = product_df['profit_change_pct'] > 1  # At least 1% improvement
    
    # Where no change is recommended, revert to base price
    for idx, row in product_df.iterrows():
        if not row['recommend_change']:
            product_df.at[idx, 'optimal_price'] = row[price_col]
            product_df.at[idx, 'price_change_pct'] = 0
            product_df.at[idx, 'profit_change_pct'] = 0
    
    # Format output
    result_cols = ['product_id', 'product_name', 'elasticity', 'elasticity_source',
                  price_col, 'optimal_price', 'price_change_pct', 
                  'profit_change_pct', 'recommend_change']
    
    # Add category if it exists
    if 'category' in product_df.columns:
        result_cols.insert(2, 'category')
    
    result_df = product_df[result_cols].copy()
    
    # Round monetary values
    result_df[price_col] = result_df[price_col].round(2)
    result_df['optimal_price'] = result_df['optimal_price'].round(2)
    result_df['price_change_pct'] = result_df['price_change_pct'].round(1)
    result_df['profit_change_pct'] = result_df['profit_change_pct'].round(1)
    
    # Sort by profit impact
    result_df = result_df.sort_values('profit_change_pct', ascending=False)
    
    return result_df

def optimize_bundles(data_dict, product_prices):
    """
    Generate optimized bundle discounts using refined approach.
    """
    logger.info("Optimizing bundle discounts...")
    
    # Load bundles
    bundle_file = 'data/external/bundles.csv'
    if not os.path.exists(bundle_file):
        logger.error(f"Bundle data file {bundle_file} not found!")
        return None
    
    bundles = pd.read_csv(bundle_file)
    
    if product_prices is None or bundles is None:
        logger.error("Missing required data for bundle optimization")
        return None
    
    # Determine the price column name
    if 'base_price' in product_prices.columns:
        price_col = 'base_price'
    elif 'price' in product_prices.columns:
        price_col = 'price'
    else:
        # Check for other possible price column names
        price_candidates = [col for col in product_prices.columns if 'price' in col.lower()]
        if price_candidates:
            price_col = price_candidates[0]
        else:
            logger.error("No price column found in product data")
            return None
    
    bundle_results = []
    
    for idx, bundle in bundles.iterrows():
        # Skip if product_ids is missing
        if 'product_ids' not in bundle or pd.isna(bundle['product_ids']):
            continue
        
        try:
            # Get the product IDs in this bundle
            product_ids = [int(pid.strip()) for pid in str(bundle['product_ids']).split(',')]
            
            # Get data for products in the bundle
            bundle_products = product_prices[product_prices['product_id'].isin(product_ids)]
            
            if bundle_products.empty:
                continue
            
            num_products = len(product_ids)
            
            # Calculate total individual price using optimal prices
            if 'optimal_price' in bundle_products.columns:
                total_price = bundle_products['optimal_price'].sum()
            else:
                # Fall back to base price if optimal not available
                total_price = bundle_products[price_col].sum()
            
            # Calculate average elasticity of products in bundle
            avg_elasticity = bundle_products['elasticity'].mean()
            
            # Refined discount calculation
            # Base discount based on number of products
            if num_products == 2:
                base_discount = 0.05
            elif num_products == 3:
                base_discount = 0.08
            elif num_products == 4:
                base_discount = 0.10
            else:
                base_discount = 0.12
            
            # Adjust for elasticity - more elastic products get bigger discounts
            elasticity_factor = min(1.5, max(0.8, abs(avg_elasticity) / 1.5))
            
            # Final discount - more conservative
            discount_pct = base_discount * elasticity_factor
            
            # Calculate bundle price
            bundle_price = total_price * (1 - discount_pct)
            
            # Format result
            bundle_results.append({
                'bundle_id': bundle['bundle_id'],
                'bundle_name': bundle['bundle_name'] if 'bundle_name' in bundle else f"Bundle {bundle['bundle_id']}",
                'num_products': num_products,
                'total_individual_price': total_price,
                'avg_elasticity': avg_elasticity,
                'discount_pct': discount_pct * 100,
                'optimal_bundle_price': bundle_price,
                'savings': total_price - bundle_price
            })
            
        except Exception as e:
            logger.warning(f"Error processing bundle {bundle['bundle_id']}: {e}")
            continue
    
    if not bundle_results:
        logger.warning("No valid bundle data for pricing")
        return None
    
    # Create DataFrame
    result_df = pd.DataFrame(bundle_results)
    
    # Round values
    result_df['total_individual_price'] = result_df['total_individual_price'].round(2)
    result_df['optimal_bundle_price'] = result_df['optimal_bundle_price'].round(2)
    result_df['savings'] = result_df['savings'].round(2)
    result_df['discount_pct'] = result_df['discount_pct'].round(1)
    result_df['avg_elasticity'] = result_df['avg_elasticity'].round(2)
    
    return result_df

def save_results(product_prices, bundle_prices):
    """Save the enhanced pricing recommendations."""
    logger.info("Saving enhanced pricing recommendations...")
    
    # Create output directory
    output_dir = 'results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d")
    
    # Save product pricing recommendations
    if product_prices is not None:
        product_filename = os.path.join(output_dir, f'enhanced_product_pricing_{timestamp}.csv')
        product_prices.to_csv(product_filename, index=False)
        logger.info(f"Enhanced product pricing saved to {product_filename}")
    
    # Save bundle pricing recommendations
    if bundle_prices is not None:
        bundle_filename = os.path.join(output_dir, f'enhanced_bundle_pricing_{timestamp}.csv')
        bundle_prices.to_csv(bundle_filename, index=False)
        logger.info(f"Enhanced bundle pricing saved to {bundle_filename}")
    
    # Create summary visualization
    if product_prices is not None and 'recommend_change' in product_prices.columns:
        plt.figure(figsize=(10, 6))
        
        recommended = product_prices[product_prices['recommend_change']]
        not_recommended = product_prices[~product_prices['recommend_change']]
        
        plt.scatter(recommended['elasticity'], recommended['price_change_pct'], 
                  label='Recommended Changes', color='green', alpha=0.7)
        plt.scatter(not_recommended['elasticity'], not_recommended['price_change_pct'], 
                  label='No Change Recommended', color='red', alpha=0.7)
        
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.axvline(x=-1, color='black', linestyle='--', alpha=0.3)
        
        plt.title('Price Elasticity vs. Recommended Price Changes')
        plt.xlabel('Price Elasticity')
        plt.ylabel('Recommended Price Change (%)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save figure
        plt.tight_layout()
        fig_file = os.path.join(output_dir, f'enhanced_pricing_plot_{timestamp}.png')
        plt.savefig(fig_file)
        logger.info(f"Summary plot saved to {fig_file}")

def main():
    """Main function to run enhanced pricing optimization."""
    logger.info("Starting enhanced pricing optimization...")
    
    # Load data
    data_dict = load_data()
    if data_dict is None:
        logger.error("Failed to load required data")
        return
    
    # Calculate refined elasticity
    elasticity_df = calculate_more_accurate_elasticity(data_dict)
    
    # Optimize product prices
    product_prices = optimize_prices(data_dict, elasticity_df)
    
    # Optimize bundle discounts
    bundle_prices = optimize_bundles(data_dict, product_prices)
    
    # Save results
    save_results(product_prices, bundle_prices)
    
    logger.info("Enhanced pricing optimization complete!")
    
    # Print summary
    if product_prices is not None and 'recommend_change' in product_prices.columns:
        recommended_changes = product_prices[product_prices['recommend_change']]
        print(f"\nEnhanced Pricing Summary:")
        print(f"- Total products analyzed: {len(product_prices)}")
        print(f"- Recommended price changes: {len(recommended_changes)} ({len(recommended_changes)/len(product_prices)*100:.1f}%)")
        
        if len(recommended_changes) > 0:
            avg_price_change = recommended_changes['price_change_pct'].mean()
            avg_profit_impact = recommended_changes['profit_change_pct'].mean()
            print(f"- Average recommended price change: {avg_price_change:.1f}%")
            print(f"- Average projected profit impact: {avg_profit_impact:.1f}%")
            
            print("\nTop 5 recommended price changes:")
            for idx, row in recommended_changes.head(5).iterrows():
                price_col = 'base_price' if 'base_price' in recommended_changes.columns else 'price'
                print(f"- {row['product_name']}: ${row[price_col]:.2f} -> ${row['optimal_price']:.2f} ({row['price_change_pct']:+.1f}%)")
                print(f"  Expected profit impact: {row['profit_change_pct']:+.1f}%")

if __name__ == "__main__":
    main()