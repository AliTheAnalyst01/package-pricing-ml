"""
A simplified approach to package pricing optimization.
"""
import pandas as pd
import numpy as np
import os
import pickle
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("pricing_optimization")

def get_latest_file(directory, prefix):
    """Get the most recent file with a given prefix in a directory."""
    files = [f for f in os.listdir(directory) if f.startswith(prefix)]
    if not files:
        return None
    return os.path.join(directory, sorted(files)[-1])

def load_data():
    """Load the necessary data for price optimization."""
    logger.info("Loading data...")
    
    # Load raw data first
    products_file = 'data/external/products.csv'
    products = pd.read_csv(products_file)
    
    bundles_file = 'data/external/bundles.csv'
    bundles = pd.read_csv(bundles_file)
    
    # Load processed features if available
    product_features_file = get_latest_file('data/processed', 'product_features_')
    if product_features_file:
        product_features = pd.read_csv(product_features_file)
    else:
        product_features = None
        logger.warning("No processed product features found")
    
    bundle_features_file = get_latest_file('data/processed', 'bundle_features_')
    if bundle_features_file:
        bundle_features = pd.read_csv(bundle_features_file)
    else:
        bundle_features = None
        logger.warning("No processed bundle features found")
    
    # Load behavioral data for demand estimation
    behavior_file = get_latest_file('data/raw/behavior', 'behavior_')
    if behavior_file:
        behavior = pd.read_csv(behavior_file)
    else:
        behavior = None
        logger.warning("No behavior data found")
    
    # Load transaction data
    transactions_file = get_latest_file('data/raw/transactions', 'transactions_')
    if transactions_file:
        transactions = pd.read_csv(transactions_file)
    else:
        transactions = None
        logger.warning("No transaction data found")
    
    items_file = get_latest_file('data/raw/transactions', 'transaction_items_')
    if items_file:
        items = pd.read_csv(items_file)
    else:
        items = None
        logger.warning("No transaction items data found")
    
    return {
        'products': products,
        'bundles': bundles,
        'product_features': product_features,
        'bundle_features': bundle_features,
        'behavior': behavior,
        'transactions': transactions,
        'items': items
    }

def estimate_price_elasticity(data_dict):
    """
    Estimate price elasticity for products based on transaction data.
    
    Price elasticity = % change in quantity / % change in price
    """
    logger.info("Estimating price elasticity...")
    
    products = data_dict['products']
    items = data_dict['items']
    
    if items is None or products is None:
        logger.error("Missing required data for elasticity estimation")
        return None
    
    # Create a simple estimate based on different price points for the same product
    if 'product_id' in items.columns and 'unit_price' in items.columns:
        # Group by product_id and unit_price to get quantity sold at each price point
        price_quantity = items.groupby(['product_id', 'unit_price']).size().reset_index(name='quantity')
        
        # Only consider products with at least 2 different price points
        product_counts = price_quantity.groupby('product_id').size()
        valid_products = product_counts[product_counts >= 2].index
        
        if len(valid_products) == 0:
            logger.warning("No products with multiple price points found, using simplified approach")
            
            # Use a simplified approach based on engagement vs. price ratio
            if data_dict['product_features'] is not None and 'engagement_score' in data_dict['product_features'].columns:
                product_df = data_dict['product_features']
                
                # Calculate price/engagement ratio
                if 'base_price' in product_df.columns and 'engagement_score' in product_df.columns:
                    product_df['price_engagement_ratio'] = product_df['base_price'] / product_df['engagement_score'].clip(lower=0.1)
                    
                    # Higher ratio might suggest lower elasticity (high price, high engagement)
                    # Normalize to get elasticity estimates between -0.5 and -2.5
                    max_ratio = product_df['price_engagement_ratio'].max()
                    min_ratio = product_df['price_engagement_ratio'].min()
                    
                    if max_ratio > min_ratio:
                        product_df['elasticity'] = -2.5 + 2.0 * (product_df['price_engagement_ratio'] - min_ratio) / (max_ratio - min_ratio)
                    else:
                        product_df['elasticity'] = -1.5  # Default elasticity
                    
                    return product_df[['product_id', 'elasticity']]
            
            # If we couldn't calculate based on engagement, return defaults
            elasticity_df = pd.DataFrame({
                'product_id': products['product_id'],
                'elasticity': -1.5  # Default elasticity
            })
            
            return elasticity_df
        
        # Calculate elasticity for products with multiple price points
        elasticity_results = []
        
        for product_id in valid_products:
            product_data = price_quantity[price_quantity['product_id'] == product_id].sort_values('unit_price')
            
            if len(product_data) >= 2:
                # Calculate % changes in price and quantity
                product_data['price_pct_change'] = product_data['unit_price'].pct_change()
                product_data['quantity_pct_change'] = product_data['quantity'].pct_change()
                
                # Calculate elasticity for each price change
                product_data['elasticity'] = product_data['quantity_pct_change'] / product_data['price_pct_change']
                
                # Average elasticity (excluding infinites and NaNs)
                valid_elasticity = product_data['elasticity'].replace([np.inf, -np.inf], np.nan).dropna()
                
                if len(valid_elasticity) > 0:
                    avg_elasticity = valid_elasticity.mean()
                    
                    # Sanity check - elasticity is typically negative and between -0.5 and -3
                    if avg_elasticity > 0:
                        avg_elasticity = -1.0  # Default if calculated wrong direction
                    elif avg_elasticity < -10:
                        avg_elasticity = -3.0  # Cap at reasonable value
                    
                    elasticity_results.append({
                        'product_id': product_id,
                        'elasticity': avg_elasticity
                    })
                else:
                    # If we couldn't calculate a valid elasticity
                    elasticity_results.append({
                        'product_id': product_id,
                        'elasticity': -1.5  # Default elasticity
                    })
            
        # Create DataFrame from results
        if elasticity_results:
            elasticity_df = pd.DataFrame(elasticity_results)
            
            # Add default elasticity for products not in the results
            missing_products = set(products['product_id']) - set(elasticity_df['product_id'])
            
            if missing_products:
                default_elasticity = pd.DataFrame({
                    'product_id': list(missing_products),
                    'elasticity': -1.5  # Default elasticity
                })
                
                elasticity_df = pd.concat([elasticity_df, default_elasticity])
            
            return elasticity_df
    
    # If we couldn't calculate elasticity from data, use default values
    elasticity_df = pd.DataFrame({
        'product_id': products['product_id'],
        'elasticity': -1.5  # Default elasticity
    })
    
    return elasticity_df

def calculate_optimal_prices(data_dict, elasticity_df):
    """
    Calculate optimal prices for products based on price elasticity.
    
    Uses the formula: Optimal Price = Cost / (1 + 1/elasticity)
    
    Since we don't have cost data, we'll assume costs are 50% of the current price
    unless we have competitor data to use as a reference.
    """
    logger.info("Calculating optimal product prices...")
    
    products = data_dict['products']
    
    if products is None or elasticity_df is None:
        logger.error("Missing required data for price optimization")
        return None
    
    # Merge product data with elasticity
    product_df = products.merge(elasticity_df, on='product_id', how='left')
    
    # Fill missing elasticity values with default
    product_df['elasticity'].fillna(-1.5, inplace=True)
    
    # Estimate cost as 50% of current price (typical retail markup)
    if 'base_price' in product_df.columns:
        product_df['estimated_cost'] = product_df['base_price'] * 0.5
        
        # Calculate optimal price
        # For a product with elasticity of -2, the formula gives:
        # Optimal Price = Cost / (1 + 1/-2) = Cost / 0.5 = Cost * 2
        product_df['optimal_price'] = product_df['estimated_cost'] / (1 + 1/product_df['elasticity'])
        
        # Apply some constraints (no negative prices, no extreme changes)
        product_df['optimal_price'] = product_df['optimal_price'].clip(lower=product_df['estimated_cost'] * 1.1)  # At least 10% above cost
        product_df['optimal_price'] = product_df['optimal_price'].clip(upper=product_df['base_price'] * 1.5)  # No more than 50% increase
        product_df['optimal_price'] = product_df['optimal_price'].clip(lower=product_df['base_price'] * 0.5)  # No more than 50% decrease
        
        # Calculate expected profit at optimal price
        product_df['current_profit_margin'] = product_df['base_price'] - product_df['estimated_cost']
        product_df['optimal_profit_margin'] = product_df['optimal_price'] - product_df['estimated_cost']
        
        # Calculate expected change in quantity
        product_df['price_change_pct'] = (product_df['optimal_price'] / product_df['base_price']) - 1
        product_df['expected_quantity_change_pct'] = product_df['price_change_pct'] * product_df['elasticity']
        
        # For simplicity, assume current quantity = 100
        product_df['current_quantity'] = 100
        product_df['expected_quantity'] = product_df['current_quantity'] * (1 + product_df['expected_quantity_change_pct'])
        
        # Calculate current and expected profit
        product_df['current_profit'] = product_df['current_profit_margin'] * product_df['current_quantity']
        product_df['expected_profit'] = product_df['optimal_profit_margin'] * product_df['expected_quantity']
        product_df['profit_change_pct'] = (product_df['expected_profit'] / product_df['current_profit']) - 1
        
        # Select columns for output
        result_df = product_df[[
            'product_id', 'product_name', 'category', 'base_price', 
            'elasticity', 'optimal_price', 'price_change_pct',
            'expected_quantity_change_pct', 'profit_change_pct'
        ]].copy()
        
        # Round monetary values and percentages
        result_df['base_price'] = result_df['base_price'].round(2)
        result_df['optimal_price'] = result_df['optimal_price'].round(2)
        result_df['price_change_pct'] = (result_df['price_change_pct'] * 100).round(1)
        result_df['expected_quantity_change_pct'] = (result_df['expected_quantity_change_pct'] * 100).round(1)
        result_df['profit_change_pct'] = (result_df['profit_change_pct'] * 100).round(1)
        
        return result_df
    else:
        logger.error("Missing price data in product dataset")
        return None

def calculate_bundle_discounts(data_dict, product_prices):
    """
    Calculate optimal bundle discounts based on product relationships.
    
    For bundles, we'll use simpler heuristics:
    - Larger bundles get bigger discounts
    - Products with higher elasticity get bigger discounts in bundles
    - Complementary products get bigger discounts
    """
    logger.info("Calculating optimal bundle discounts...")
    
    bundles = data_dict['bundles']
    
    if bundles is None or product_prices is None:
        logger.error("Missing required data for bundle optimization")
        return None
    
    # Start with a simple rule-based approach
    bundle_results = []
    
    for idx, bundle in bundles.iterrows():
        # Skip if product_ids is missing
        if 'product_ids' not in bundle or pd.isna(bundle['product_ids']):
            continue
        
        try:
            # Get the product IDs in this bundle
            product_ids = [int(pid.strip()) for pid in str(bundle['product_ids']).split(',')]
            
            # Calculate bundle statistics
            num_products = len(product_ids)
            
            # Get data for products in the bundle
            bundle_products = product_prices[product_prices['product_id'].isin(product_ids)]
            
            if bundle_products.empty:
                continue
            
            # Calculate total price without discount
            total_price = bundle_products['optimal_price'].sum()
            
            # Calculate average elasticity of products in the bundle
            avg_elasticity = bundle_products['elasticity'].mean()
            
            # Calculate discount percentage based on bundle size and elasticity
            # Base discount starts at 5% for 2 items and increases with bundle size
            base_discount = min(0.05 + (num_products - 2) * 0.03, 0.25)  # Cap at 25%
            
            # Adjust discount based on elasticity (more elastic = higher discount)
            elasticity_factor = min(1.5, max(0.5, abs(avg_elasticity) / 1.5))
            discount_pct = base_discount * elasticity_factor
            
            # Calculate final bundle price
            bundle_price = total_price * (1 - discount_pct)
            
            # Format results
            bundle_results.append({
                'bundle_id': bundle['bundle_id'],
                'bundle_name': bundle['bundle_name'] if 'bundle_name' in bundle else f"Bundle {bundle['bundle_id']}",
                'num_products': num_products,
                'avg_elasticity': avg_elasticity,
                'total_individual_price': total_price,
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
    
    # Round monetary values
    result_df['total_individual_price'] = result_df['total_individual_price'].round(2)
    result_df['optimal_bundle_price'] = result_df['optimal_bundle_price'].round(2)
    result_df['savings'] = result_df['savings'].round(2)
    result_df['discount_pct'] = result_df['discount_pct'].round(1)
    result_df['avg_elasticity'] = result_df['avg_elasticity'].round(2)
    
    return result_df

def visualize_results(product_prices, bundle_prices, output_dir='results'):
    """Create visualizations of pricing recommendations."""
    logger.info("Creating visualizations...")
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Visualization 1: Price Change Distribution
    if product_prices is not None:
        plt.figure(figsize=(10, 6))
        sns.histplot(product_prices['price_change_pct'], bins=20)
        plt.title('Distribution of Recommended Price Changes')
        plt.xlabel('Price Change (%)')
        plt.ylabel('Number of Products')
        plt.axvline(x=0, color='red', linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'price_change_distribution.png'))
        
        # Visualization 2: Price Elasticity vs. Price Change
        plt.figure(figsize=(10, 6))
        plt.scatter(product_prices['elasticity'], product_prices['price_change_pct'])
        plt.title('Price Elasticity vs. Recommended Price Change')
        plt.xlabel('Price Elasticity')
        plt.ylabel('Recommended Price Change (%)')
        plt.axhline(y=0, color='red', linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'elasticity_vs_price_change.png'))
        
        # Visualization 3: Category-wise Price Changes
        if 'category' in product_prices.columns:
            plt.figure(figsize=(12, 6))
            category_changes = product_prices.groupby('category')['price_change_pct'].mean().sort_values()
            sns.barplot(x=category_changes.index, y=category_changes.values)
            plt.title('Average Price Change by Category')
            plt.xlabel('Category')
            plt.ylabel('Average Price Change (%)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'category_price_changes.png'))
    
    # Visualization 4: Bundle Discount Distribution
    if bundle_prices is not None:
        plt.figure(figsize=(10, 6))
        sns.histplot(bundle_prices['discount_pct'], bins=10)
        plt.title('Distribution of Bundle Discounts')
        plt.xlabel('Discount (%)')
        plt.ylabel('Number of Bundles')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'bundle_discount_distribution.png'))
        
        # Visualization 5: Bundle Size vs. Discount
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=bundle_prices, x='num_products', y='discount_pct')
        plt.title('Bundle Size vs. Discount Percentage')
        plt.xlabel('Number of Products in Bundle')
        plt.ylabel('Discount (%)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'bundle_size_vs_discount.png'))
    
    logger.info(f"Visualizations saved to {output_dir} directory")

def save_results(product_prices, bundle_prices, output_dir='results'):
    """Save pricing recommendations to CSV files."""
    logger.info("Saving pricing recommendations...")
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d")
    
    # Save product pricing recommendations
    if product_prices is not None:
        product_filename = os.path.join(output_dir, f'product_pricing_recommendations_{timestamp}.csv')
        product_prices.to_csv(product_filename, index=False)
        logger.info(f"Product pricing recommendations saved to {product_filename}")
    
    # Save bundle pricing recommendations
    if bundle_prices is not None:
        bundle_filename = os.path.join(output_dir, f'bundle_pricing_recommendations_{timestamp}.csv')
        bundle_prices.to_csv(bundle_filename, index=False)
        logger.info(f"Bundle pricing recommendations saved to {bundle_filename}")
    
    # Create a simple HTML report
    if product_prices is not None or bundle_prices is not None:
        html_filename = os.path.join(output_dir, f'pricing_report_{timestamp}.html')
        
        with open(html_filename, 'w') as f:
            f.write("<html><head>")
            f.write("<title>Package Pricing Optimization Report</title>")
            f.write("<style>")
            f.write("body { font-family: Arial, sans-serif; margin: 20px; }")
            f.write("h1, h2 { color: #2c3e50; }")
            f.write("table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }")
            f.write("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
            f.write("th { background-color: #f2f2f2; }")
            f.write("tr:nth-child(even) { background-color: #f9f9f9; }")
            f.write(".up { color: green; }")
            f.write(".down { color: red; }")
            f.write("</style>")
            f.write("</head><body>")
            
            f.write("<h1>Package Pricing Optimization Report</h1>")
            f.write(f"<p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
            
            # Product pricing recommendations
            if product_prices is not None:
                f.write("<h2>Product Pricing Recommendations</h2>")
                
                # Summary statistics
                avg_price_change = product_prices['price_change_pct'].mean()
                price_increases = (product_prices['price_change_pct'] > 0).sum()
                price_decreases = (product_prices['price_change_pct'] < 0).sum()
                
                f.write("<p><strong>Summary:</strong></p>")
                f.write("<ul>")
                f.write(f"<li>Average recommended price change: {avg_price_change:.1f}%</li>")
                f.write(f"<li>Products with price increases: {price_increases} ({price_increases/len(product_prices)*100:.1f}%)</li>")
                f.write(f"<li>Products with price decreases: {price_decreases} ({price_decreases/len(product_prices)*100:.1f}%)</li>")
                f.write("</ul>")
                
                f.write("<p><strong>Top 5 Recommended Price Increases:</strong></p>")
                top_increases = product_prices[product_prices['price_change_pct'] > 0].sort_values('price_change_pct', ascending=False).head(5)
                
                f.write("<table>")
                f.write("<tr><th>Product</th><th>Category</th><th>Current Price</th><th>Recommended Price</th><th>Change (%)</th></tr>")
                
                for idx, row in top_increases.iterrows():
                    f.write("<tr>")
                    f.write(f"<td>{row['product_name']}</td>")
                    f.write(f"<td>{row['category']}</td>")
                    f.write(f"<td>${row['base_price']:.2f}</td>")
                    f.write(f"<td>${row['optimal_price']:.2f}</td>")
                    f.write(f"<td class='up'>+{row['price_change_pct']:.1f}%</td>")
                    f.write("</tr>")
                
                f.write("</table>")
                
                f.write("<p><strong>Top 5 Recommended Price Decreases:</strong></p>")
                top_decreases = product_prices[product_prices['price_change_pct'] < 0].sort_values('price_change_pct').head(5)
                
                f.write("<table>")
                f.write("<tr><th>Product</th><th>Category</th><th>Current Price</th><th>Recommended Price</th><th>Change (%)</th></tr>")
                
                for idx, row in top_decreases.iterrows():
                    f.write("<tr>")
                    f.write(f"<td>{row['product_name']}</td>")
                    f.write(f"<td>{row['category']}</td>")
                    f.write(f"<td>${row['base_price']:.2f}</td>")
                    f.write(f"<td>${row['optimal_price']:.2f}</td>")
                    f.write(f"<td class='down'>{row['price_change_pct']:.1f}%</td>")
                    f.write("</tr>")
                
                f.write("</table>")
            
            # Bundle pricing recommendations
            if bundle_prices is not None:
                f.write("<h2>Bundle Pricing Recommendations</h2>")
                
                # Summary statistics
                avg_discount = bundle_prices['discount_pct'].mean()
                avg_savings = bundle_prices['savings'].mean()
                
                f.write("<p><strong>Summary:</strong></p>")
                f.write("<ul>")
                f.write(f"<li>Average recommended bundle discount: {avg_discount:.1f}%</li>")
                f.write(f"<li>Average customer savings per bundle: ${avg_savings:.2f}</li>")
                f.write("</ul>")
                
                f.write("<p><strong>All Bundle Recommendations:</strong></p>")
                
                f.write("<table>")
                f.write("<tr><th>Bundle</th><th>Products</th><th>Total Individual Price</th><th>Bundle Price</th><th>Discount (%)</th><th>Customer Savings</th></tr>")
                
                for idx, row in bundle_prices.sort_values('discount_pct', ascending=False).iterrows():
                    f.write("<tr>")
                    f.write(f"<td>{row['bundle_name']}</td>")
                    f.write(f"<td>{row['num_products']}</td>")
                    f.write(f"<td>${row['total_individual_price']:.2f}</td>")
                    f.write(f"<td>${row['optimal_bundle_price']:.2f}</td>")
                    f.write(f"<td>{row['discount_pct']:.1f}%</td>")
                    f.write(f"<td>${row['savings']:.2f}</td>")
                    f.write("</tr>")
                
                f.write("</table>")
            
            # Add visualizations if they exist
            for img_file in ['price_change_distribution.png', 'category_price_changes.png', 'bundle_discount_distribution.png']:
                img_path = os.path.join(output_dir, img_file)
                if os.path.exists(img_path):
                    img_name = img_file.replace('.png', '').replace('_', ' ').title()
                    f.write(f"<h2>{img_name}</h2>")
                    f.write(f"<img src='{img_file}' alt='{img_name}' style='max-width:100%;'>")
            
            f.write("</body></html>")
        
        logger.info(f"Pricing report saved to {html_filename}")

def main():
    """Main function to run pricing optimization."""
    logger.info("Starting pricing optimization...")
    
    # Create output directory
    output_dir = 'results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load data
    data_dict = load_data()
    
    # Estimate price elasticity
    elasticity_df = estimate_price_elasticity(data_dict)
    
    # Calculate optimal product prices
    product_prices = calculate_optimal_prices(data_dict, elasticity_df)
    
    # Calculate optimal bundle discounts
    bundle_prices = calculate_bundle_discounts(data_dict, product_prices)
    
    # Create visualizations
    visualize_results(product_prices, bundle_prices, output_dir)
    
    # Save results
    save_results(product_prices, bundle_prices, output_dir)
    
    logger.info("Pricing optimization complete!")
    
    return {
        'product_prices': product_prices,
        'bundle_prices': bundle_prices
    }

if __name__ == "__main__":
    main()