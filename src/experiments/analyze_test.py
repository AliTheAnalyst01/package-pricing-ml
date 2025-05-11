"""
Script to analyze the results of pricing A/B tests.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_analysis")

def get_latest_file(directory, prefix):
    """Get the most recent file with a given prefix in a directory."""
    files = [f for f in os.listdir(directory) if f.startswith(prefix)]
    if not files:
        return None
    return os.path.join(directory, sorted(files)[-1])

def analyze_test(experiment_id=None):
    """
    Analyze results from a pricing A/B test.
    
    Args:
        experiment_id: ID of the experiment to analyze, or None for latest
    """
    # Find experiment to analyze
    if experiment_id is None:
        experiment_file = get_latest_file('experiments', 'price_test')
        if not experiment_file or not experiment_file.endswith('_design.csv'):
            logger.error("No experiment design found!")
            return None
        experiment_id = os.path.basename(experiment_file).replace('_design.csv', '')
    
    # Load experiment design
    design_file = os.path.join('experiments', f"{experiment_id}_design.csv")
    if not os.path.exists(design_file):
        logger.error(f"Experiment design {design_file} not found!")
        return None
    
    experiment = pd.read_csv(design_file)
    logger.info(f"Analyzing results for experiment {experiment_id}...")
    
    # Simulate sales data for demonstration purposes
    # In a real system, you would load actual transaction data here
    simulated_results = []
    
    for _, product in experiment.iterrows():
        # Basic product info
        product_data = {
            'product_id': product['product_id'],
            'product_name': product['product_name'] if 'product_name' in product else f"Product {product['product_id']}",
            'control_price': product['control_price'],
            'test_price': product['test_price'],
            'price_change_pct': product['price_change_pct']
        }
        
        # Simulate control group data
        product_data['control_views'] = np.random.randint(500, 2000)
        product_data['control_conversion_rate'] = np.random.uniform(0.01, 0.05)
        product_data['control_orders'] = int(product_data['control_views'] * product_data['control_conversion_rate'])
        product_data['control_revenue'] = product_data['control_orders'] * product_data['control_price']
        
        # Simulate test group data with elasticity effect
        product_data['test_views'] = np.random.randint(500, 2000)
        
        # Apply price elasticity effect
        price_ratio = product_data['test_price'] / product_data['control_price']
        elasticity = -1.5  # Typical price elasticity
        
        # Calculate test conversion rate with elasticity effect
        conversion_effect = (price_ratio - 1) * elasticity
        product_data['test_conversion_rate'] = product_data['control_conversion_rate'] * (1 + conversion_effect)
        
        # Ensure sensible bounds for conversion rate
        product_data['test_conversion_rate'] = max(0.001, min(0.1, product_data['test_conversion_rate']))
        
        # Calculate test orders and revenue
        product_data['test_orders'] = int(product_data['test_views'] * product_data['test_conversion_rate'])
        product_data['test_revenue'] = product_data['test_orders'] * product_data['test_price']
        
        # Calculate metrics
        product_data['conversion_lift'] = (product_data['test_conversion_rate'] / product_data['control_conversion_rate'] - 1) * 100
        
        # Revenue per visitor
        product_data['control_rpm'] = product_data['control_revenue'] / product_data['control_views']
        product_data['test_rpm'] = product_data['test_revenue'] / product_data['test_views']
        product_data['rpm_lift'] = (product_data['test_rpm'] / product_data['control_rpm'] - 1) * 100
        
        # Determine if result is statistically significant (simplified)
        # In a real analysis, you would use proper statistical testing
        product_data['is_significant'] = abs(product_data['rpm_lift']) > 15
        
        # Recommendation
        if product_data['rpm_lift'] > 0 and product_data['is_significant']:
            product_data['recommendation'] = 'Implement new price'
        else:
            product_data['recommendation'] = 'Keep current price'
        
        simulated_results.append(product_data)
    
    # Create results DataFrame
    results = pd.DataFrame(simulated_results)
    
    # Save results
    output_file = os.path.join('experiments', f"{experiment_id}_results.csv")
    results.to_csv(output_file, index=False)
    logger.info(f"Analysis results saved to {output_file}")
    
    # Create a simple report
    report_file = os.path.join('experiments', f"{experiment_id}_report.txt")
    
    with open(report_file, 'w') as f:
        f.write(f"A/B TEST RESULTS: {experiment_id}\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write(f"Test Duration: {experiment['start_date'].iloc[0]} to {experiment['end_date'].iloc[0]}\n\n")
        
        f.write("SUMMARY:\n")
        successful = results[results['recommendation'] == 'Implement new price']
        f.write(f"Total products tested: {len(results)}\n")
        f.write(f"Successful price changes: {len(successful)} ({len(successful)/len(results)*100:.1f}%)\n\n")
        
        f.write("PRODUCT RESULTS:\n")
        for _, res in results.iterrows():
            f.write(f"- {res['product_name']}:\n")
            f.write(f"  Price change: ${res['control_price']:.2f} -> ${res['test_price']:.2f} ({res['price_change_pct']:.1f}%)\n")
            f.write(f"  Conversion rate: {res['control_conversion_rate']*100:.2f}% -> {res['test_conversion_rate']*100:.2f}% ({res['conversion_lift']:+.1f}%)\n")
            f.write(f"  Revenue per mille: ${res['control_rpm']*1000:.2f} -> ${res['test_rpm']*1000:.2f} ({res['rpm_lift']:+.1f}%)\n")
            f.write(f"  Recommendation: {res['recommendation']}\n\n")
    
    logger.info(f"Analysis report created at {report_file}")
    
    # Create a simple visualization
    plt.figure(figsize=(10, 6))
    
    # Create x-axis labels using product names
    labels = results['product_name'].apply(lambda x: x[:20] + '...' if len(x) > 20 else x)
    
    # Set positions for bars
    x = np.arange(len(labels))
    width = 0.35
    
    # Create bars
    plt.bar(x - width/2, results['control_rpm'] * 1000, width, label='Current Price')
    plt.bar(x + width/2, results['test_rpm'] * 1000, width, label='Test Price')
    
    # Add details
    plt.xlabel('Products')
    plt.ylabel('Revenue per 1000 Visitors ($)')
    plt.title('A/B Test Results: Revenue Impact of Price Changes')
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    plot_file = os.path.join('experiments', f"{experiment_id}_plot.png")
    plt.savefig(plot_file)
    logger.info(f"Results plot saved to {plot_file}")
    
    return results

if __name__ == "__main__":
    results = analyze_test()
    
    if results is not None:
        print("\nTest Analysis Complete!")
        print(f"Results saved to experiments folder")
        
        # Print a brief summary to console
        successful = results[results['recommendation'] == 'Implement new price']
        print(f"\nSummary: {len(successful)} out of {len(results)} price changes recommended")
        
        if len(successful) > 0:
            print("\nRecommended price changes:")
            for _, product in successful.iterrows():
                print(f"- {product['product_name']}: ${product['control_price']:.2f} -> ${product['test_price']:.2f}")