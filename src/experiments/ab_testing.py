"""
Module for designing and analyzing A/B tests for pricing optimization.
"""
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ab_testing")

def get_latest_file(directory, prefix):
    """Get the most recent file with a given prefix in a directory."""
    files = [f for f in os.listdir(directory) if f.startswith(prefix)]
    if not files:
        return None
    return os.path.join(directory, sorted(files)[-1])

def design_experiment(test_duration_days=14):
    """
    Design an A/B test for pricing recommendations.
    
    Args:
        test_duration_days: Duration of the test in days
    
    Returns:
        DataFrame with experiment design
    """
    logger.info("Designing pricing A/B test...")
    
    # Load latest product recommendations
    product_file = get_latest_file('results', 'product_pricing_recommendations_')
    if not product_file:
        logger.error("No product recommendations found!")
        return None
    
    products = pd.read_csv(product_file)
    
    # Select products for testing (those with significant price changes)
    test_candidates = products[abs(products['price_change_pct']) > 10].copy()
    
    if len(test_candidates) < 2:
        logger.warning("Not enough products with significant price changes for testing!")
        test_candidates = products.copy()
    
    # Select up to 5 products for testing
    test_products = test_candidates.sample(min(5, len(test_candidates)))
    
    # Create experiment design
    experiment_id = f"price_test_{datetime.now().strftime('%Y%m%d')}"
    start_date = datetime.now()
    end_date = start_date + timedelta(days=test_duration_days)
    
    experiment = test_products.copy()
    experiment['experiment_id'] = experiment_id
    experiment['start_date'] = start_date.strftime("%Y-%m-%d")
    experiment['end_date'] = end_date.strftime("%Y-%m-%d")
    experiment['control_price'] = experiment['base_price']
    experiment['test_price'] = experiment['optimal_price']
    
    # Calculate expected metrics
    experiment['expected_lift'] = experiment['profit_change_pct'] / 100  # Convert from percentage
    
    # Save experiment design
    output_dir = 'experiments'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_file = os.path.join(output_dir, f"{experiment_id}_design.csv")
    experiment.to_csv(output_file, index=False)
    
    logger.info(f"A/B test design saved to {output_file}")
    
    return experiment
def main():
    """Main function to create an A/B test design."""
    experiment = design_experiment()
    
    if experiment is not None:
        # Print summary
        print("\nA/B Test Design Summary:")
        print(f"Experiment ID: {experiment['experiment_id'].iloc[0]}")
        print(f"Start Date: {experiment['start_date'].iloc[0]}")
        print(f"End Date: {experiment['end_date'].iloc[0]}")
        print(f"Number of products in test: {len(experiment)}")
        
        print("\nTest Products:")
        for idx, row in experiment.iterrows():
            print(f"- {row['product_name']} (ID: {row['product_id']})")
            print(f"  Current Price: ${row['control_price']:.2f}")
            print(f"  Test Price: ${row['test_price']:.2f}")
            print(f"  Price Change: {row['price_change_pct']:.1f}%")
            print(f"  Expected Profit Lift: {row['expected_lift']*100:.1f}%")
            print("")

def analyze_results(experiment_id=None):
    """
    Analyze the results of a completed A/B test.
    
    Args:
        experiment_id: ID of experiment to analyze, or None for latest
    
    Returns:
        DataFrame with analysis results
    """
    # Find experiment to analyze
    if experiment_id is None:
        experiment_file = get_latest_file('experiments', '')
        if not experiment_file:
            logger.error("No experiments found!")
            return None
    else:
        experiment_file = os.path.join('experiments', f"{experiment_id}_design.csv")
        if not os.path.exists(experiment_file):
            logger.error(f"Experiment {experiment_id} not found!")
            return None
    
    # Load experiment design
    experiment = pd.read_csv(experiment_file)
    experiment_id = experiment['experiment_id'].iloc[0]
    
    logger.info(f"Analyzing results for experiment {experiment_id}...")
    
    # In a real-world scenario, you would load actual transaction data here
    # For this example, we'll simulate results
    results = []
    
    for idx, product in experiment.iterrows():
        # Get product details
        product_id = product['product_id']
        control_price = product['control_price']
        test_price = product['test_price']
        
        # Simulate control group metrics
        control_visitors = np.random.randint(800, 1200)
        control_conversion = np.random.uniform(0.01, 0.05)
        control_purchases = int(control_visitors * control_conversion)
        control_revenue = control_purchases * control_price
        
        # Simulate test group metrics with elasticity effect
        test_visitors = np.random.randint(800, 1200)
        
        # Price elasticity effect (higher price = lower conversion)
        price_change = test_price / control_price - 1
        elasticity = -1.5  # Assumed elasticity
        conversion_change = price_change * elasticity
        
        test_conversion = control_conversion * (1 + conversion_change)
        test_conversion = max(0.001, min(0.1, test_conversion))  # Keep in reasonable bounds
        
        test_purchases = int(test_visitors * test_conversion)
        test_revenue = test_purchases * test_price
        
        # Calculate metrics
        conversion_lift = (test_conversion / control_conversion - 1) * 100
        revenue_per_visitor_control = control_revenue / control_visitors
        revenue_per_visitor_test = test_revenue / test_visitors
        revenue_lift = (revenue_per_visitor_test / revenue_per_visitor_control - 1) * 100
        
        # Statistical significance (simplified)
        is_significant = abs(revenue_lift) > 10  # Simplified - in reality would use proper statistics
        
        results.append({
            'product_id': product_id,
            'product_name': product['product_name'] if 'product_name' in product else f"Product {product_id}",
            'control_price': control_price,
            'test_price': test_price,
            'control_visitors': control_visitors,
            'test_visitors': test_visitors,
            'control_conversion': control_conversion,
            'test_conversion': test_conversion,
            'control_purchases': control_purchases,
            'test_purchases': test_purchases,
            'control_revenue': control_revenue,
            'test_revenue': test_revenue,
            'conversion_lift': conversion_lift,
            'revenue_lift': revenue_lift,
            'is_significant': is_significant,
            'recommendation': 'Adopt' if revenue_lift > 0 and is_significant else 'Reject'
        })
    
    # Create analysis DataFrame
    analysis = pd.DataFrame(results)
    
    # Save analysis results
    output_file = os.path.join('experiments', f"{experiment_id}_results.csv")
    analysis.to_csv(output_file, index=False)
    
    logger.info(f"Analysis results saved to {output_file}")
    
    return analysis

if __name__ == "__main__":
    # Create a new experiment design
    main()
    
    # For demonstration, also analyze a simulated experiment
    # Uncomment this line to run the analysis
    # analyze_results()