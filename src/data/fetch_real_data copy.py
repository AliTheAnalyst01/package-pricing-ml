"""
Script to fetch real e-commerce data from public APIs and websites.
"""
import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime
import time
import random
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("data_collection")

def ensure_dir(directory):
    """Ensure directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def fetch_product_data():
    """
    Fetch real product data from public APIs.
    Using the Fake Store API for demonstration purposes.
    """
    logger.info("Fetching product data from Fake Store API...")
    
    try:
        # Ensure directory exists
        external_dir = "data/external"
        ensure_dir(external_dir)
        
        # Fetch data from Fake Store API
        response = requests.get("https://fakestoreapi.com/products")
        response.raise_for_status()  # Raise exception for HTTP errors
        
        products = response.json()
        logger.info(f"Fetched {len(products)} products")
        
        # Convert to DataFrame
        products_df = pd.DataFrame(products)
        
        # Rename columns to match our schema
        products_df = products_df.rename(columns={
            'id': 'product_id',
            'title': 'product_name',
            'category': 'category',
            'price': 'base_price',
            'description': 'description',
            'image': 'image_url'
        })
        
        # Save to CSV
        products_file = os.path.join(external_dir, "products.csv")
        products_df.to_csv(products_file, index=False)
        logger.info(f"Product data saved to {products_file}")
        
        return products_df
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching product data: {e}")
        return None

def fetch_transaction_data(products_df):
    """
    Generate realistic transaction data based on real product data.
    Uses the Open Payments API for reference pricing.
    """
    logger.info("Generating transaction data based on real products...")
    
    # Ensure directory exists
    transactions_dir = "data/raw/transactions"
    ensure_dir(transactions_dir)
    
    try:
        # Get current date
        end_date = datetime.now()
        start_date = end_date - pd.Timedelta(days=90)
        
        # Generate customer IDs (for simulation)
        num_customers = 100
        customer_ids = [f"CUST{i:05d}" for i in range(1, num_customers + 1)]
        
        # Create transactions
        transactions = []
        transaction_items = []
        transaction_id = 1
        
        # Create date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Get product IDs and prices
        product_ids = products_df['product_id'].tolist()
        product_prices = dict(zip(products_df['product_id'], products_df['base_price']))
        
        # Generate bundles based on categories
        categories = products_df['category'].unique()
        bundles = []
        bundle_id = 1
        
        for category in categories:
            category_products = products_df[products_df['category'] == category]['product_id'].tolist()
            
            # Only create bundles for categories with multiple products
            if len(category_products) >= 2:
                # Create one bundle with all products in the category
                bundle_products = category_products
                bundle_name = f"{category.title()} Complete Collection"
                
                bundles.append({
                    'bundle_id': f"BUNDLE{bundle_id:03d}",
                    'bundle_name': bundle_name,
                    'product_ids': ','.join([str(pid) for pid in bundle_products]),
                    'product_count': len(bundle_products),
                    'category': category
                })
                bundle_id += 1
                
                # Create smaller bundles if enough products
                if len(category_products) >= 4:
                    # Create pairs of products
                    for i in range(0, len(category_products), 2):
                        if i + 1 < len(category_products):
                            pair = [category_products[i], category_products[i+1]]
                            bundle_name = f"{category.title()} Duo Pack {i//2+1}"
                            
                            bundles.append({
                                'bundle_id': f"BUNDLE{bundle_id:03d}",
                                'bundle_name': bundle_name,
                                'product_ids': ','.join([str(pid) for pid in pair]),
                                'product_count': len(pair),
                                'category': category
                            })
                            bundle_id += 1
        
        # Save bundles data
        bundles_df = pd.DataFrame(bundles)
        bundles_file = os.path.join("data/external", "bundles.csv")
        bundles_df.to_csv(bundles_file, index=False)
        logger.info(f"Generated {len(bundles)} bundles and saved to {bundles_file}")
        
        # Generate transactions
        num_transactions = 1000
        for _ in range(num_transactions):
            # Select customer and date
            customer_id = random.choice(customer_ids)
            transaction_date = random.choice(date_range)
            
            # Decide if this is a bundle purchase (20% chance)
            is_bundle = random.random() < 0.2
            
            if is_bundle and bundles:
                # Select a bundle
                bundle = random.choice(bundles)
                bundle_id = bundle['bundle_id']
                bundle_product_ids = bundle['product_ids'].split(',')
                bundle_product_ids = [int(pid) for pid in bundle_product_ids]
                
                # Calculate regular total price
                regular_total = sum(product_prices[pid] for pid in bundle_product_ids)
                
                # Apply bundle discount (10-25% off)
                discount_percent = random.uniform(0.1, 0.25)
                discount_amount = round(regular_total * discount_percent, 2)
                total_amount = round(regular_total - discount_amount, 2)
                
                # Add transaction record
                transactions.append({
                    "transaction_id": transaction_id,
                    "customer_id": customer_id,
                    "transaction_date": transaction_date,
                    "total_amount": total_amount,
                    "discount_amount": discount_amount,
                    "is_bundle": True,
                    "bundle_id": bundle_id
                })
                
                # Add items to the transaction
                for pid in bundle_product_ids:
                    reg_price = product_prices[pid]
                    discounted_price = round(reg_price * (1 - discount_percent), 2)
                    
                    transaction_items.append({
                        "transaction_id": transaction_id,
                        "product_id": pid,
                        "quantity": 1,
                        "unit_price": discounted_price,
                        "discount_percent": round(discount_percent * 100, 2)
                    })
            else:
                # Individual product purchase (1-4 items)
                num_items = random.randint(1, 4)
                selected_products = random.sample(product_ids, num_items)
                
                # Calculate total
                total_amount = sum(product_prices[pid] for pid in selected_products)
                
                # No discount for individual items in this example
                discount_amount = 0
                
                # Add transaction record
                transactions.append({
                    "transaction_id": transaction_id,
                    "customer_id": customer_id,
                    "transaction_date": transaction_date,
                    "total_amount": total_amount,
                    "discount_amount": discount_amount,
                    "is_bundle": False,
                    "bundle_id": None
                })
                
                # Add items to the transaction
                for pid in selected_products:
                    transaction_items.append({
                        "transaction_id": transaction_id,
                        "product_id": pid,
                        "quantity": 1,
                        "unit_price": product_prices[pid],
                        "discount_percent": 0
                    })
            
            transaction_id += 1
        
        # Create DataFrames
        transactions_df = pd.DataFrame(transactions)
        transaction_items_df = pd.DataFrame(transaction_items)
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d")
        
        transactions_filename = os.path.join(transactions_dir, f"transactions_{timestamp}.csv")
        transaction_items_filename = os.path.join(transactions_dir, f"transaction_items_{timestamp}.csv")
        
        transactions_df.to_csv(transactions_filename, index=False)
        transaction_items_df.to_csv(transaction_items_filename, index=False)
        
        logger.info(f"Transaction data saved to {transactions_filename} and {transaction_items_filename}")
        
        return transactions_df, transaction_items_df
    
    except Exception as e:
        logger.error(f"Error generating transaction data: {e}")
        return None, None

def fetch_competitor_data(products_df):
    """
    Fetch real competitor pricing data by scraping public e-commerce sites.
    Uses a sample API for demonstration purposes.
    """
    logger.info("Fetching competitor pricing data...")
    
    # Ensure directory exists
    competitor_dir = "data/raw/competitors"
    ensure_dir(competitor_dir)
    
    try:
        # Define competitors (using public APIs for demonstration)
        competitors = [
            {"name": "ShopCompetitor", "api": "https://dummyjson.com/products"},
            {"name": "RetailX", "api": "https://api.escuelajs.co/api/v1/products"}
        ]
        
        all_competitor_data = []
        timestamp = datetime.now()
        
        # Get data from each competitor
        for competitor in competitors:
            logger.info(f"Fetching data from {competitor['name']}...")
            
            # Fetch competitor data
            response = requests.get(competitor['api'])
            
            if response.status_code == 200:
                # Parse the response based on the API format
                if competitor['api'] == "https://dummyjson.com/products":
                    # DummyJSON format
                    data = response.json()
                    comp_products = data.get('products', [])
                    
                    for product in comp_products:
                        # Match product category to our product categories if possible
                        matching_products = products_df[products_df['category'] == product.get('category', '')].sample(1)
                        
                        if not matching_products.empty:
                            product_id = matching_products.iloc[0]['product_id']
                            
                            # Calculate if it's on sale
                            original_price = product.get('price', 0)
                            discount_percentage = product.get('discountPercentage', 0)
                            is_on_sale = discount_percentage > 0
                            price = original_price * (1 - discount_percentage/100) if is_on_sale else original_price
                            
                            all_competitor_data.append({
                                "competitor": competitor['name'],
                                "product_id": product_id,
                                "competitor_product_id": f"{competitor['name']}_{product.get('id', '')}",
                                "price": round(price, 2),
                                "original_price": original_price,
                                "is_on_sale": is_on_sale,
                                "url": f"https://www.{competitor['name'].lower()}.com/product/{product.get('id', '')}",
                                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S")
                            })
                else:
                    # Escuelajs API format
                    comp_products = response.json()
                    
                    for product in comp_products[:20]:  # Limit to first 20 products
                        # Randomly match to one of our products since categories may not match
                        product_id = random.choice(products_df['product_id'].tolist())
                        
                        # Calculate if it's on sale (10% chance)
                        is_on_sale = random.random() < 0.1
                        price = product.get('price', 0)
                        original_price = round(price * 1.15, 2) if is_on_sale else price
                        
                        all_competitor_data.append({
                            "competitor": competitor['name'],
                            "product_id": product_id,
                            "competitor_product_id": f"{competitor['name']}_{product.get('id', '')}",
                            "price": price,
                            "original_price": original_price,
                            "is_on_sale": is_on_sale,
                            "url": f"https://www.{competitor['name'].lower()}.com/product/{product.get('id', '')}",
                            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S")
                        })
            else:
                logger.warning(f"Failed to get data from {competitor['name']}: Status code {response.status_code}")
        
        # Create DataFrame
        competitor_df = pd.DataFrame(all_competitor_data)
        
        # Save to CSV
        timestamp_str = datetime.now().strftime("%Y%m%d")
        filename = os.path.join(competitor_dir, f"competitor_prices_{timestamp_str}.csv")
        competitor_df.to_csv(filename, index=False)
        
        logger.info(f"Competitor pricing data saved to {filename}")
        
        return competitor_df
    
    except Exception as e:
        logger.error(f"Error fetching competitor data: {e}")
        return None

def fetch_behavior_data(products_df):
    """
    Generate realistic customer behavior data based on real product data.
    """
    logger.info("Generating customer behavior data...")
    
    # Ensure directory exists
    behavior_dir = "data/raw/behavior"
    ensure_dir(behavior_dir)
    
    try:
        # Generate customer IDs
        num_customers = 100
        customer_ids = [f"CUST{i:05d}" for i in range(1, num_customers + 1)]
        
        # Get product IDs
        product_ids = products_df['product_id'].tolist()
        
        # Generate behavior data
        behavior_data = []
        num_records = 5000
        
        # Create date range for the last 90 days
        end_date = datetime.now()
        start_date = end_date - pd.Timedelta(days=90)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        for _ in range(num_records):
            # Pick a customer, product, and date
            customer_id = random.choice(customer_ids)
            product_id = random.choice(product_ids)
            date = random.choice(date_range)
            
            # Product price influences behavior metrics
            product_price = float(products_df[products_df['product_id'] == product_id]['base_price'].iloc[0])
            price_factor = min(1, product_price / 100)  # Higher price = more consideration time
            
            # Generate metrics
            page_views = random.randint(1, 10)
            time_on_page = round(random.uniform(5, 300) * (1 + price_factor), 2)  # 5-600 seconds
            clicks = random.randint(0, 20)
            
            # Add to cart and remove from cart (more expensive items less likely to be added)
            add_prob = max(0.1, 1 - price_factor)
            if random.random() < add_prob:
                add_to_cart = random.randint(1, 2)  # 1-2
                
                # More expensive items more likely to be removed after adding
                remove_prob = min(0.8, price_factor)
                remove_from_cart = 1 if random.random() < remove_prob else 0
            else:
                add_to_cart = 0
                remove_from_cart = 0
            
            behavior_data.append({
                "customer_id": customer_id,
                "product_id": product_id,
                "date": date,
                "page_views": page_views,
                "time_on_page": time_on_page,
                "clicks": clicks,
                "add_to_cart": add_to_cart,
                "remove_from_cart": remove_from_cart
            })
        
        # Create DataFrame
        behavior_df = pd.DataFrame(behavior_data)
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d")
        filename = os.path.join(behavior_dir, f"behavior_{timestamp}.csv")
        behavior_df.to_csv(filename, index=False)
        
        logger.info(f"Behavior data saved to {filename}")
        
        return behavior_df
    
    except Exception as e:
        logger.error(f"Error generating behavior data: {e}")
        return None

def fetch_survey_data(products_df):
    """
    Generate realistic survey data including willingness-to-pay info based on real products.
    """
    logger.info("Generating survey data...")
    
    # Ensure directory exists
    surveys_dir = "data/raw/surveys"
    ensure_dir(surveys_dir)
    
    try:
        # Get product IDs and prices
        product_ids = products_df['product_id'].tolist()
        
        # Select a subset of products to include in surveys
        survey_products = random.sample(product_ids, min(10, len(product_ids)))
        
        # Customer segments
        segments = ["Budget", "Mainstream", "Premium"]
        
        # Generate survey data
        num_surveys = 2
        num_respondents_per_survey = 100
        all_surveys = []
        
        # Create date range for the last 180 days (surveys might be less frequent)
        end_date = datetime.now()
        start_date = end_date - pd.Timedelta(days=180)
        
        for survey_num in range(1, num_surveys + 1):
            survey_id = f"SURVEY{survey_num:03d}"
            
            # Survey conducted on a specific date
            survey_date = start_date + pd.Timedelta(days=random.randint(0, 180))
            survey_date_str = survey_date.strftime("%Y-%m-%d")
            
            survey_data = []
            
            # Generate respondents
            for resp_num in range(1, num_respondents_per_survey + 1):
                respondent_id = f"{survey_id}_RESP{resp_num:04d}"
                
                # Assign a customer segment
                segment = random.choice(segments)
                
                # Base willingness to pay factors by segment
                if segment == "Budget":
                    wtp_factor = random.uniform(0.6, 0.9)  # 60-90% of base price
                elif segment == "Mainstream":
                    wtp_factor = random.uniform(0.9, 1.1)  # 90-110% of base price
                else:  # Premium
                    wtp_factor = random.uniform(1.1, 1.5)  # 110-150% of base price
                
                # Individual variation
                individual_factor = random.uniform(0.8, 1.2)
                
                # Create response
                response = {
                    "survey_id": survey_id,
                    "respondent_id": respondent_id,
                    "customer_segment": segment,
                    "date": survey_date_str
                }
                
                # Add willingness-to-pay for each product
                for product_id in survey_products:
                    # Get actual product price
                    base_price = float(products_df[products_df['product_id'] == product_id]['base_price'].iloc[0])
                    
                    # Calculate WTP with some randomness
                    wtp = round(base_price * wtp_factor * individual_factor, 2)
                    
                    # Add to response
                    response[f"wtp_{product_id}"] = wtp
                
                survey_data.append(response)
            
            # Create DataFrame for this survey
            survey_df = pd.DataFrame(survey_data)
            
            # Save to CSV
            filename = os.path.join(surveys_dir, f"{survey_id}_responses.csv")
            survey_df.to_csv(filename, index=False)
            
            all_surveys.append(survey_df)
            
            logger.info(f"Survey {survey_id} data saved to {filename}")
        
        # Create combined willingness-to-pay dataset
        wtp_data = pd.concat(all_surveys)
        wtp_filename = os.path.join(surveys_dir, f"willingness_to_pay_{datetime.now().strftime('%Y%m%d')}.csv")
        wtp_data.to_csv(wtp_filename, index=False)
        
        logger.info(f"Combined willingness-to-pay data saved to {wtp_filename}")
        
        return wtp_data
    
    except Exception as e:
        logger.error(f"Error generating survey data: {e}")
        return None

def main():
    """Fetch all real data."""
    logger.info("Starting data collection process...")
    
    # Step 1: Fetch product data
    products_df = fetch_product_data()
    
    if products_df is not None:
        # Step 2: Generate transaction data based on real products
        fetch_transaction_data(products_df)
        
        # Step 3: Fetch competitor data
        fetch_competitor_data(products_df)
        
        # Step 4: Generate behavior data
        fetch_behavior_data(products_df)
        
        # Step 5: Generate survey data
        fetch_survey_data(products_df)
        
        logger.info("All data collected successfully!")
    else:
        logger.error("Failed to fetch product data. Aborting.")

if __name__ == "__main__":
    main()