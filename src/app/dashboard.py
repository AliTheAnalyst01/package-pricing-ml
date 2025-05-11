"""
Simple Streamlit dashboard for package pricing optimization results.
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

def get_latest_file(directory, prefix):
    """Get the most recent file with a given prefix in a directory."""
    files = [f for f in os.listdir(directory) if f.startswith(prefix)]
    if not files:
        return None
    return os.path.join(directory, sorted(files)[-1])

def load_data():
    """Load the latest pricing recommendations."""
    # Load product pricing recommendations
    product_file = get_latest_file('results', 'product_pricing_recommendations_')
    if product_file:
        product_prices = pd.read_csv(product_file)
    else:
        product_prices = None
    
    # Load bundle pricing recommendations
    bundle_file = get_latest_file('results', 'bundle_pricing_recommendations_')
    if bundle_file:
        bundle_prices = pd.read_csv(bundle_file)
    else:
        bundle_prices = None
    
    return product_prices, bundle_prices

def main():
    """Main function for the Streamlit dashboard."""
    st.set_page_config(
        page_title="Package Pricing Optimization Dashboard",
        page_icon="💰",
        layout="wide"
    )
    
    st.title("📊 Package Pricing Optimization Dashboard")
    st.markdown("""
    This dashboard displays the results of the package pricing optimization analysis.
    Use the filters on the sidebar to explore different aspects of the pricing recommendations.
    """)
    
    # Load data
    product_prices, bundle_prices = load_data()
    
    if product_prices is None and bundle_prices is None:
        st.error("No pricing data found. Please run the optimization script first.")
        return
    
    # Sidebar filters
    st.sidebar.title("Filters")
    
    # Product section
    if product_prices is not None:
        st.header("Product Pricing Recommendations")
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_price_change = product_prices['price_change_pct'].mean()
            st.metric("Average Price Change", f"{avg_price_change:.1f}%")
        
        with col2:
            price_increases = (product_prices['price_change_pct'] > 0).sum()
            st.metric("Products with Price Increases", f"{price_increases} ({price_increases/len(product_prices)*100:.1f}%)")
        
        with col3:
            price_decreases = (product_prices['price_change_pct'] < 0).sum()
            st.metric("Products with Price Decreases", f"{price_decreases} ({price_decreases/len(product_prices)*100:.1f}%)")
        
        # Price change distribution chart
        fig = px.histogram(
            product_prices, 
            x='price_change_pct',
            title='Distribution of Recommended Price Changes',
            labels={'price_change_pct': 'Price Change (%)'},
            color_discrete_sequence=['#3498db']
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
        
        # Category-wise price changes
        if 'category' in product_prices.columns:
            category_changes = product_prices.groupby('category')['price_change_pct'].mean().reset_index()
            fig = px.bar(
                category_changes, 
                x='category', 
                y='price_change_pct',
                title='Average Price Change by Category',
                labels={'price_change_pct': 'Average Price Change (%)', 'category': 'Category'},
                color_discrete_sequence=['#2ecc71']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Elasticity vs. price change
        fig = px.scatter(
            product_prices, 
            x='elasticity', 
            y='price_change_pct',
            hover_data=['product_name', 'base_price', 'optimal_price'],
            title='Price Elasticity vs. Recommended Price Change',
            labels={
                'elasticity': 'Price Elasticity', 
                'price_change_pct': 'Recommended Price Change (%)',
                'product_name': 'Product',
                'base_price': 'Current Price',
                'optimal_price': 'Recommended Price'
            },
            color='category' if 'category' in product_prices.columns else None,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
        
        # Product table
        st.subheader("Product Details")
        
        # Add filter for category if it exists
        if 'category' in product_prices.columns:
            categories = ['All'] + sorted(product_prices['category'].unique().tolist())
            selected_category = st.sidebar.selectbox("Filter by Category", categories)
            
            if selected_category != 'All':
                filtered_products = product_prices[product_prices['category'] == selected_category]
            else:
                filtered_products = product_prices
        else:
            filtered_products = product_prices
        
        # Add price change filter
        price_change_filter = st.sidebar.radio(
            "Filter by Price Change",
            ["All", "Price Increases Only", "Price Decreases Only"]
        )
        
        if price_change_filter == "Price Increases Only":
            filtered_products = filtered_products[filtered_products['price_change_pct'] > 0]
        elif price_change_filter == "Price Decreases Only":
            filtered_products = filtered_products[filtered_products['price_change_pct'] < 0]
        
        # Sort option
        sort_by = st.sidebar.selectbox(
            "Sort Products By",
            ["Name (A-Z)", "Price (Low to High)", "Price (High to Low)", 
             "Price Change (Highest)", "Price Change (Lowest)"]
        )
        
        if sort_by == "Name (A-Z)":
            filtered_products = filtered_products.sort_values('product_name')
        elif sort_by == "Price (Low to High)":
            filtered_products = filtered_products.sort_values('base_price')
        elif sort_by == "Price (High to Low)":
            filtered_products = filtered_products.sort_values('base_price', ascending=False)
        elif sort_by == "Price Change (Highest)":
            filtered_products = filtered_products.sort_values('price_change_pct', ascending=False)
        elif sort_by == "Price Change (Lowest)":
            filtered_products = filtered_products.sort_values('price_change_pct')
        
        # Display table with styling
        display_cols = ['product_name', 'category', 'base_price', 'optimal_price', 'price_change_pct']
        display_cols = [col for col in display_cols if col in filtered_products.columns]
        
        st.dataframe(
            filtered_products[display_cols].style.format({
                'base_price': '${:.2f}',
                'optimal_price': '${:.2f}',
                'price_change_pct': '{:.1f}%'
            }),
            height=400
        )
    
    # Bundle section
    if bundle_prices is not None:
        st.header("Bundle Pricing Recommendations")
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_discount = bundle_prices['discount_pct'].mean()
            st.metric("Average Bundle Discount", f"{avg_discount:.1f}%")
        
        with col2:
            avg_savings = bundle_prices['savings'].mean()
            st.metric("Average Customer Savings", f"${avg_savings:.2f}")
        
        with col3:
            total_bundles = len(bundle_prices)
            st.metric("Total Bundles", f"{total_bundles}")
        
        # Bundle discount distribution chart
        fig = px.histogram(
            bundle_prices, 
            x='discount_pct',
            title='Distribution of Bundle Discounts',
            labels={'discount_pct': 'Discount (%)'},
            color_discrete_sequence=['#9b59b6']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Bundle size vs. discount
        fig = px.scatter(
            bundle_prices, 
            x='num_products', 
            y='discount_pct',
            size='total_individual_price',
            hover_data=['bundle_name', 'optimal_bundle_price', 'savings'],
            title='Bundle Size vs. Discount Percentage',
            labels={
                'num_products': 'Number of Products in Bundle', 
                'discount_pct': 'Discount (%)',
                'bundle_name': 'Bundle',
                'total_individual_price': 'Total Individual Price',
                'optimal_bundle_price': 'Bundle Price',
                'savings': 'Customer Savings'
            },
            color_discrete_sequence=['#e74c3c']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Bundle table
        st.subheader("Bundle Details")
        
        # Sort bundles
        sort_bundles_by = st.sidebar.selectbox(
            "Sort Bundles By",
            ["Name (A-Z)", "Discount (Highest)", "Discount (Lowest)", 
             "Savings (Highest)", "Number of Products"]
        )
        
        if sort_bundles_by == "Name (A-Z)":
            sorted_bundles = bundle_prices.sort_values('bundle_name')
        elif sort_bundles_by == "Discount (Highest)":
            sorted_bundles = bundle_prices.sort_values('discount_pct', ascending=False)
        elif sort_bundles_by == "Discount (Lowest)":
            sorted_bundles = bundle_prices.sort_values('discount_pct')
        elif sort_bundles_by == "Savings (Highest)":
            sorted_bundles = bundle_prices.sort_values('savings', ascending=False)
        elif sort_bundles_by == "Number of Products":
            sorted_bundles = bundle_prices.sort_values('num_products', ascending=False)
        
        # Display table with styling
        display_cols = ['bundle_name', 'num_products', 'total_individual_price', 
                         'optimal_bundle_price', 'discount_pct', 'savings']
        display_cols = [col for col in display_cols if col in sorted_bundles.columns]
        
        st.dataframe(
            sorted_bundles[display_cols].style.format({
                'total_individual_price': '${:.2f}',
                'optimal_bundle_price': '${:.2f}',
                'discount_pct': '{:.1f}%',
                'savings': '${:.2f}'
            }),
            height=400
        )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center">
        <p>Package Pricing ML Project Dashboard</p>
        <p>Generated on: {}</p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

if __name__ == "__main__":
    main()