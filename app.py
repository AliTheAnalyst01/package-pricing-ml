"""
Streamlit app for package pricing optimization dashboard.
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io

# Set page configuration
st.set_page_config(
    page_title="Package Pricing Optimizer",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_data_from_uploads():
    """Load data from user uploaded files."""
    product_prices = None
    bundle_prices = None
    test_results = None
    
    st.sidebar.header("Upload Data Files")
    
    # Allow users to upload product pricing data
    product_file = st.sidebar.file_uploader(
        "Upload Product Pricing Data (CSV)",
        type=["csv"],
        help="Upload your enhanced_product_pricing_*.csv file"
    )
    
    if product_file is not None:
        try:
            product_prices = pd.read_csv(product_file)
            st.sidebar.success(f"Loaded product pricing data: {product_file.name}")
        except Exception as e:
            st.sidebar.error(f"Error loading product data: {e}")
    
    # Allow users to upload bundle pricing data
    bundle_file = st.sidebar.file_uploader(
        "Upload Bundle Pricing Data (CSV)",
        type=["csv"],
        help="Upload your enhanced_bundle_pricing_*.csv file"
    )
    
    if bundle_file is not None:
        try:
            bundle_prices = pd.read_csv(bundle_file)
            st.sidebar.success(f"Loaded bundle pricing data: {bundle_file.name}")
        except Exception as e:
            st.sidebar.error(f"Error loading bundle data: {e}")
    
    # Allow users to upload test results data
    test_file = st.sidebar.file_uploader(
        "Upload A/B Test Results (CSV)",
        type=["csv"],
        help="Upload your price_test_*_results.csv file"
    )
    
    if test_file is not None:
        try:
            test_results = pd.read_csv(test_file)
            st.sidebar.success(f"Loaded A/B test results: {test_file.name}")
        except Exception as e:
            st.sidebar.error(f"Error loading test results: {e}")
    
    return product_prices, bundle_prices, test_results

def main():
    """Main function for the Streamlit dashboard."""
    # Header
    st.title("📊 Package Pricing Optimization Dashboard")
    
    st.markdown("""
    This dashboard displays the results of the package pricing optimization analysis.
    
    ### How to use:
    1. Upload your data files using the sidebar.
    2. Use the filters on the sidebar to explore different aspects of the pricing recommendations.
    3. View detailed insights in each section below.
    """)
    
    # Load data from uploads
    product_prices, bundle_prices, test_results = load_data_from_uploads()
    
    if product_prices is None and bundle_prices is None and test_results is None:
        st.warning("No data loaded. Please upload your pricing data files using the sidebar.")
        st.info("""
        ### Expected files:
        - **Product Pricing**: enhanced_product_pricing_YYYYMMDD.csv (from results folder)
        - **Bundle Pricing**: enhanced_bundle_pricing_YYYYMMDD.csv (from results folder)
        - **A/B Test Results**: price_test_YYYYMMDD_results.csv (from experiments folder)
        """)
        
        # Show instructions for finding files
        with st.expander("How to find your data files"):
            st.markdown("""
            The data files are located in your project folders:
            
            1. **Product Pricing Files**:
               - Path: `results/enhanced_product_pricing_*.csv`
               - These files contain your optimized product pricing recommendations
            
            2. **Bundle Pricing Files**:
               - Path: `results/enhanced_bundle_pricing_*.csv`
               - These files contain your optimized bundle discount recommendations
            
            3. **A/B Test Results Files**:
               - Path: `experiments/price_test_*_results.csv`
               - These files contain the results of your pricing A/B tests
            
            The most recent files (with the latest dates in the filename) contain your most up-to-date data.
            """)
        
        return
    
    # Sidebar filters
    st.sidebar.title("Filters")
    
    # Product section
    if product_prices is not None:
        st.header("Product Pricing Recommendations")
        
        # Determine price column names
        if 'base_price' in product_prices.columns:
            price_col = 'base_price'
        else:
            price_col = 'price'
        
        # Summary metrics
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
            
        if 'profit_change_pct' in product_prices.columns:
            col1, col2 = st.columns(2)
            with col1:
                avg_profit_impact = product_prices['profit_change_pct'].mean()
                st.metric("Average Profit Impact", f"{avg_profit_impact:.1f}%")
            
            with col2:
                if 'recommend_change' in product_prices.columns:
                    recommended = product_prices[product_prices['recommend_change']].shape[0]
                    st.metric("Recommended Changes", f"{recommended} ({recommended/len(product_prices)*100:.1f}%)")
        
        # Price change distribution
        fig = px.histogram(
            product_prices, 
            x='price_change_pct',
            title='Distribution of Recommended Price Changes',
            labels={'price_change_pct': 'Price Change (%)'},
            color_discrete_sequence=['#3498db']
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
        
        # Elasticity vs Price Change
        if 'elasticity' in product_prices.columns:
            fig = px.scatter(
                product_prices, 
                x='elasticity', 
                y='price_change_pct',
                hover_data=['product_name', price_col, 'optimal_price'],
                title='Price Elasticity vs. Recommended Price Change',
                labels={
                    'elasticity': 'Price Elasticity', 
                    'price_change_pct': 'Recommended Price Change (%)',
                    'product_name': 'Product',
                    price_col: 'Current Price',
                    'optimal_price': 'Recommended Price'
                },
                color='recommend_change' if 'recommend_change' in product_prices.columns else None,
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
        
        # Product table
        st.subheader("Product Details")
        
        # Add price change filter
        price_change_filter = st.sidebar.radio(
            "Filter by Price Change",
            ["All", "Price Increases Only", "Price Decreases Only"]
        )
        
        filtered_products = product_prices.copy()
        
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
            filtered_products = filtered_products.sort_values(price_col)
        elif sort_by == "Price (High to Low)":
            filtered_products = filtered_products.sort_values(price_col, ascending=False)
        elif sort_by == "Price Change (Highest)":
            filtered_products = filtered_products.sort_values('price_change_pct', ascending=False)
        elif sort_by == "Price Change (Lowest)":
            filtered_products = filtered_products.sort_values('price_change_pct')
        
        # Display columns
        display_cols = ['product_name', price_col, 'optimal_price', 'price_change_pct']
        
        # Add profit impact if available
        if 'profit_change_pct' in filtered_products.columns:
            display_cols.append('profit_change_pct')
        
        # Create a formatted dataframe for display
        display_df = filtered_products[display_cols].copy()
        
        # Format columns
        display_df = display_df.rename(columns={
            price_col: 'Current Price',
            'optimal_price': 'Recommended Price',
            'price_change_pct': 'Price Change (%)',
            'profit_change_pct': 'Profit Impact (%)'
        })
        
        # Format price columns as currency
        for col in ['Current Price', 'Recommended Price']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"${x:.2f}")
        
        # Format percentage columns
        for col in ['Price Change (%)', 'Profit Impact (%)']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:+.1f}%" if x != 0 else "0.0%")
        
        st.dataframe(display_df, use_container_width=True)
    
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
        
        # Bundle discount distribution
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
        
        # Create a formatted dataframe for display
        display_cols = ['bundle_name', 'num_products', 'total_individual_price', 
                       'discount_pct', 'optimal_bundle_price', 'savings']
        
        display_df = sorted_bundles[display_cols].copy()
        
        # Format columns
        display_df = display_df.rename(columns={
            'bundle_name': 'Bundle Name',
            'num_products': 'Number of Products',
            'total_individual_price': 'Total Individual Price',
            'discount_pct': 'Discount (%)',
            'optimal_bundle_price': 'Bundle Price',
            'savings': 'Customer Savings'
        })
        
        # Format price columns as currency
        for col in ['Total Individual Price', 'Bundle Price', 'Customer Savings']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"${x:.2f}")
        
        # Format percentage columns
        if 'Discount (%)' in display_df.columns:
            display_df['Discount (%)'] = display_df['Discount (%)'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(display_df, use_container_width=True)
    
    # A/B Testing Results section
    if test_results is not None:
        st.header("A/B Test Results")
        
        # Check if results have recommendation column
        if 'recommendation' in test_results.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                recommended = sum(test_results['recommendation'] == 'Implement new price')
                st.metric("Successful Tests", f"{recommended} out of {len(test_results)}")
            
            with col2:
                if 'revenue_lift' in test_results.columns:
                    avg_revenue_lift = test_results['revenue_lift'].mean()
                    st.metric("Average Revenue Lift", f"{avg_revenue_lift:.1f}%")
            
            # Test results table
            st.subheader("Test Results Details")
            
            # Create a formatted dataframe for display
            if 'product_name' in test_results.columns and 'control_price' in test_results.columns:
                display_cols = ['product_name', 'control_price', 'test_price', 'revenue_lift', 'recommendation']
                
                # Only display columns that exist
                display_cols = [col for col in display_cols if col in test_results.columns]
                
                display_df = test_results[display_cols].copy()
                
                # Format columns
                rename_dict = {
                    'product_name': 'Product Name',
                    'control_price': 'Control Price',
                    'test_price': 'Test Price',
                    'revenue_lift': 'Revenue Lift (%)',
                    'recommendation': 'Recommendation'
                }
                
                # Only rename columns that exist
                rename_dict = {k: v for k, v in rename_dict.items() if k in display_cols}
                
                display_df = display_df.rename(columns=rename_dict)
                
                # Format price columns as currency
                for col in ['Control Price', 'Test Price']:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"${x:.2f}")
                
                # Format percentage columns
                if 'Revenue Lift (%)' in display_df.columns:
                    display_df['Revenue Lift (%)'] = display_df['Revenue Lift (%)'].apply(lambda x: f"{x:+.1f}%")
                
                st.dataframe(display_df, use_container_width=True)
    
    # Download buttons for recommended prices
    if product_prices is not None or bundle_prices is not None:
        st.header("Export Optimized Pricing")
        
        col1, col2 = st.columns(2)
        
        if product_prices is not None:
            with col1:
                csv_product = product_prices.to_csv(index=False)
                st.download_button(
                    label="Download Product Pricing CSV",
                    data=csv_product,
                    file_name=f"product_pricing_export_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        if bundle_prices is not None:
            with col2:
                csv_bundle = bundle_prices.to_csv(index=False)
                st.download_button(
                    label="Download Bundle Pricing CSV",
                    data=csv_bundle,
                    file_name=f"bundle_pricing_export_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center">
        <p>Package Pricing ML Optimization - Version 1.0</p>
        <p>Last updated: {}</p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d")), unsafe_allow_html=True)

if __name__ == "__main__":
    main()