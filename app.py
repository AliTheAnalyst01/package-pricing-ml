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
        # Inspect columns and map to expected column names
        df_columns = product_prices.columns.str.lower()
        
        # Map for price change percentage
        price_change_cols = [col for col in df_columns if 'price_change' in col or 'change_pct' in col]
        price_change_col = price_change_cols[0] if price_change_cols else None
        
        # Map for price columns
        if 'base_price' in product_prices.columns:
            price_col = 'base_price'
        elif 'price' in product_prices.columns:
            price_col = 'price'
        else:
            price_candidates = [col for col in product_prices.columns if 'price' in col.lower() and 'change' not in col.lower() and 'optimal' not in col.lower()]
            price_col = price_candidates[0] if price_candidates else None
        
        # Map for optimal price
        if 'optimal_price' in product_prices.columns:
            optimal_price_col = 'optimal_price'
        else:
            optimal_candidates = [col for col in product_prices.columns if 'optimal' in col.lower() and 'price' in col.lower()]
            optimal_price_col = optimal_candidates[0] if optimal_candidates else None
        
        # Map for profit change
        profit_change_cols = [col for col in df_columns if 'profit' in col and 'change' in col]
        profit_change_col = profit_change_cols[0] if profit_change_cols else None
        
        # Map for recommendation
        recommend_cols = [col for col in df_columns if 'recommend' in col]
        recommend_col = recommend_cols[0] if recommend_cols else None
        
        # Map for elasticity
        elasticity_cols = [col for col in df_columns if 'elastic' in col and 'source' not in col]
        elasticity_col = elasticity_cols[0] if elasticity_cols else None
        
        # Check if we have minimum required columns
        if price_col and price_change_col:
            st.header("Product Pricing Recommendations")
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_price_change = product_prices[price_change_col].mean()
                st.metric("Average Price Change", f"{avg_price_change:.1f}%")
            
            with col2:
                price_increases = (product_prices[price_change_col] > 0).sum()
                st.metric("Products with Price Increases", f"{price_increases} ({price_increases/len(product_prices)*100:.1f}%)")
            
            with col3:
                price_decreases = (product_prices[price_change_col] < 0).sum()
                st.metric("Products with Price Decreases", f"{price_decreases} ({price_decreases/len(product_prices)*100:.1f}%)")
                
            if profit_change_col:
                col1, col2 = st.columns(2)
                with col1:
                    avg_profit_impact = product_prices[profit_change_col].mean()
                    st.metric("Average Profit Impact", f"{avg_profit_impact:.1f}%")
                
                with col2:
                    if recommend_col:
                        recommended = product_prices[recommend_col].sum() if product_prices[recommend_col].dtype == bool else None
                        if recommended is not None:
                            st.metric("Recommended Changes", f"{recommended} ({recommended/len(product_prices)*100:.1f}%)")
            
            # Price change distribution
            fig = px.histogram(
                product_prices, 
                x=price_change_col,
                title='Distribution of Recommended Price Changes',
                labels={price_change_col: 'Price Change (%)'},
                color_discrete_sequence=['#3498db']
            )
            fig.add_vline(x=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
            
            # Elasticity vs Price Change
            if elasticity_col:
                fig = px.scatter(
                    product_prices, 
                    x=elasticity_col, 
                    y=price_change_col,
                    hover_data=['product_name', price_col] + ([optimal_price_col] if optimal_price_col else []),
                    title='Price Elasticity vs. Recommended Price Change',
                    labels={
                        elasticity_col: 'Price Elasticity', 
                        price_change_col: 'Recommended Price Change (%)',
                        'product_name': 'Product',
                        price_col: 'Current Price',
                        optimal_price_col: 'Recommended Price' if optimal_price_col else None
                    },
                    color=recommend_col if recommend_col else None,
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
                filtered_products = filtered_products[filtered_products[price_change_col] > 0]
            elif price_change_filter == "Price Decreases Only":
                filtered_products = filtered_products[filtered_products[price_change_col] < 0]
            
            # Sort option
            sort_by = st.sidebar.selectbox(
                "Sort Products By",
                ["Name (A-Z)", "Price (Low to High)", "Price (High to Low)", 
                 "Price Change (Highest)", "Price Change (Lowest)"]
            )
            
            if sort_by == "Name (A-Z)":
                filtered_products = filtered_products.sort_values('product_name')
            elif sort_by == "Price (Low to High)" and price_col:
                filtered_products = filtered_products.sort_values(price_col)
            elif sort_by == "Price (High to Low)" and price_col:
                filtered_products = filtered_products.sort_values(price_col, ascending=False)
            elif sort_by == "Price Change (Highest)" and price_change_col:
                filtered_products = filtered_products.sort_values(price_change_col, ascending=False)
            elif sort_by == "Price Change (Lowest)" and price_change_col:
                filtered_products = filtered_products.sort_values(price_change_col)
            
            # Display columns
            display_cols = ['product_name']
            if price_col:
                display_cols.append(price_col)
            if optimal_price_col:
                display_cols.append(optimal_price_col)
            if price_change_col:
                display_cols.append(price_change_col)
            if profit_change_col:
                display_cols.append(profit_change_col)
            
            # Only include columns that exist
            display_cols = [col for col in display_cols if col in filtered_products.columns]
            
            # Create a formatted dataframe for display
            display_df = filtered_products[display_cols].copy()
            
            # Format columns
            rename_dict = {
                'product_name': 'Product Name'
            }
            if price_col:
                rename_dict[price_col] = 'Current Price'
            if optimal_price_col:
                rename_dict[optimal_price_col] = 'Recommended Price'
            if price_change_col:
                rename_dict[price_change_col] = 'Price Change (%)'
            if profit_change_col:
                rename_dict[profit_change_col] = 'Profit Impact (%)'
                
            display_df = display_df.rename(columns=rename_dict)
            
            # Format price columns as currency
            for col in ['Current Price', 'Recommended Price']:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"${x:.2f}")
            
            # Format percentage columns
            for col in ['Price Change (%)', 'Profit Impact (%)']:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:+.1f}%" if x != 0 else "0.0%")
            
            st.dataframe(display_df, use_container_width=True)
        else:
            st.warning("Product pricing data doesn't contain the expected columns. Please check your file.")
    
    # Bundle section
    if bundle_prices is not None:
        # Inspect bundle columns
        df_columns = bundle_prices.columns.str.lower()
        
        # Check if required columns exist
        discount_cols = [col for col in df_columns if 'discount' in col]
        discount_col = discount_cols[0] if discount_cols else None
        
        savings_cols = [col for col in df_columns if 'saving' in col]
        savings_col = savings_cols[0] if savings_cols else None
        
        num_products_cols = [col for col in df_columns if 'num' in col and 'product' in col]
        num_products_col = num_products_cols[0] if num_products_cols else None
        
        bundle_name_cols = [col for col in df_columns if 'bundle' in col and 'name' in col]
        bundle_name_col = bundle_name_cols[0] if bundle_name_cols else None
        
        total_price_cols = [col for col in df_columns if 'total' in col and 'price' in col]
        total_price_col = total_price_cols[0] if total_price_cols else None
        
        bundle_price_cols = [col for col in df_columns if ('bundle' in col or 'optimal' in col) and 'price' in col]
        bundle_price_col = bundle_price_cols[0] if bundle_price_cols else None
        
        if discount_col and bundle_name_col:
            st.header("Bundle Pricing Recommendations")
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_discount = bundle_prices[discount_col].mean()
                st.metric("Average Bundle Discount", f"{avg_discount:.1f}%")
            
            with col2:
                if savings_col:
                    avg_savings = bundle_prices[savings_col].mean()
                    st.metric("Average Customer Savings", f"${avg_savings:.2f}")
            
            with col3:
                total_bundles = len(bundle_prices)
                st.metric("Total Bundles", f"{total_bundles}")
            
            # Bundle discount distribution
            if discount_col:
                fig = px.histogram(
                    bundle_prices, 
                    x=discount_col,
                    title='Distribution of Bundle Discounts',
                    labels={discount_col: 'Discount (%)'},
                    color_discrete_sequence=['#9b59b6']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Bundle size vs. discount
            if num_products_col and discount_col:
                hover_data = [bundle_name_col]
                if bundle_price_col:
                    hover_data.append(bundle_price_col)
                if savings_col:
                    hover_data.append(savings_col)
                
                fig = px.scatter(
                    bundle_prices, 
                    x=num_products_col, 
                    y=discount_col,
                    size=total_price_col if total_price_col else None,
                    hover_data=hover_data,
                    title='Bundle Size vs. Discount Percentage',
                    labels={
                        num_products_col: 'Number of Products in Bundle', 
                        discount_col: 'Discount (%)',
                        bundle_name_col: 'Bundle',
                        total_price_col: 'Total Individual Price' if total_price_col else None,
                        bundle_price_col: 'Bundle Price' if bundle_price_col else None,
                        savings_col: 'Customer Savings' if savings_col else None
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
            
            sorted_bundles = bundle_prices.copy()
            
            if sort_bundles_by == "Name (A-Z)" and bundle_name_col:
                sorted_bundles = sorted_bundles.sort_values(bundle_name_col)
            elif sort_bundles_by == "Discount (Highest)" and discount_col:
                sorted_bundles = sorted_bundles.sort_values(discount_col, ascending=False)
            elif sort_bundles_by == "Discount (Lowest)" and discount_col:
                sorted_bundles = sorted_bundles.sort_values(discount_col)
            elif sort_bundles_by == "Savings (Highest)" and savings_col:
                sorted_bundles = sorted_bundles.sort_values(savings_col, ascending=False)
            elif sort_bundles_by == "Number of Products" and num_products_col:
                sorted_bundles = sorted_bundles.sort_values(num_products_col, ascending=False)
            
            # Create a formatted dataframe for display
            display_cols = []
            if bundle_name_col:
                display_cols.append(bundle_name_col)
            if num_products_col:
                display_cols.append(num_products_col)
            if total_price_col:
                display_cols.append(total_price_col)
            if discount_col:
                display_cols.append(discount_col)
            if bundle_price_col:
                display_cols.append(bundle_price_col)
            if savings_col:
                display_cols.append(savings_col)
            
            # Only include columns that exist
            display_cols = [col for col in display_cols if col in sorted_bundles.columns]
            
            if display_cols:
                display_df = sorted_bundles[display_cols].copy()
                
                # Format columns
                rename_dict = {}
                if bundle_name_col:
                    rename_dict[bundle_name_col] = 'Bundle Name'
                if num_products_col:
                    rename_dict[num_products_col] = 'Number of Products'
                if total_price_col:
                    rename_dict[total_price_col] = 'Total Individual Price'
                if discount_col:
                    rename_dict[discount_col] = 'Discount (%)'
                if bundle_price_col:
                    rename_dict[bundle_price_col] = 'Bundle Price'
                if savings_col:
                    rename_dict[savings_col] = 'Customer Savings'
                
                display_df = display_df.rename(columns=rename_dict)
                
                # Format price columns as currency
                for col in ['Total Individual Price', 'Bundle Price', 'Customer Savings']:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"${x:.2f}")
                
                # Format percentage columns
                if 'Discount (%)' in display_df.columns:
                    display_df['Discount (%)'] = display_df['Discount (%)'].apply(lambda x: f"{x:.1f}%")
                
                st.dataframe(display_df, use_container_width=True)
            else:
                st.warning("Bundle pricing data doesn't contain enough columns for display.")
        else:
            st.warning("Bundle pricing data doesn't contain the expected columns. Please check your file.")
    
    # A/B Testing Results section
    if test_results is not None:
        # Inspect test results columns
        df_columns = test_results.columns.str.lower()
        
        # Check if required columns exist
        recommendation_cols = [col for col in df_columns if 'recommend' in col]
        recommendation_col = recommendation_cols[0] if recommendation_cols else None
        
        product_name_cols = [col for col in df_columns if 'product' in col and 'name' in col]
        product_name_col = product_name_cols[0] if product_name_cols else None
        
        control_price_cols = [col for col in df_columns if 'control' in col and 'price' in col]
        control_price_col = control_price_cols[0] if control_price_cols else None
        
        test_price_cols = [col for col in df_columns if 'test' in col and 'price' in col]
        test_price_col = test_price_cols[0] if test_price_cols else None
        
        revenue_lift_cols = [col for col in df_columns if 'revenue' in col and ('lift' in col or 'change' in col)]
        revenue_lift_col = revenue_lift_cols[0] if revenue_lift_cols else None
        
        if recommendation_col or (product_name_col and control_price_col):
            st.header("A/B Test Results")
            
            if recommendation_col:
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'Implement new price' in test_results[recommendation_col].values:
                        recommended = sum(test_results[recommendation_col] == 'Implement new price')
                        st.metric("Successful Tests", f"{recommended} out of {len(test_results)}")
                    else:
                        # Try to find a success indicator
                        st.text("Test Results Summary")
                
                with col2:
                    if revenue_lift_col:
                        avg_revenue_lift = test_results[revenue_lift_col].mean()
                        st.metric("Average Revenue Lift", f"{avg_revenue_lift:.1f}%")
            
            # Test results table
            st.subheader("Test Results Details")
            
            # Create a formatted dataframe for display
            if product_name_col or control_price_col:
                display_cols = []
                if product_name_col:
                    display_cols.append(product_name_col)
                if control_price_col:
                    display_cols.append(control_price_col)
                if test_price_col:
                    display_cols.append(test_price_col)
                if revenue_lift_col:
                    display_cols.append(revenue_lift_col)
                if recommendation_col:
                    display_cols.append(recommendation_col)
                
                # Only include columns that exist
                display_cols = [col for col in display_cols if col in test_results.columns]
                
                if display_cols:
                    display_df = test_results[display_cols].copy()
                    
                    # Format columns
                    rename_dict = {}
                    if product_name_col:
                        rename_dict[product_name_col] = 'Product Name'
                    if control_price_col:
                        rename_dict[control_price_col] = 'Control Price'
                    if test_price_col:
                        rename_dict[test_price_col] = 'Test Price'
                    if revenue_lift_col:
                        rename_dict[revenue_lift_col] = 'Revenue Lift (%)'
                    if recommendation_col:
                        rename_dict[recommendation_col] = 'Recommendation'
                    
                    display_df = display_df.rename(columns=rename_dict)
                    
                    # Format price columns as currency
                    for col in ['Control Price', 'Test Price']:
                        if col in display_df.columns:
                            display_df[col] = display_df[col].apply(lambda x: f"${x:.2f}")
                    
                    # Format percentage columns
                    if 'Revenue Lift (%)' in display_df.columns:
                        display_df['Revenue Lift (%)'] = display_df['Revenue Lift (%)'].apply(lambda x: f"{x:+.1f}%")
                    
                    st.dataframe(display_df, use_container_width=True)
                else:
                    st.warning("Test results don't contain enough columns for display.")
            else:
                st.warning("Test results don't contain the expected columns. Please check your file.")
    
    # Download buttons for pricing data
    if product_prices is not None or bundle_prices is not None:
        st.header("Export Pricing Data")
        
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