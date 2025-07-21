# E-commerce Sales Analysis & Customer Segmentation
# Professional Data Science Project Implementation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Statistical analysis
from scipy import stats
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Set style for visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
np.random.seed(42)

class EcommerceAnalyzer:
    """
    Professional E-commerce Data Analysis Class
    Handles data preprocessing, EDA, customer segmentation, and predictive modeling
    """
    
    def __init__(self):
        self.data = None
        self.rfm_data = None
        self.customer_segments = None
        self.scaler = StandardScaler()
        
    def generate_synthetic_data(self, n_customers=1000, n_transactions=10000):
        """Generate realistic synthetic e-commerce data"""
        np.random.seed(42)
        
        # Customer demographics
        customers = pd.DataFrame({
            'customer_id': range(1, n_customers + 1),
            'registration_date': pd.date_range('2022-01-01', '2024-01-01', freq='D')[:n_customers],
            'customer_age': np.random.normal(35, 12, n_customers).astype(int),
            'customer_segment': np.random.choice(['Premium', 'Regular', 'Budget'], n_customers, p=[0.2, 0.5, 0.3])
        })
        
        # Product categories
        categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books', 'Beauty', 'Toys']
        
        # Generate transactions
        transactions = []
        
        for i in range(n_transactions):
            customer_id = np.random.choice(customers['customer_id'])
            customer_info = customers[customers['customer_id'] == customer_id].iloc[0]
            
            # Customer behavior patterns
            if customer_info['customer_segment'] == 'Premium':
                avg_amount = np.random.gamma(2, 150)
                category_prefs = ['Electronics', 'Beauty', 'Clothing']
            elif customer_info['customer_segment'] == 'Regular':
                avg_amount = np.random.gamma(2, 75)
                category_prefs = ['Clothing', 'Home & Garden', 'Sports']
            else:
                avg_amount = np.random.gamma(2, 35)
                category_prefs = ['Books', 'Toys', 'Clothing']
            
            # Transaction details
            transaction_date = customer_info['registration_date'] + timedelta(
                days=np.random.randint(0, (datetime(2024, 7, 1) - customer_info['registration_date']).days)
            )
            
            category = np.random.choice(category_prefs + categories, p=[0.3, 0.25, 0.25] + [0.2/4]*4)
            quantity = np.random.poisson(2) + 1
            unit_price = max(10, avg_amount / quantity)
            
            transactions.append({
                'transaction_id': f'T{i+1:06d}',
                'customer_id': customer_id,
                'transaction_date': transaction_date,
                'product_category': category,
                'quantity': quantity,
                'unit_price': unit_price,
                'total_amount': quantity * unit_price,
                'discount_applied': np.random.choice([0, 0.05, 0.1, 0.15], p=[0.6, 0.2, 0.15, 0.05])
            })
        
        self.data = pd.DataFrame(transactions)
        self.data['final_amount'] = self.data['total_amount'] * (1 - self.data['discount_applied'])
        self.data['transaction_date'] = pd.to_datetime(self.data['transaction_date'])
        
        # Add some missing values for realistic data cleaning
        missing_indices = np.random.choice(self.data.index, size=int(0.02 * len(self.data)), replace=False)
        self.data.loc[missing_indices, 'discount_applied'] = np.nan
        
        return self.data
    
    def data_cleaning_preprocessing(self):
        """Clean and preprocess the data"""
        print("=== DATA CLEANING & PREPROCESSING ===")
        print(f"Original dataset shape: {self.data.shape}")
        
        # Handle missing values
        missing_before = self.data.isnull().sum().sum()
        self.data['discount_applied'].fillna(0, inplace=True)
        print(f"Missing values handled: {missing_before} → {self.data.isnull().sum().sum()}")
        
        # Remove outliers (transactions > 3 std deviations from mean)
        q1 = self.data['final_amount'].quantile(0.25)
        q3 = self.data['final_amount'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = len(self.data[(self.data['final_amount'] < lower_bound) | 
                                (self.data['final_amount'] > upper_bound)])
        self.data = self.data[(self.data['final_amount'] >= lower_bound) & 
                             (self.data['final_amount'] <= upper_bound)]
        
        print(f"Outliers removed: {outliers}")
        print(f"Final dataset shape: {self.data.shape}")
        
        # Feature engineering
        self.data['year'] = self.data['transaction_date'].dt.year
        self.data['month'] = self.data['transaction_date'].dt.month
        self.data['quarter'] = self.data['transaction_date'].dt.quarter
        self.data['day_of_week'] = self.data['transaction_date'].dt.dayofweek
        
        return self.data
    
    def exploratory_data_analysis(self):
        """Comprehensive EDA with visualizations"""
        print("\n=== EXPLORATORY DATA ANALYSIS ===")
        
        # Basic statistics
        print(f"Analysis Period: {self.data['transaction_date'].min()} to {self.data['transaction_date'].max()}")
        print(f"Total Revenue: ${self.data['final_amount'].sum():,.2f}")
        print(f"Total Transactions: {len(self.data):,}")
        print(f"Unique Customers: {self.data['customer_id'].nunique():,}")
        print(f"Average Order Value: ${self.data['final_amount'].mean():.2f}")
        
        # Create comprehensive visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('E-commerce Business Analytics Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Revenue trend over time
        monthly_revenue = self.data.groupby(['year', 'month'])['final_amount'].sum().reset_index()
        monthly_revenue['date'] = pd.to_datetime(monthly_revenue[['year', 'month']].assign(day=1))
        
        axes[0, 0].plot(monthly_revenue['date'], monthly_revenue['final_amount'], marker='o', linewidth=2)
        axes[0, 0].set_title('Monthly Revenue Trend', fontweight='bold')
        axes[0, 0].set_ylabel('Revenue ($)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Category performance
        category_revenue = self.data.groupby('product_category')['final_amount'].sum().sort_values(ascending=True)
        axes[0, 1].barh(category_revenue.index, category_revenue.values)
        axes[0, 1].set_title('Revenue by Product Category', fontweight='bold')
        axes[0, 1].set_xlabel('Revenue ($)')
        
        # 3. Order value distribution
        axes[0, 2].hist(self.data['final_amount'], bins=50, alpha=0.7, edgecolor='black')
        axes[0, 2].axvline(self.data['final_amount'].mean(), color='red', linestyle='--', 
                          label=f'Mean: ${self.data["final_amount"].mean():.2f}')
        axes[0, 2].set_title('Order Value Distribution', fontweight='bold')
        axes[0, 2].set_xlabel('Order Value ($)')
        axes[0, 2].legend()
        
        # 4. Seasonal patterns
        seasonal_data = self.data.groupby('quarter')['final_amount'].agg(['sum', 'count']).reset_index()
        seasonal_data['avg_order'] = seasonal_data['sum'] / seasonal_data['count']
        
        ax_twin = axes[1, 0].twinx()
        bars = axes[1, 0].bar(seasonal_data['quarter'], seasonal_data['sum'], alpha=0.7, label='Total Revenue')
        line = ax_twin.plot(seasonal_data['quarter'], seasonal_data['avg_order'], color='red', marker='o', 
                           linewidth=2, label='Avg Order Value')
        
        axes[1, 0].set_title('Quarterly Performance', fontweight='bold')
        axes[1, 0].set_xlabel('Quarter')
        axes[1, 0].set_ylabel('Total Revenue ($)')
        ax_twin.set_ylabel('Average Order Value ($)')
        
        # 5. Customer purchase frequency
        customer_freq = self.data.groupby('customer_id').size()
        axes[1, 1].hist(customer_freq, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(customer_freq.mean(), color='red', linestyle='--',
                          label=f'Mean: {customer_freq.mean():.1f} orders')
        axes[1, 1].set_title('Customer Purchase Frequency', fontweight='bold')
        axes[1, 1].set_xlabel('Number of Orders')
        axes[1, 1].legend()
        
        # 6. Day of week analysis
        dow_data = self.data.groupby('day_of_week')['final_amount'].agg(['sum', 'mean']).reset_index()
        dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        dow_data['day_name'] = [dow_names[i] for i in dow_data['day_of_week']]
        
        axes[1, 2].bar(dow_data['day_name'], dow_data['sum'])
        axes[1, 2].set_title('Revenue by Day of Week', fontweight='bold')
        axes[1, 2].set_ylabel('Total Revenue ($)')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return monthly_revenue, category_revenue, seasonal_data
    
    def rfm_analysis(self):
        """RFM (Recency, Frequency, Monetary) Analysis"""
        print("\n=== RFM ANALYSIS ===")
        
        # Calculate RFM metrics
        analysis_date = self.data['transaction_date'].max() + timedelta(days=1)
        
        rfm = self.data.groupby('customer_id').agg({
            'transaction_date': lambda x: (analysis_date - x.max()).days,  # Recency
            'transaction_id': 'count',  # Frequency
            'final_amount': 'sum'  # Monetary
        }).reset_index()
        
        rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']
        
        # Create RFM scores
        rfm['recency_score'] = pd.qcut(rfm['recency'], q=5, labels=[5,4,3,2,1])
        rfm['frequency_score'] = pd.qcut(rfm['frequency'].rank(method='first'), q=5, labels=[1,2,3,4,5])
        rfm['monetary_score'] = pd.qcut(rfm['monetary'], q=5, labels=[1,2,3,4,5])
        
        # Combine scores
        rfm['rfm_score'] = rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str) + rfm['monetary_score'].astype(str)
        rfm['rfm_score_numeric'] = rfm['recency_score'].astype(int) + rfm['frequency_score'].astype(int) + rfm['monetary_score'].astype(int)
        
        # Customer segmentation based on RFM
        def segment_customers(row):
            if row['rfm_score_numeric'] >= 13:
                return 'Champions'
            elif row['rfm_score_numeric'] >= 11:
                return 'Loyal Customers'
            elif row['rfm_score_numeric'] >= 8:
                return 'Potential Loyalists'
            elif row['rfm_score_numeric'] >= 6:
                return 'At Risk'
            else:
                return 'Lost Customers'
        
        rfm['customer_segment'] = rfm.apply(segment_customers, axis=1)
        self.rfm_data = rfm
        
        # Segment analysis
        segment_analysis = rfm.groupby('customer_segment').agg({
            'customer_id': 'count',
            'recency': 'mean',
            'frequency': 'mean',
            'monetary': 'mean'
        }).round(2)
        
        segment_analysis['customer_percentage'] = (segment_analysis['customer_id'] / len(rfm) * 100).round(2)
        segment_analysis = segment_analysis.sort_values('monetary', ascending=False)
        
        print("Customer Segment Analysis:")
        print(segment_analysis)
        
        # Visualize RFM analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('RFM Analysis & Customer Segmentation', fontsize=16, fontweight='bold')
        
        # RFM distribution
        axes[0, 0].scatter(rfm['recency'], rfm['monetary'], alpha=0.6)
        axes[0, 0].set_xlabel('Recency (days)')
        axes[0, 0].set_ylabel('Monetary Value ($)')
        axes[0, 0].set_title('Recency vs Monetary Value')
        
        axes[0, 1].scatter(rfm['frequency'], rfm['monetary'], alpha=0.6)
        axes[0, 1].set_xlabel('Frequency (orders)')
        axes[0, 1].set_ylabel('Monetary Value ($)')
        axes[0, 1].set_title('Frequency vs Monetary Value')
        
        # Customer segments
        segment_counts = rfm['customer_segment'].value_counts()
        axes[1, 0].pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%')
        axes[1, 0].set_title('Customer Segment Distribution')
        
        # Segment value
        segment_value = rfm.groupby('customer_segment')['monetary'].sum().sort_values(ascending=True)
        axes[1, 1].barh(segment_value.index, segment_value.values)
        axes[1, 1].set_title('Revenue by Customer Segment')
        axes[1, 1].set_xlabel('Total Revenue ($)')
        
        plt.tight_layout()
        plt.show()
        
        return rfm, segment_analysis
    
    def customer_segmentation_clustering(self):
        """Advanced customer segmentation using K-Means clustering"""
        print("\n=== MACHINE LEARNING CUSTOMER SEGMENTATION ===")
        
        # Prepare features for clustering
        features = ['recency', 'frequency', 'monetary']
        X = self.rfm_data[features].copy()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Find optimal number of clusters using elbow method
        inertias = []
        k_range = range(2, 11)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
        
        # Perform clustering with optimal k=5
        optimal_k = 5
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        self.rfm_data['ml_segment'] = kmeans.fit_predict(X_scaled)
        
        # Analyze ML segments
        ml_segment_analysis = self.rfm_data.groupby('ml_segment').agg({
            'customer_id': 'count',
            'recency': 'mean',
            'frequency': 'mean',
            'monetary': 'mean'
        }).round(2)
        
        # Assign meaningful names to clusters
        cluster_names = {
            0: 'High-Value Loyalists',
            1: 'Moderate Regulars',
            2: 'New Customers',
            3: 'At-Risk Customers',
            4: 'Lost Customers'
        }
        
        ml_segment_analysis['segment_name'] = [cluster_names[i] for i in ml_segment_analysis.index]
        
        print("Machine Learning Segmentation Results:")
        print(ml_segment_analysis)
        
        # Visualize clustering results
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Elbow curve
        axes[0].plot(k_range, inertias, marker='o')
        axes[0].axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal k={optimal_k}')
        axes[0].set_xlabel('Number of Clusters (k)')
        axes[0].set_ylabel('Inertia')
        axes[0].set_title('Elbow Method for Optimal k')
        axes[0].legend()
        
        # 3D scatter plot of segments
        from mpl_toolkits.mplot3d import Axes3D
        ax = fig.add_subplot(122, projection='3d')
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for segment in range(optimal_k):
            segment_data = self.rfm_data[self.rfm_data['ml_segment'] == segment]
            ax.scatter(segment_data['recency'], segment_data['frequency'], segment_data['monetary'], 
                      c=colors[segment], label=cluster_names[segment], alpha=0.6)
        
        ax.set_xlabel('Recency')
        ax.set_ylabel('Frequency')
        ax.set_zlabel('Monetary')
        ax.set_title('3D Customer Segments')
        ax.legend()
        
        plt.tight_layout()
        plt.show()
        
        return ml_segment_analysis
    
    def market_basket_analysis(self):
        """Market Basket Analysis to find product associations"""
        print("\n=== MARKET BASKET ANALYSIS ===")
        
        # Prepare transaction data
        basket = self.data.groupby(['transaction_id', 'product_category'])['quantity'].sum().unstack().fillna(0)
        
        # Convert to binary matrix (bought/not bought)
        basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)
        
        # Apply Apriori algorithm
        frequent_itemsets = apriori(basket_sets, min_support=0.01, use_colnames=True)
        
        if len(frequent_itemsets) > 0:
            # Generate association rules
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0, num_itemsets=len(frequent_itemsets))
            
            if len(rules) > 0:
                # Sort by lift and confidence
                rules = rules.sort_values(['lift', 'confidence'], ascending=False)
                
                print("Top 10 Association Rules:")
                print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))
                
                # Visualize top rules
                plt.figure(figsize=(12, 8))
                
                # Support vs Confidence scatter plot
                plt.subplot(2, 2, 1)
                plt.scatter(rules['support'], rules['confidence'], c=rules['lift'], cmap='viridis', alpha=0.6)
                plt.colorbar(label='Lift')
                plt.xlabel('Support')
                plt.ylabel('Confidence')
                plt.title('Association Rules: Support vs Confidence')
                
                # Lift distribution
                plt.subplot(2, 2, 2)
                plt.hist(rules['lift'], bins=20, alpha=0.7, edgecolor='black')
                plt.xlabel('Lift')
                plt.ylabel('Frequency')
                plt.title('Distribution of Lift Values')
                
                # Top rules by lift
                plt.subplot(2, 1, 2)
                top_rules = rules.head(10)
                rule_labels = [f"{list(ant)[0]} → {list(cons)[0]}" 
                              for ant, cons in zip(top_rules['antecedents'], top_rules['consequents'])]
                
                plt.barh(range(len(top_rules)), top_rules['lift'])
                plt.yticks(range(len(top_rules)), rule_labels)
                plt.xlabel('Lift')
                plt.title('Top 10 Association Rules by Lift')
                
                plt.tight_layout()
                plt.show()
                
                return rules
            else:
                print("No significant association rules found")
                return None
        else:
            print("No frequent itemsets found")
            return None
    
    def predictive_modeling(self):
        """Build predictive models for CLV and churn prediction"""
        print("\n=== PREDICTIVE MODELING ===")
        
        # Prepare customer-level features
        customer_features = self.data.group
