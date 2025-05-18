"""
# Telco Customer Churn Analysis and Prediction
# Extended Analysis for Business Stakeholders

## Presentation Requirements Addressed:
# 1. Motivating Question: How can we predict and reduce customer churn to improve revenue retention?
# 2. Stakeholder: Chief Revenue Officer (CRO) who needs actionable insights to reduce churn rates
# 3. Subject-Matter Expertise: Telecommunications industry knowledge, customer behavior analysis, retention strategy
# 4. Data Collected: Telco customer dataset with 7,043 customers and 21 features
# 5. Extension Plans: Detailed in the code below
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import shap

# Set style for presentation-quality visualizations
plt.style.use('seaborn-v0_8-whitegrid')
colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]
sns.set_palette(colors)

# Function to create directory for saving outputs
def create_output_directory(dir_path="churn_analysis_results"):
    import os
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path

# -------------------------------------------------------------------------
# 1. DATA LOADING AND EXPLORATION
# -------------------------------------------------------------------------

def load_and_explore_data(file_path):
    """
    Load and perform initial data exploration for the telco dataset.
    This addresses presentation requirements #1 (motivating question) and #4 (data collected).
    """
    print("1. LOADING AND EXPLORING DATA")
    print("=" * 50)
    
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Display basic information
    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nFeature overview:")
    for col in df.columns:
        print(f"- {col}: {df[col].nunique()} unique values")
    
    # Calculate churn rate
    churn_rate = df[df['Churn'] == 'Yes'].shape[0] / df.shape[0] * 100
    print(f"\nOverall churn rate: {churn_rate:.2f}%")
    
    # Convert TotalCharges to numeric and handle missing values
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(f"\nMissing values found:\n{missing_values[missing_values > 0]}")
        df.dropna(inplace=True)
        print(f"After dropping missing values: {df.shape[0]} rows remaining")
    
    # Preview first few rows
    print("\nPreview of data:")
    print(df.head())
    
    return df

# -------------------------------------------------------------------------
# 2. DATA PREPROCESSING
# -------------------------------------------------------------------------

def preprocess_data(df):
    """
    Clean and preprocess the telco dataset.
    """
    print("\n2. PREPROCESSING DATA")
    print("=" * 50)
    
    # Create a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Separate numerical and categorical features
    numerical_features = processed_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = processed_df.select_dtypes(include=['object']).columns.tolist()
    categorical_features.remove('customerID')  # Remove ID column
    if 'Churn' in categorical_features:
        categorical_features.remove('Churn')  # Remove target variable
    
    print(f"Numerical features: {numerical_features}")
    print(f"Categorical features: {categorical_features}")
    
    # Handle categorical features
    label_encoders = {}
    for column in categorical_features:
        le = LabelEncoder()
        processed_df[column] = le.fit_transform(processed_df[column])
        label_encoders[column] = le
        mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        print(f"Encoding for {column}: {mapping}")
    
    # Binary encode Churn (target variable)
    churn_le = LabelEncoder()
    processed_df['Churn'] = churn_le.fit_transform(processed_df['Churn'])
    print(f"Churn encoding: {dict(zip(churn_le.classes_, churn_le.transform(churn_le.classes_)))}")
    
    return processed_df, label_encoders, numerical_features, categorical_features

# -------------------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# -------------------------------------------------------------------------

def perform_eda(df, output_dir):
    """
    Perform exploratory data analysis with visualizations focused on insights
    for the CRO stakeholder.
    This addresses presentation requirement #2 (stakeholder needs).
    """
    print("\n3. EXPLORATORY DATA ANALYSIS")
    print("=" * 50)
    
    # Create a figure directory
    import os
    figures_dir = os.path.join(output_dir, 'figures')
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    
    # Convert Churn to numeric for calculations if it's still categorical
    df_analysis = df.copy()
    if df_analysis['Churn'].dtype == 'object':
        df_analysis['Churn_Numeric'] = df_analysis['Churn'].map({'Yes': 1, 'No': 0})
    else:
        df_analysis['Churn_Numeric'] = df_analysis['Churn']
    
    # 1. Churn distribution
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data=df, x='Churn')
    plt.title("Customer Churn Distribution", fontsize=16)
    plt.xlabel("Churn Status", fontsize=14)
    plt.ylabel("Number of Customers", fontsize=14)
    
    # Add percentage labels
    total = len(df)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + 0.1,
                '{:.1f}%'.format(100 * height/total),
                ha="center", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "1_churn_distribution.png"), dpi=300)
    
    # 2. Churn rate by contract type - Key insight for the CRO
    plt.figure(figsize=(10, 6))
    contract_churn = df_analysis.groupby('Contract')['Churn_Numeric'].mean().reset_index()
    contract_churn.columns = ['Contract', 'ChurnRate']
    contract_churn['ChurnRate'] = contract_churn['ChurnRate'] * 100
    
    ax = sns.barplot(x='Contract', y='ChurnRate', data=contract_churn)
    plt.title("Churn Rate by Contract Type", fontsize=16)
    plt.xlabel("Contract Type", fontsize=14)
    plt.ylabel("Churn Rate (%)", fontsize=14)
    
    # Add value labels
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%", 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center', va = 'bottom', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "2_churn_by_contract.png"), dpi=300)
    
    # 3. Tenure vs. Churn - Critical for understanding customer lifecycle
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Churn', y='tenure', data=df)
    plt.title("Customer Tenure by Churn Status", fontsize=16)
    plt.xlabel("Churn Status", fontsize=14)
    plt.ylabel("Tenure (months)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "3_tenure_vs_churn.png"), dpi=300)
    
    # 4. Monthly Charges vs. Churn - Financial impact insight for CRO
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
    plt.title("Monthly Charges by Churn Status", fontsize=16)
    plt.xlabel("Churn Status", fontsize=14)
    plt.ylabel("Monthly Charges ($)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "4_charges_vs_churn.png"), dpi=300)
    
    # 5. Services impact on churn
    plt.figure(figsize=(14, 8))
    
    # Select service features
    service_features = ['InternetService', 'OnlineSecurity', 'OnlineBackup', 
                        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    # Create a dataframe for plotting
    service_impact = pd.DataFrame()
    for feature in service_features:
        feature_data = df_analysis.groupby(feature)['Churn_Numeric'].mean() * 100
        feature_df = pd.DataFrame({
            'Service': feature,
            'Category': feature_data.index.astype(str),
            'ChurnRate': feature_data.values
        })
        service_impact = pd.concat([service_impact, feature_df])
    
    # Plot
    sns.barplot(x='Category', y='ChurnRate', hue='Service', data=service_impact)
    plt.title("Impact of Services on Churn Rate", fontsize=16)
    plt.xlabel("Service Category", fontsize=14)
    plt.ylabel("Churn Rate (%)", fontsize=14)
    plt.xticks(rotation=45)
    plt.legend(title="Service Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "5_services_impact.png"), dpi=300)
    
    print(f"EDA visualizations saved to {figures_dir}")
    
    return figures_dir

# -------------------------------------------------------------------------
# 4. CUSTOMER SEGMENTATION
# -------------------------------------------------------------------------

def perform_customer_segmentation(df, numerical_features, output_dir):
    """
    Segment customers using K-Means clustering.
    This incorporates subject-matter expertise (requirement #3) by identifying
    distinct customer groups for targeted retention strategies.
    """
    print("\n4. CUSTOMER SEGMENTATION")
    print("=" * 50)
    
    # Select relevant features for clustering
    cluster_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    print(f"Using features for clustering: {cluster_features}")
    
    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[cluster_features])
    
    # Determine optimal number of clusters using the elbow method
    inertia = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_features)
        inertia.append(kmeans.inertia_)
    
    # Plot elbow method
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia, 'o-')
    plt.title("Elbow Method for Optimal k", fontsize=16)
    plt.xlabel("Number of clusters (k)", fontsize=14)
    plt.ylabel("Inertia", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figures/6_elbow_method.png", dpi=300)
    
    # Apply K-Means with the chosen number of clusters (4)
    n_clusters = 4
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(scaled_features)
    
    # Analyze clusters
    cluster_summary = df.groupby('Cluster').agg({
        'tenure': 'mean',
        'MonthlyCharges': 'mean',
        'TotalCharges': 'mean',
        'Churn': 'mean'
    })
    cluster_summary['ChurnRate'] = cluster_summary['Churn'] * 100
    cluster_summary['Count'] = df.groupby('Cluster').size()
    cluster_summary['Percentage'] = (cluster_summary['Count'] / len(df) * 100).round(2)
    
    print("\nCluster analysis:")
    print(cluster_summary)
    
    # Visualize clusters
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(df['tenure'], 
                         df['MonthlyCharges'], 
                         c=df['Cluster'], 
                         alpha=0.6, 
                         s=50,
                         cmap='viridis')
    plt.colorbar(scatter, label='Cluster')
    
    # Mark churned customers
    churned = df[df['Churn'] == 1]
    plt.scatter(churned['tenure'], 
               churned['MonthlyCharges'], 
               color='red',
               marker='x', 
               alpha=0.3,
               s=30,
               label='Churned')
    
    plt.title("Customer Segments by Tenure and Monthly Charges", fontsize=16)
    plt.xlabel("Tenure (months)", fontsize=14)
    plt.ylabel("Monthly Charges ($)", fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figures/7_customer_segments.png", dpi=300)
    
    # Analyze churn by cluster
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Cluster', y='ChurnRate', data=cluster_summary.reset_index())
    plt.title("Churn Rate by Customer Segment", fontsize=16)
    plt.xlabel("Customer Segment", fontsize=14)
    plt.ylabel("Churn Rate (%)", fontsize=14)
    
    # Add cluster size annotations
    for i, row in enumerate(cluster_summary.itertuples()):
        plt.text(i, row.ChurnRate + 1, 
                f"n={row.Count}\n({row.Percentage}%)", 
                ha='center', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figures/8_churn_by_segment.png", dpi=300)
    
    # Save cluster descriptions
    cluster_descriptions = pd.DataFrame({
        'Cluster': range(n_clusters),
        'Description': [
            'Long-term customers with moderate charges',
            'New customers with high charges (high risk)',
            'Mid-term customers with high charges',
            'New customers with low charges'
        ],
        'ChurnRate': cluster_summary['ChurnRate'].values,
        'Size': cluster_summary['Count'].values,
        'Percentage': cluster_summary['Percentage'].values
    })
    
    print("\nCluster descriptions:")
    print(cluster_descriptions)
    cluster_descriptions.to_csv(f"{output_dir}/cluster_descriptions.csv", index=False)
    
    return df, cluster_descriptions

# -------------------------------------------------------------------------
# 5. PREDICTIVE MODELING
# -------------------------------------------------------------------------

def build_predictive_models(df, output_dir):
    """
    Build and evaluate models to predict customer churn.
    """
    print("\n5. PREDICTIVE MODELING")
    print("=" * 50)
    
    # Prepare features and target
    X = df.drop(columns=['customerID', 'Churn'])
    y = df['Churn']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # Train models
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    # Dictionary to store results
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Evaluate
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Store results
        results[name] = {
            'model': model,
            'report': report,
            'confusion_matrix': conf_matrix,
            'predictions': y_pred
        }
        
        # Print metrics
        print(f"{name} Classification Report:")
        print(classification_report(y_test, y_pred))
    
    # Compare models
    model_comparison = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [results[model]['report']['accuracy'] for model in results],
        'Precision': [results[model]['report']['weighted avg']['precision'] for model in results],
        'Recall': [results[model]['report']['weighted avg']['recall'] for model in results],
        'F1-Score': [results[model]['report']['weighted avg']['f1-score'] for model in results]
    })
    
    print("\nModel Comparison:")
    print(model_comparison)
    model_comparison.to_csv(f"{output_dir}/model_comparison.csv", index=False)
    
    # Visualize confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for i, (name, result) in enumerate(results.items()):
        conf_matrix = result['confusion_matrix']
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Churn', 'Churn'],
                   yticklabels=['No Churn', 'Churn'],
                   ax=axes[i])
        axes[i].set_title(f"{name} Confusion Matrix", fontsize=14)
        axes[i].set_xlabel("Predicted", fontsize=12)
        axes[i].set_ylabel("Actual", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figures/9_confusion_matrices.png", dpi=300)
    
    # Feature importance for Random Forest
    if 'Random Forest' in results:
        rf_model = results['Random Forest']['model']
        feature_imp = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_imp.head(15))
        plt.title("Top 15 Features for Churn Prediction", fontsize=16)
        plt.xlabel("Feature Importance", fontsize=14)
        plt.ylabel("Feature", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/figures/10_feature_importance.png", dpi=300)
        
        # Save feature importance data
        feature_imp.to_csv(f"{output_dir}/feature_importance.csv", index=False)
    
    return results, model_comparison

# -------------------------------------------------------------------------
# 6. BUSINESS IMPACT ANALYSIS
# -------------------------------------------------------------------------

def analyze_business_impact(df, results, output_dir):
    """
    Analyze the business impact of churn and proposed interventions.
    This addresses presentation requirement #2 (stakeholder needs) by providing
    actionable insights for the CRO.
    """
    print("\n6. BUSINESS IMPACT ANALYSIS")
    print("=" * 50)
    
    csv_dir = f"{output_dir}/csv_results"
    figures_dir = f"{output_dir}/figures"
    
    # Create a copy of df with numeric churn for calculations
    df_analysis = df.copy()
    if df_analysis['Churn'].dtype == 'object':
        df_analysis['Churn'] = df_analysis['Churn'].map({'Yes': 1, 'No': 0})
    
    # Calculate average revenue per customer
    avg_monthly_revenue = df['MonthlyCharges'].mean()
    avg_customer_lifetime = df['tenure'].mean()
    avg_lifetime_value = avg_monthly_revenue * avg_customer_lifetime
    
    print(f"Average Monthly Revenue per Customer: ${avg_monthly_revenue:.2f}")
    print(f"Average Customer Lifetime (months): {avg_customer_lifetime:.2f}")
    print(f"Average Customer Lifetime Value: ${avg_lifetime_value:.2f}")
    
    # Save customer value metrics
    customer_value = pd.DataFrame({
        'Metric': ['Average Monthly Revenue', 'Average Customer Lifetime (months)', 'Average Customer Lifetime Value'],
        'Value': [avg_monthly_revenue, avg_customer_lifetime, avg_lifetime_value]
    })
    customer_value.to_csv(f"{csv_dir}/customer_value_metrics.csv", index=False)
    
    # Create customer value visualization
    plt.figure(figsize=(8, 5))
    plt.bar(['Monthly Revenue', 'Lifetime Value'], 
            [avg_monthly_revenue, avg_lifetime_value], 
            color=['#3498db', '#e74c3c'])
    plt.title("Customer Value Metrics", fontsize=16)
    plt.ylabel("Value ($)", fontsize=14)
    # Add value labels
    plt.text(0, avg_monthly_revenue/2, f"${avg_monthly_revenue:.2f}/mo", 
             ha='center', color='white', fontsize=12, fontweight='bold')
    plt.text(1, avg_lifetime_value/2, f"${avg_lifetime_value:.2f}\nlifetime", 
             ha='center', color='white', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/customer_value_metrics.png", dpi=300)
    
    # Estimated annual revenue loss due to churn
    monthly_churn_rate = df_analysis['Churn'].mean()
    annual_churn_rate = 1 - (1 - monthly_churn_rate) ** 12
    total_customers = len(df)
    customers_churned_annually = total_customers * annual_churn_rate
    annual_revenue_loss = customers_churned_annually * avg_monthly_revenue * 12
    
    print(f"Monthly Churn Rate: {monthly_churn_rate:.4f}")
    print(f"Projected Annual Churn Rate: {annual_churn_rate:.4f}")
    print(f"Estimated Customers Churned Annually: {customers_churned_annually:.0f}")
    print(f"Estimated Annual Revenue Loss: ${annual_revenue_loss:,.2f}")
    
    # Save churn impact metrics
    churn_impact = pd.DataFrame({
        'Metric': ['Monthly Churn Rate', 'Annual Churn Rate', 'Customers Churned Annually', 'Annual Revenue Loss'],
        'Value': [monthly_churn_rate, annual_churn_rate, customers_churned_annually, annual_revenue_loss]
    })
    churn_impact.to_csv(f"{csv_dir}/churn_impact_metrics.csv", index=False)
    
    # Visualize annual revenue loss
    plt.figure(figsize=(10, 6))
    plt.bar(['Annual Revenue Loss'], [annual_revenue_loss], color='#e74c3c')
    plt.title("Annual Revenue Loss Due to Churn", fontsize=16)
    plt.ylabel("Revenue Loss ($)", fontsize=14)
    plt.ylim(0, annual_revenue_loss * 1.3)  # Add space for text
    
    # Add details
    plt.text(0, annual_revenue_loss + annual_revenue_loss*0.05, 
             f"${annual_revenue_loss:,.0f}", 
             ha='center', fontsize=14, fontweight='bold')
    plt.text(0, annual_revenue_loss + annual_revenue_loss*0.15, 
             f"Based on {monthly_churn_rate:.1%} monthly churn\n{customers_churned_annually:.0f} customers lost annually", 
             ha='center', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/annual_revenue_loss.png", dpi=300)
    
    # Impact of reducing churn
    churn_reduction_targets = [0.05, 0.10, 0.20]  # 5%, 10%, 20% reduction
    
    churn_reduction_impact = []
    for target in churn_reduction_targets:
        reduced_monthly_churn = monthly_churn_rate * (1 - target)
        reduced_annual_churn = 1 - (1 - reduced_monthly_churn) ** 12
        reduced_customers_churned = total_customers * reduced_annual_churn
        customers_saved = customers_churned_annually - reduced_customers_churned
        revenue_saved = customers_saved * avg_monthly_revenue * 12
        
        churn_reduction_impact.append({
            'Reduction Target': f"{target*100}%",
            'New Monthly Churn Rate': reduced_monthly_churn,
            'New Annual Churn Rate': reduced_annual_churn,
            'Customers Saved': customers_saved,
            'Annual Revenue Saved': revenue_saved
        })
    
    impact_df = pd.DataFrame(churn_reduction_impact)
    print("\nImpact of Churn Reduction:")
    print(impact_df)
    impact_df.to_csv(f"{csv_dir}/churn_reduction_impact.csv", index=False)
    
    # Visualize revenue impact
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Reduction Target', y='Annual Revenue Saved', data=impact_df)
    
    # Add value labels
    for p in ax.patches:
        ax.annotate(f"${p.get_height():,.0f}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha = 'center', va = 'bottom', fontsize=12)
    
    plt.title("Annual Revenue Saved by Churn Reduction", fontsize=16)
    plt.xlabel("Churn Reduction Target", fontsize=14)
    plt.ylabel("Annual Revenue Saved ($)", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/11_revenue_impact.png", dpi=300)
    
    # High-quality standalone version for presentation
    plt.figure(figsize=(12, 7))
    colors = ["#3498db", "#2ecc71", "#e74c3c"]
    ax = sns.barplot(x='Reduction Target', y='Annual Revenue Saved', data=impact_df, palette=colors)
    
    # Add value labels
    for i, p in enumerate(ax.patches):
        ax.annotate(f"${p.get_height():,.0f}", 
                   (p.get_x() + p.get_width() / 2., p.get_height() + 5000),
                   ha = 'center', va = 'bottom', fontsize=14, fontweight='bold')
        
        # Add number of customers saved
        customers_saved = impact_df.iloc[i]['Customers Saved']
        ax.annotate(f"{customers_saved:.0f} customers", 
                   (p.get_x() + p.get_width() / 2., p.get_height()/2),
                   ha = 'center', va = 'center', fontsize=12, color='white', fontweight='bold')
    
    plt.title("Revenue Impact of Reducing Customer Churn", fontsize=18)
    plt.xlabel("Churn Reduction Target", fontsize=16)
    plt.ylabel("Annual Revenue Saved ($)", fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/revenue_impact_presentation.png", dpi=300, bbox_inches='tight')
    
    # Cost of retention vs. acquisition
    acquisition_cost = 500  # Assumed cost to acquire a new customer
    retention_cost = 100    # Assumed cost to retain a customer
    
    # Save acquisition vs retention cost comparison
    cost_comparison = pd.DataFrame({
        'Cost Type': ['Customer Acquisition', 'Customer Retention'],
        'Cost per Customer': [acquisition_cost, retention_cost]
    })
    cost_comparison.to_csv(f"{csv_dir}/acquisition_vs_retention_costs.csv", index=False)
    
    # Visualize the cost comparison
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Cost Type', y='Cost per Customer', data=cost_comparison, palette=['#e74c3c', '#2ecc71'])
    plt.title("Customer Acquisition vs. Retention Costs", fontsize=16)
    plt.ylabel("Cost per Customer ($)", fontsize=14)
    
    # Add value labels
    ax = plt.gca()
    for i, p in enumerate(ax.patches):
        ax.annotate(f"${p.get_height():.0f}", 
                   (p.get_x() + p.get_width() / 2., p.get_height() + 10),
                   ha = 'center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/acquisition_vs_retention_costs.png", dpi=300)
    
    # Calculate ROI of retention efforts
    roi_data = []
    for target in churn_reduction_targets:
        reduced_monthly_churn = monthly_churn_rate * (1 - target)
        reduced_annual_churn = 1 - (1 - reduced_monthly_churn) ** 12
        reduced_customers_churned = total_customers * reduced_annual_churn
        customers_saved = customers_churned_annually - reduced_customers_churned
        revenue_saved = customers_saved * avg_monthly_revenue * 12
        
        retention_investment = customers_saved * retention_cost
        acquisition_equivalent = customers_saved * acquisition_cost
        net_benefit = revenue_saved + acquisition_equivalent - retention_investment
        roi = (net_benefit / retention_investment) * 100
        
        roi_data.append({
            'Reduction Target': f"{target*100}%",
            'Retention Investment': retention_investment,
            'Acquisition Equivalent': acquisition_equivalent,
            'Net Benefit': net_benefit,
            'ROI': roi
        })
    
    roi_df = pd.DataFrame(roi_data)
    print("\nROI of Retention Efforts:")
    print(roi_df)
    roi_df.to_csv(f"{csv_dir}/retention_roi.csv", index=False)
    
    # Visualize ROI
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Reduction Target', y='ROI', data=roi_df)
    
    # Add value labels
    ax = plt.gca()
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%", 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha = 'center', va = 'bottom', fontsize=12)
    
    plt.title("ROI of Customer Retention Efforts", fontsize=16)
    plt.xlabel("Churn Reduction Target", fontsize=14)
    plt.ylabel("Return on Investment (%)", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/12_retention_roi.png", dpi=300)
    
    # Create comprehensive ROI visualization for presentation
    plt.figure(figsize=(12, 8))
    
    # Create a more detailed visualization showing the components of ROI
    x = np.arange(len(churn_reduction_targets))
    width = 0.25
    
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Plot bars for investment and benefit
    ax1.bar(x - width, roi_df['Retention Investment'], width, label='Investment', color='#3498db')
    ax1.bar(x, roi_df['Acquisition Equivalent'], width, label='Acquisition Cost Avoided', color='#2ecc71')
    ax1.bar(x + width, roi_df['Net Benefit'], width, label='Net Benefit', color='#e74c3c')
    
    # Add a second y-axis for ROI percentage
    ax2 = ax1.twinx()
    ax2.plot(x, roi_df['ROI'], 'o-', linewidth=3, color='#9b59b6', label='ROI (%)')
    
    # Add annotations
    for i, (investment, equivalent, benefit, roi) in enumerate(zip(
        roi_df['Retention Investment'], 
        roi_df['Acquisition Equivalent'],
        roi_df['Net Benefit'],
        roi_df['ROI']
    )):
        ax1.annotate(f"${investment:,.0f}", 
                    (i - width, investment + 5000), 
                    ha='center', va='bottom', fontsize=10)
        ax1.annotate(f"${equivalent:,.0f}", 
                    (i, equivalent + 5000), 
                    ha='center', va='bottom', fontsize=10)
        ax1.annotate(f"${benefit:,.0f}", 
                    (i + width, benefit + 5000), 
                    ha='center', va='bottom', fontsize=10)
        ax2.annotate(f"{roi:.1f}%", 
                    (i, roi + 20), 
                    ha='center', va='bottom', color='#9b59b6', fontsize=12, fontweight='bold')
    
    # Configure axes
    ax1.set_xlabel('Churn Reduction Target', fontsize=14)
    ax1.set_ylabel('Dollar Value ($)', fontsize=14)
    ax2.set_ylabel('ROI (%)', fontsize=14, color='#9b59b6')
    ax1.set_title('Economics of Customer Retention Investments', fontsize=18)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{target*100}%" for target in churn_reduction_targets])
    ax2.tick_params(axis='y', colors='#9b59b6')
    
    # Add grid and legend
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/retention_economics_presentation.png", dpi=300, bbox_inches='tight')
    
    # Create a simple summary table of key metrics
    key_metrics = pd.DataFrame({
        'Metric': [
            'Monthly Churn Rate', 
            'Annual Revenue Loss', 
            'Customer Lifetime Value',
            'Acquisition Cost per Customer',
            'Retention Cost per Customer',
            'ROI of 10% Churn Reduction'
        ],
        'Value': [
            f"{monthly_churn_rate:.1%}",
            f"${annual_revenue_loss:,.0f}",
            f"${avg_lifetime_value:.2f}",
            f"${acquisition_cost:.0f}",
            f"${retention_cost:.0f}",
            f"{roi_df.iloc[1]['ROI']:.1f}%"
        ]
    })
    
    key_metrics.to_csv(f"{csv_dir}/key_business_metrics.csv", index=False)
    
    return impact_df, roi_df

# -------------------------------------------------------------------------
# 7. STRATEGIC RECOMMENDATIONS
# -------------------------------------------------------------------------

def generate_recommendations(df, cluster_descriptions, feature_importance, output_dir):
    """
    Generate strategic recommendations for reducing churn.
    This addresses presentation requirement #2 (stakeholder needs) and #3 (subject-matter expertise).
    """
    print("\n7. STRATEGIC RECOMMENDATIONS")
    print("=" * 50)
    
    # Create a recommendations document
    recommendations = {
        'Overall Strategy': [
            "Focus retention efforts on high-risk clusters identified in the analysis",
            "Develop targeted offers for customers with month-to-month contracts",
            "Invest in improving online security and tech support services",
            "Implement an early warning system using the predictive model"
        ],
        'Cluster-Specific Tactics': {},
        'Implementation Plan': [
            "Phase 1: Launch targeted retention offers for highest-risk segments",
            "Phase 2: Improve service quality based on feature importance analysis",
            "Phase 3: Implement predictive churn model for proactive interventions",
            "Phase 4: Measure impact and refine approach"
        ]
    }
    
    # Add cluster-specific recommendations
    for _, row in cluster_descriptions.iterrows():
        cluster = row['Cluster']
        
        if row['ChurnRate'] > 30:
            risk_level = "High"
            priority = "Critical"
        elif row['ChurnRate'] > 15:
            risk_level = "Medium"
            priority = "Important"
        else:
            risk_level = "Low"
            priority = "Monitor"
        
        recommendations['Cluster-Specific Tactics'][f"Cluster {cluster}"] = {
            'Description': row['Description'],
            'Churn Risk': f"{risk_level} ({row['ChurnRate']:.1f}%)",
            'Customer Base': f"{row['Percentage']}% of customers",
            'Priority': priority,
            'Recommendations': []
        }
    
    # Add specific recommendations for each cluster based on analysis
    recommendations['Cluster-Specific Tactics']['Cluster 0']['Recommendations'] = [
        "Offer loyalty rewards for long-term customers",
        "Create premium service bundles with slight discounts",
        "Implement satisfaction surveys to monitor experience"
    ]
    
    recommendations['Cluster-Specific Tactics']['Cluster 1']['Recommendations'] = [
        "Offer contract incentives to move from month-to-month to annual",
        "Provide free service upgrades for signing longer contracts",
        "Implement 90-day check-ins for new high-value customers"
    ]
    
    recommendations['Cluster-Specific Tactics']['Cluster 2']['Recommendations'] = [
        "Create tailored service bundles to optimize value perception",
        "Offer tech support improvements and online security enhancements",
        "Develop mid-contract service reviews and adjustments"
    ]
    
    recommendations['Cluster-Specific Tactics']['Cluster 3']['Recommendations'] = [
        "Create entry-level service bundles with promotional pricing",
        "Offer step-up plans with clear value communication",
        "Implement onboarding satisfaction checks"
    ]
    
    # Save recommendations
    import json
    with open(f"{output_dir}/strategic_recommendations.json", 'w') as f:
        json.dump(recommendations, f, indent=4)
    
    print("Strategic recommendations saved to output directory")
    
    # Create a visualization of recommendation priority by cluster
    cluster_priority = pd.DataFrame({
        'Cluster': [f"Cluster {i}" for i in range(len(cluster_descriptions))],
        'ChurnRate': cluster_descriptions['ChurnRate'].values,
        'Size': cluster_descriptions['Size'].values,
        'Priority': [
            'Critical' if rate > 30 else 'Important' if rate > 15 else 'Monitor'
            for rate in cluster_descriptions['ChurnRate'].values
        ]
    })
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        cluster_priority['Size'], 
        cluster_priority['ChurnRate'],
        s=cluster_priority['Size'] / 10,
        c=[0, 1, 2, 3],
        cmap='viridis',
        alpha=0.7
    )
    
    # Add cluster labels
    for i, row in cluster_priority.iterrows():
        plt.annotate(
            f"Cluster {i}\n({row['Priority']})",
            (row['Size'], row['ChurnRate']),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center'
        )
    
    plt.title("Retention Strategy Priority by Customer Segment", fontsize=16)
    plt.xlabel("Segment Size (# of customers)", fontsize=14)
    plt.ylabel("Churn Rate (%)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figures/13_retention_priority.png", dpi=300)
    
    return recommendations

# -------------------------------------------------------------------------
# 8. EXTENSION PLANS
# -------------------------------------------------------------------------

def define_extension_plans():
    """
    Define plans for extending the project.
    This addresses presentation requirement #5 (extension plans).
    """
    print("\n8. EXTENSION PLANS")
    print("=" * 50)
    
    extension_plans = {
        'Data Enrichment': [
            "Incorporate customer interaction data (calls, support tickets)",
            "Add competitor pricing and offering data",
            "Include network performance metrics by geographic region"
        ],
        'Advanced Modeling': [
            "Implement time-series churn prediction (survival analysis)",
            "Develop customer lifetime value prediction models",
            "Create next-best-offer recommendation system"
        ],
        'Implementation Support': [
            "Design a real-time churn risk dashboard for business users",
            "Develop an A/B testing framework for retention initiatives",
            "Create ROI measurement tools for retention campaigns"
        ],
        'Expansion Areas': [
            "Extend analysis to customer acquisition optimization",
            "Develop cross-sell/upsell propensity models",
            "Build customer segmentation automation pipeline"
        ]
    }
    
    for category, plans in extension_plans.items():
        print(f"\n{category}:")
        for plan in plans:
            print(f"- {plan}")
    
    return extension_plans

# -------------------------------------------------------------------------
# 8. COMPREHENSIVE REPORT GENERATION
# -------------------------------------------------------------------------

def generate_comprehensive_report(output_dir, file_summary, df, cluster_descriptions, model_comparison, impact_df, roi_df):
    """
    Generate a comprehensive report with all analysis results
    """
    print("\n8. GENERATING COMPREHENSIVE REPORT")
    print("=" * 50)
    
    import os
    import base64
    from datetime import datetime
    
    # Create markdown report
    report_path = os.path.join(output_dir, "Telco_Churn_Analysis_Report.md")
    html_report_path = os.path.join(output_dir, "Telco_Churn_Analysis_Report.html")
    figures_dir = os.path.join(output_dir, 'figures')
    csv_dir = os.path.join(output_dir, 'csv_results')
    
    print(f"Generating comprehensive report at: {report_path}")
    
    # Load various CSV files
    churn_distribution = pd.read_csv(os.path.join(csv_dir, "churn_distribution.csv"))
    contract_churn = pd.read_csv(os.path.join(csv_dir, "contract_churn_rates.csv"))
    key_metrics = pd.read_csv(os.path.join(csv_dir, "key_business_metrics.csv"))
    try:
        top_factors = pd.read_csv(os.path.join(csv_dir, "top_10_churn_factors.csv"))
    except:
        top_factors = pd.DataFrame({'Feature': ['Feature data not available'], 'Importance': [0]})
    
    # Load recommendations
    try:
        recommendations = pd.read_csv(os.path.join(csv_dir, "strategic_recommendations.csv"))
    except:
        recommendations = pd.DataFrame({'Category': ['Recommendations not available'], 'Recommendation': ['Not available']})
    
    # Load extension plans
    try:
        extension_plans = pd.read_csv(os.path.join(csv_dir, "extension_plans.csv"))
    except:
        extension_plans = pd.DataFrame({'Category': ['Extension plans not available'], 'Plan': ['Not available']})
    
    # Set up the report content
    report_content = f"""# Telco Customer Churn Analysis
### Comprehensive Analysis Report
*Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Data Overview](#data-overview)
3. [Churn Analysis Findings](#churn-analysis-findings)
4. [Customer Segmentation](#customer-segmentation)
5. [Predictive Modeling Results](#predictive-modeling-results)
6. [Business Impact Analysis](#business-impact-analysis)
7. [Strategic Recommendations](#strategic-recommendations)
8. [Extension Plans](#extension-plans)

## Executive Summary <a name="executive-summary"></a>

This analysis addresses the critical business problem of customer churn in the telecommunications industry. With a churn rate of {churn_distribution[churn_distribution['Churn_Status'] == 'Yes']['Percentage'].values[0]:.2f}%, the company is 
losing significant revenue and needs targeted retention strategies.

Key findings:
- Contract type is the strongest predictor of churn, with month-to-month customers churning at {contract_churn[contract_churn['Contract'] == 'Month-to-month']['Churn_Rate'].values[0]:.1f}%
- Customer tenure shows a strong inverse relationship with churn probability
- Service quality and technical support significantly impact retention
- Customer segmentation revealed 4 distinct customer groups with varying churn risk profiles

The primary stakeholder for this analysis is the **Chief Revenue Officer (CRO)**, who needs actionable insights
to implement targeted retention strategies to improve revenue retention.

## Data Overview <a name="data-overview"></a>

The analysis is based on telco customer data with {len(df)} customers and 21 features including:
- Demographics (gender, senior citizen status, partners, dependents)
- Services subscribed (phone, internet, security, streaming)
- Contract details (type, tenure, billing method)
- Financial metrics (monthly charges, total charges)

### Churn Distribution
| Churn Status | Count | Percentage |
|--------------|-------|------------|
"""

    # Add churn distribution table
    for _, row in churn_distribution.iterrows():
        report_content += f"| {row['Churn_Status']} | {row['Count']} | {row['Percentage']:.2f}% |\n"
    
    report_content += """
## Churn Analysis Findings <a name="churn-analysis-findings"></a>

### Churn by Contract Type
| Contract Type | Total Customers | Churned Customers | Churn Rate |
|---------------|----------------|-------------------|------------|
"""
    
    # Add contract churn data
    for _, row in contract_churn.iterrows():
        report_content += f"| {row['Contract']} | {row['Total']} | {row['Yes']} | {row['Churn_Rate']:.2f}% |\n"
    
    report_content += """
### Key Visualizations

#### Churn Distribution
![Churn Distribution](figures/1_churn_distribution.png)

#### Churn by Contract Type
![Churn by Contract Type](figures/2_churn_by_contract.png)

#### Tenure vs Churn
![Tenure vs Churn](figures/3_tenure_vs_churn.png)

#### Monthly Charges vs Churn
![Monthly Charges vs Churn](figures/4_charges_vs_churn.png)

## Customer Segmentation <a name="customer-segmentation"></a>

K-means clustering was used to identify distinct customer segments based on tenure, monthly charges, and total charges.

### Customer Segments
| Cluster ID | Description | Churn Rate | Size |
|------------|-------------|------------|------|
"""
    
    # Add cluster descriptions
    for _, row in cluster_descriptions.iterrows():
        report_content += f"| Cluster {row['Cluster']} | {row['Description']} | {row['ChurnRate']:.2f}% | {row['Size']} ({row['Percentage']:.2f}%) |\n"
    
    report_content += """
### Segment Visualization
![Customer Segments](figures/7_customer_segments.png)

### Churn Rate by Segment
![Churn by Segment](figures/8_churn_by_segment.png)

## Predictive Modeling Results <a name="predictive-modeling-results"></a>

Two machine learning models were built to predict customer churn:

### Model Performance Comparison
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
"""
    
    # Add model comparison data
    for _, row in model_comparison.iterrows():
        report_content += f"| {row['Model']} | {row['Accuracy']:.4f} | {row['Precision']:.4f} | {row['Recall']:.4f} | {row['F1-Score']:.4f} |\n"
    
    report_content += """
### Top Churn Factors
| Feature | Importance |
|---------|------------|
"""
    
    # Add top factors
    for _, row in top_factors.head(10).iterrows():
        report_content += f"| {row['Feature']} | {row['Importance']:.4f} |\n"
    
    report_content += """
### Feature Importance Visualization
![Feature Importance](figures/10_feature_importance.png)

## Business Impact Analysis <a name="business-impact-analysis"></a>

### Key Business Metrics
| Metric | Value |
|--------|-------|
"""
    
    # Add key business metrics
    for _, row in key_metrics.iterrows():
        report_content += f"| {row['Metric']} | {row['Value']} |\n"
    
    report_content += """
### Impact of Churn Reduction
| Reduction Target | Customers Saved | Annual Revenue Saved |
|------------------|----------------|---------------------|
"""
    
    # Add impact analysis data
    for _, row in impact_df.iterrows():
        report_content += f"| {row['Reduction Target']} | {row['Customers Saved']:.0f} | ${row['Annual Revenue Saved']:,.2f} |\n"
    
    report_content += """
### ROI of Retention Efforts
| Reduction Target | Retention Investment | Net Benefit | ROI |
|------------------|---------------------|------------|-----|
"""
    
    # Add ROI data
    for _, row in roi_df.iterrows():
        report_content += f"| {row['Reduction Target']} | ${row['Retention Investment']:,.2f} | ${row['Net Benefit']:,.2f} | {row['ROI']:.2f}% |\n"
    
    report_content += """
### Revenue Impact Visualization
![Revenue Impact](figures/revenue_impact_presentation.png)

### Retention Economics
![Retention Economics](figures/retention_economics_presentation.png)

## Strategic Recommendations <a name="strategic-recommendations"></a>

Based on the analysis, the following strategic recommendations are provided:

"""
    
    # Add recommendations by category
    current_category = ""
    for _, row in recommendations.iterrows():
        if 'Category' in row and row['Category'] != current_category:
            current_category = row['Category']
            report_content += f"\n### {current_category}\n"
        if 'Recommendation' in row:
            report_content += f"- {row['Recommendation']}\n"
    
    report_content += """
## Extension Plans <a name="extension-plans"></a>

The following extensions are planned for this analysis:

"""
    
    # Add extension plans by category
    current_category = ""
    for _, row in extension_plans.iterrows():
        if 'Category' in row and row['Category'] != current_category:
            if current_category != "":
                report_content += "</ul>"
            current_category = row['Category']
            report_content += f"<h3>{current_category}</h3><ul>"
        if 'Plan' in row:
            report_content += f"<li>{row['Plan']}</li>"
    
    if current_category != "":
        report_content += "</ul>"
    
    report_content += """
</body>
</html>
"""
    
    # Save markdown report
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    # Create HTML version with embedded images
    html_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Telco Customer Churn Analysis Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #2980b9;
            border-bottom: 1px solid #ddd;
            padding-bottom: 5px;
            margin-top: 30px;
        }
        h3 {
            color: #3498db;
            margin-top: 25px;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }
        th, td {
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f2f2f2;
            color: #333;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        tr:hover {
            background-color: #f1f1f1;
        }
        img {
            max-width: 100%;
            height: auto;
            margin: 20px 0;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 5px;
        }
        .container {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
        }
        .viz-container {
            width: 48%;
            margin-bottom: 20px;
        }
        .toc {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 30px;
        }
        .toc a {
            color: #3498db;
            text-decoration: none;
        }
        .toc a:hover {
            text-decoration: underline;
        }
        .key-metric {
            font-weight: bold;
            color: #e74c3c;
        }
        .date {
            color: #7f8c8d;
            font-style: italic;
            margin-bottom: 30px;
        }
        @media print {
            body {
                padding: 0;
                font-size: 12pt;
            }
            h1 {
                font-size: 18pt;
            }
            h2 {
                font-size: 16pt;
            }
            h3 {
                font-size: 14pt;
            }
        }
    </style>
</head>
<body>
    <h1>Telco Customer Churn Analysis</h1>
    <h3>Comprehensive Analysis Report</h3>
    <div class="date">Generated on """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</div>
    
    <div class="toc">
        <h2>Table of Contents</h2>
        <ol>
            <li><a href="#executive-summary">Executive Summary</a></li>
            <li><a href="#data-overview">Data Overview</a></li>
            <li><a href="#churn-analysis">Churn Analysis Findings</a></li>
            <li><a href="#customer-segmentation">Customer Segmentation</a></li>
            <li><a href="#predictive-modeling">Predictive Modeling Results</a></li>
            <li><a href="#business-impact">Business Impact Analysis</a></li>
            <li><a href="#recommendations">Strategic Recommendations</a></li>
            <li><a href="#extension-plans">Extension Plans</a></li>
        </ol>
    </div>
    
    <h2 id="executive-summary">1. Executive Summary</h2>
    <p>This analysis addresses the critical business problem of customer churn in the telecommunications industry. 
       With a churn rate of """ + f"{churn_distribution[churn_distribution['Churn_Status'] == 'Yes']['Percentage'].values[0]:.2f}%" + """, the company is 
       losing significant revenue and needs targeted retention strategies.</p>
    
    <p><strong>Key findings:</strong></p>
    <ul>
        <li>Contract type is the strongest predictor of churn, with month-to-month customers churning at 
            <span class="key-metric">""" + f"{contract_churn[contract_churn['Contract'] == 'Month-to-month']['Churn_Rate'].values[0]:.1f}%" + """</span></li>
        <li>Customer tenure shows a strong inverse relationship with churn probability</li>
        <li>Service quality and technical support significantly impact retention</li>
        <li>Customer segmentation revealed 4 distinct customer groups with varying churn risk profiles</li>
    </ul>
    
    <p>The primary stakeholder for this analysis is the <strong>Chief Revenue Officer (CRO)</strong>, who needs actionable insights
       to implement targeted retention strategies to improve revenue retention.</p>
    
    <h2 id="data-overview">2. Data Overview</h2>
    <p>The analysis is based on telco customer data with """ + f"{len(df)}" + """ customers and 21 features including:</p>
    <ul>
        <li>Demographics (gender, senior citizen status, partners, dependents)</li>
        <li>Services subscribed (phone, internet, security, streaming)</li>
        <li>Contract details (type, tenure, billing method)</li>
        <li>Financial metrics (monthly charges, total charges)</li>
    </ul>
    
    <h3>Churn Distribution</h3>
    <table>
        <tr>
            <th>Churn Status</th>
            <th>Count</th>
            <th>Percentage</th>
        </tr>
"""
    
    # Add churn distribution table
    for _, row in churn_distribution.iterrows():
        html_content += f"""
        <tr>
            <td>{row['Churn_Status']}</td>
            <td>{row['Count']}</td>
            <td>{row['Percentage']:.2f}%</td>
        </tr>"""
    
    html_content += """
    </table>
    
    <h2 id="churn-analysis">3. Churn Analysis Findings</h2>
    
    <h3>Churn by Contract Type</h3>
    <table>
        <tr>
            <th>Contract Type</th>
            <th>Total Customers</th>
            <th>Churned Customers</th>
            <th>Churn Rate</th>
        </tr>
"""
    
    # Add contract churn data
    for _, row in contract_churn.iterrows():
        html_content += f"""
        <tr>
            <td>{row['Contract']}</td>
            <td>{row['Total']}</td>
            <td>{row['Yes']}</td>
            <td>{row['Churn_Rate']:.2f}%</td>
        </tr>"""
    
    html_content += """
    </table>
    
    <h3>Key Visualizations</h3>
    <div class="container">
        <div class="viz-container">
            <h4>Churn Distribution</h4>
            <img src="figures/1_churn_distribution.png" alt="Churn Distribution">
        </div>
        <div class="viz-container">
            <h4>Churn by Contract Type</h4>
            <img src="figures/2_churn_by_contract.png" alt="Churn by Contract Type">
        </div>
        <div class="viz-container">
            <h4>Tenure vs Churn</h4>
            <img src="figures/3_tenure_vs_churn.png" alt="Tenure vs Churn">
        </div>
        <div class="viz-container">
            <h4>Monthly Charges vs Churn</h4>
            <img src="figures/4_charges_vs_churn.png" alt="Monthly Charges vs Churn">
        </div>
    </div>
    
    <h2 id="customer-segmentation">4. Customer Segmentation</h2>
    <p>K-means clustering was used to identify distinct customer segments based on tenure, monthly charges, and total charges.</p>
    
    <h3>Customer Segments</h3>
    <table>
        <tr>
            <th>Cluster ID</th>
            <th>Description</th>
            <th>Churn Rate</th>
            <th>Size</th>
        </tr>
"""
    
    # Add cluster descriptions
    for _, row in cluster_descriptions.iterrows():
        html_content += f"""
        <tr>
            <td>Cluster {row['Cluster']}</td>
            <td>{row['Description']}</td>
            <td>{row['ChurnRate']:.2f}%</td>
            <td>{row['Size']} ({row['Percentage']:.2f}%)</td>
        </tr>"""
    
    html_content += """
    </table>
    
    <div class="container">
        <div class="viz-container">
            <h4>Customer Segments</h4>
            <img src="figures/7_customer_segments.png" alt="Customer Segments">
        </div>
        <div class="viz-container">
            <h4>Churn Rate by Segment</h4>
            <img src="figures/8_churn_by_segment.png" alt="Churn by Segment">
        </div>
    </div>
    
    <h2 id="predictive-modeling">5. Predictive Modeling Results</h2>
    <p>Two machine learning models were built to predict customer churn:</p>
    
    <h3>Model Performance Comparison</h3>
    <table>
        <tr>
            <th>Model</th>
            <th>Accuracy</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1-Score</th>
        </tr>
"""
    
    # Add model comparison data
    for _, row in model_comparison.iterrows():
        html_content += f"""
        <tr>
            <td>{row['Model']}</td>
            <td>{row['Accuracy']:.4f}</td>
            <td>{row['Precision']:.4f}</td>
            <td>{row['Recall']:.4f}</td>
            <td>{row['F1-Score']:.4f}</td>
        </tr>"""
    
    html_content += """
    </table>
    
    <h3>Top Churn Factors</h3>
    <table>
        <tr>
            <th>Feature</th>
            <th>Importance</th>
        </tr>
"""
    
    # Add top factors
    for _, row in top_factors.head(10).iterrows():
        html_content += f"""
        <tr>
            <td>{row['Feature']}</td>
            <td>{row['Importance']:.4f}</td>
        </tr>"""
    
    html_content += """
    </table>
    
    <div class="viz-container">
        <h4>Feature Importance</h4>
        <img src="figures/10_feature_importance.png" alt="Feature Importance">
    </div>
    
    <h2 id="business-impact">6. Business Impact Analysis</h2>
    
    <h3>Key Business Metrics</h3>
    <table>
        <tr>
            <th>Metric</th>
            <th>Value</th>
        </tr>
"""
    
    # Add key business metrics
    for _, row in key_metrics.iterrows():
        html_content += f"""
        <tr>
            <td>{row['Metric']}</td>
            <td>{row['Value']}</td>
        </tr>"""
    
    html_content += """
    </table>
    
    <h3>Impact of Churn Reduction</h3>
    <table>
        <tr>
            <th>Reduction Target</th>
            <th>Customers Saved</th>
            <th>Annual Revenue Saved</th>
        </tr>
"""
    
    # Add impact analysis data
    for _, row in impact_df.iterrows():
        html_content += f"""
        <tr>
            <td>{row['Reduction Target']}</td>
            <td>{row['Customers Saved']:.0f}</td>
            <td>${row['Annual Revenue Saved']:,.2f}</td>
        </tr>"""
    
    html_content += """
    </table>
    
    <h3>ROI of Retention Efforts</h3>
    <table>
        <tr>
            <th>Reduction Target</th>
            <th>Retention Investment</th>
            <th>Net Benefit</th>
            <th>ROI</th>
        </tr>
"""
    
    # Add ROI data
    for _, row in roi_df.iterrows():
        html_content += f"""
        <tr>
            <td>{row['Reduction Target']}</td>
            <td>${row['Retention Investment']:,.2f}</td>
            <td>${row['Net Benefit']:,.2f}</td>
            <td>{row['ROI']:.2f}%</td>
        </tr>"""
    
    html_content += """
    </table>
    
    <div class="container">
        <div class="viz-container">
            <h4>Revenue Impact</h4>
            <img src="figures/revenue_impact_presentation.png" alt="Revenue Impact">
        </div>
        <div class="viz-container">
            <h4>Retention Economics</h4>
            <img src="figures/retention_economics_presentation.png" alt="Retention Economics">
        </div>
    </div>
    
    <h2 id="recommendations">7. Strategic Recommendations</h2>
    <p>Based on the analysis, the following strategic recommendations are provided:</p>
"""
    
    # Add recommendations by category
    current_category = ""
    for _, row in recommendations.iterrows():
        if 'Category' in row and row['Category'] != current_category:
            current_category = row['Category']
            html_content += f"<h3>{current_category}</h3><ul>"
        if 'Recommendation' in row:
            html_content += f"<li>{row['Recommendation']}</li>"
        if 'Category' in row and row['Category'] != current_category:
            html_content += "</ul>"
    
    html_content += """
    <h2 id="extension-plans">8. Extension Plans</h2>
    <p>The following extensions are planned for this analysis:</p>
"""
    
    # Add extension plans by category
    current_category = ""
    for _, row in extension_plans.iterrows():
        if 'Category' in row and row['Category'] != current_category:
            if current_category != "":
                html_content += "</ul>"
            current_category = row['Category']
            html_content += f"<h3>{current_category}</h3><ul>"
        if 'Plan' in row:
            html_content += f"<li>{row['Plan']}</li>"
    
    if current_category != "":
        html_content += "</ul>"
    
    html_content += """
</body>
</html>
"""
    
    # Save HTML report
    with open(html_report_path, 'w') as f:
        f.write(html_content)
    
    print(f"Markdown report saved to: {report_path}")
    print(f"HTML report saved to: {html_report_path}")
    
    return report_path, html_report_path

# -------------------------------------------------------------------------
# MAIN EXECUTION FUNCTION
# -------------------------------------------------------------------------

def run_telco_churn_analysis(file_path="WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    """
    Main function to run the entire analysis pipeline.
    """
    # Create output directory
    output_dir = create_output_directory()
    figures_dir = os.path.join(output_dir, "figures")
    csv_dir = os.path.join(output_dir, "csv_results")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    
    print(f"Results will be saved to: {output_dir}")
    print(f"Visualizations will be saved to: {figures_dir}")
    print(f"CSV results will be saved to: {csv_dir}")
    
    # Execute the pipeline
    df = load_and_explore_data(file_path)
    # Save raw data summary
    df.describe().to_csv(os.path.join(csv_dir, "data_summary_statistics.csv"))
    df.dtypes.to_csv(os.path.join(csv_dir, "data_types.csv"))
    
    # Save churn distribution as CSV
    churn_counts = df['Churn'].value_counts().reset_index()
    churn_counts.columns = ['Churn_Status', 'Count']
    churn_counts['Percentage'] = churn_counts['Count'] / churn_counts['Count'].sum() * 100
    churn_counts.to_csv(os.path.join(csv_dir, "churn_distribution.csv"), index=False)
    
    processed_df, label_encoders, numerical_features, categorical_features = preprocess_data(df)
    # Save encoding mappings
    encoding_mappings = {}
    for col, le in label_encoders.items():
        encoding_mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))
    
    pd.DataFrame([encoding_mappings]).to_csv(os.path.join(csv_dir, "encoding_mappings.csv"))
    
    figures_dir = perform_eda(df, output_dir)
    
    # Additional EDA visualizations and CSV exports
    # Contract type churn rates
    contract_churn = df.groupby('Contract')['Churn'].value_counts().unstack().reset_index()
    contract_churn['Total'] = contract_churn['No'] + contract_churn['Yes']
    contract_churn['Churn_Rate'] = contract_churn['Yes'] / contract_churn['Total'] * 100
    contract_churn.to_csv(os.path.join(csv_dir, "contract_churn_rates.csv"), index=False)
    
    # Internet service churn rates
    internet_churn = df.groupby('InternetService')['Churn'].value_counts().unstack().reset_index()
    internet_churn['Total'] = internet_churn['No'] + internet_churn['Yes']
    internet_churn['Churn_Rate'] = internet_churn['Yes'] / internet_churn['Total'] * 100
    internet_churn.to_csv(os.path.join(csv_dir, "internet_service_churn_rates.csv"), index=False)
    
    # Create tenure groups and save churn rates
    df['TenureGroup'] = pd.cut(
        df['tenure'], 
        bins=[0, 12, 24, 36, 48, 60, float('inf')],
        labels=['0-12', '13-24', '25-36', '37-48', '49-60', '61+']
    )
    tenure_churn = df.groupby('TenureGroup')['Churn'].value_counts().unstack().reset_index()
    tenure_churn['Total'] = tenure_churn['No'] + tenure_churn['Yes']
    tenure_churn['Churn_Rate'] = tenure_churn['Yes'] / tenure_churn['Total'] * 100
    tenure_churn.to_csv(os.path.join(csv_dir, "tenure_group_churn_rates.csv"), index=False)
    
    # Extra visualization: Tenure groups churn rates
    plt.figure(figsize=(12, 6))
    sns.barplot(x='TenureGroup', y='Churn_Rate', data=tenure_churn)
    plt.title("Churn Rate by Tenure Group", fontsize=16)
    plt.xlabel("Tenure (months)", fontsize=14)
    plt.ylabel("Churn Rate (%)", fontsize=14)
    plt.xticks(rotation=45)
    for i, v in enumerate(tenure_churn['Churn_Rate']):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "tenure_group_churn_rates.png"), dpi=300)
    
    segmented_df, cluster_descriptions = perform_customer_segmentation(
        processed_df, numerical_features, output_dir
    )
    
    # Save detailed cluster analysis results
    cluster_profile = segmented_df.groupby('Cluster').agg({
        'tenure': ['mean', 'median', 'min', 'max'],
        'MonthlyCharges': ['mean', 'median', 'min', 'max'],
        'TotalCharges': ['mean', 'median', 'min', 'max'],
        'Churn': ['mean', 'count']
    })
    cluster_profile.columns = ['_'.join(col).strip() for col in cluster_profile.columns.values]
    cluster_profile['ChurnRate'] = cluster_profile['Churn_mean'] * 100
    cluster_profile['ClusterSize'] = cluster_profile['Churn_count']
    cluster_profile['ClusterPercentage'] = cluster_profile['ClusterSize'] / cluster_profile['ClusterSize'].sum() * 100
    
    # Save detailed cluster profiles
    cluster_profile.to_csv(os.path.join(csv_dir, "detailed_cluster_profiles.csv"))
    cluster_descriptions.to_csv(os.path.join(csv_dir, "cluster_descriptions.csv"), index=False)
    
    # Additional cluster visualization: Profile radar chart
    cluster_radar = pd.DataFrame({
        'Cluster': range(len(cluster_descriptions)),
        'AvgTenure': segmented_df.groupby('Cluster')['tenure'].mean(),
        'AvgMonthlyCharges': segmented_df.groupby('Cluster')['MonthlyCharges'].mean(),
        'ChurnRate': segmented_df.groupby('Cluster')['Churn'].mean() * 100
    })
    cluster_radar.to_csv(os.path.join(csv_dir, "cluster_radar_data.csv"), index=False)
    
    # Save standardized versions of key variables for comparison
    for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
        segmented_df[f"{col}_std"] = (segmented_df[col] - segmented_df[col].mean()) / segmented_df[col].std()
    
    radar_data = pd.DataFrame()
    for cluster in range(len(cluster_descriptions)):
        cluster_data = segmented_df[segmented_df['Cluster'] == cluster]
        data = {
            'Cluster': cluster,
            'Tenure_std': cluster_data['tenure_std'].mean(),
            'MonthlyCharges_std': cluster_data['MonthlyCharges_std'].mean(),
            'TotalCharges_std': cluster_data['TotalCharges_std'].mean(),
            'ChurnRate': cluster_data['Churn'].mean() * 100
        }
        radar_data = pd.concat([radar_data, pd.DataFrame([data])])
    
    radar_data.to_csv(os.path.join(csv_dir, "cluster_radar_standardized.csv"), index=False)
    
    model_results, model_comparison = build_predictive_models(segmented_df, output_dir)
    model_comparison.to_csv(os.path.join(csv_dir, "model_comparison_metrics.csv"), index=False)
    
    # Save detailed model results
    for model_name, result in model_results.items():
        report = result['report']
        # Convert classification report to dataframe
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(os.path.join(csv_dir, f"{model_name.replace(' ', '_').lower()}_classification_report.csv"))
        
        # Save confusion matrix
        conf_matrix = result['confusion_matrix']
        conf_df = pd.DataFrame(
            conf_matrix, 
            index=['Actual: No Churn', 'Actual: Churn'],
            columns=['Predicted: No Churn', 'Predicted: Churn']
        )
        conf_df.to_csv(os.path.join(csv_dir, f"{model_name.replace(' ', '_').lower()}_confusion_matrix.csv"))
    
    impact_analysis, roi_analysis = analyze_business_impact(
        df, model_results, output_dir
    )
    
    # Save additional business impact visualizations
    # Monthly revenue by customer segments
    segment_revenue = segmented_df.groupby('Cluster').agg({
        'MonthlyCharges': 'sum',
        'Churn': 'mean',
        'customerID': 'count'
    }).reset_index()
    segment_revenue.columns = ['Cluster', 'TotalMonthlyRevenue', 'ChurnRate', 'CustomerCount']
    segment_revenue['AtRiskRevenue'] = segment_revenue['TotalMonthlyRevenue'] * segment_revenue['ChurnRate']
    segment_revenue['PercentageOfTotalRevenue'] = segment_revenue['TotalMonthlyRevenue'] / segment_revenue['TotalMonthlyRevenue'].sum() * 100
    
    segment_revenue.to_csv(os.path.join(csv_dir, "segment_revenue_risk.csv"), index=False)
    
    # Revenue at risk visualization
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='Cluster', y='AtRiskRevenue', data=segment_revenue)
    plt.title("Monthly Revenue at Risk by Customer Segment", fontsize=16)
    plt.xlabel("Customer Segment", fontsize=14)
    plt.ylabel("At-Risk Monthly Revenue ($)", fontsize=14)
    
    # Add value annotations
    for i, v in enumerate(segment_revenue['AtRiskRevenue']):
        plt.text(i, v + 100, f"${v:.0f}", ha='center', fontsize=12)
        plt.text(i, v/2, f"{segment_revenue['ChurnRate'].iloc[i]*100:.1f}%\nchurn rate", 
                ha='center', va='center', fontsize=10, color='white')
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "revenue_at_risk_by_segment.png"), dpi=300)
    
    # Get feature importance for recommendations
    feature_importance = pd.read_csv(os.path.join(output_dir, "feature_importance.csv"))
    
    # Save feature importance in a more readable format
    top_features = feature_importance.head(10)
    top_features.to_csv(os.path.join(csv_dir, "top_10_churn_factors.csv"), index=False)
    
    recommendations = generate_recommendations(
        df, cluster_descriptions, feature_importance, output_dir
    )
    
    # Save recommendations in CSV format
    recommendation_df = pd.DataFrame(columns=['Category', 'Recommendation'])
    
    # Add overall recommendations
    for rec in recommendations['Overall Strategy']:
        recommendation_df = pd.concat([recommendation_df, pd.DataFrame([{
            'Category': 'Overall Strategy',
            'Recommendation': rec
        }])])
    
    # Add cluster-specific recommendations
    for cluster, details in recommendations['Cluster-Specific Tactics'].items():
        for rec in details['Recommendations']:
            recommendation_df = pd.concat([recommendation_df, pd.DataFrame([{
                'Category': f"{cluster} ({details['Description']})",
                'Recommendation': rec,
                'Priority': details['Priority'],
                'ChurnRisk': details['Churn Risk'],
                'CustomerBase': details['Customer Base']
            }])])
    
    # Add implementation plan
    for step in recommendations['Implementation Plan']:
        recommendation_df = pd.concat([recommendation_df, pd.DataFrame([{
            'Category': 'Implementation Plan',
            'Recommendation': step
        }])])
    
    recommendation_df.to_csv(os.path.join(csv_dir, "strategic_recommendations.csv"), index=False)
    
    extension_plans = define_extension_plans()
    
    # Save extension plans
    extension_df = pd.DataFrame(columns=['Category', 'Plan'])
    for category, plans in extension_plans.items():
        for plan in plans:
            extension_df = pd.concat([extension_df, pd.DataFrame([{
                'Category': category,
                'Plan': plan
            }])])
    
    extension_df.to_csv(os.path.join(csv_dir, "extension_plans.csv"), index=False)
    
    # Create a summary of all files generated
    all_files = []
    
    # Get all PNG files
    for file in os.listdir(figures_dir):
        if file.endswith('.png'):
            all_files.append({
                'Filename': file,
                'Type': 'Visualization (PNG)',
                'Path': os.path.join(figures_dir, file)
            })
    
    # Get all CSV files
    for file in os.listdir(csv_dir):
        if file.endswith('.csv'):
            all_files.append({
                'Filename': file,
                'Type': 'Data (CSV)',
                'Path': os.path.join(csv_dir, file)
            })
    
    # Save file summary
    file_summary = pd.DataFrame(all_files)
    file_summary.to_csv(os.path.join(output_dir, "file_summary.csv"), index=False)
    
    # Generate comprehensive report with all results
    report_path, html_report_path = generate_comprehensive_report(
        output_dir, file_summary, df, cluster_descriptions, 
        model_comparison, impact_analysis, roi_analysis
    )
    
    print("\nAnalysis complete! All results saved to the output directory.")
    print(f"\nGenerated {len(file_summary[file_summary['Type'] == 'Visualization (PNG)'])} visualization PNG files")
    print(f"Generated {len(file_summary[file_summary['Type'] == 'Data (CSV)'])} data CSV files")
    print(f"Comprehensive report saved to: {report_path}")
    print(f"HTML report saved to: {html_report_path}")
    
    return output_dir, file_summary, report_path, html_report_path

# Run the analysis if this script is executed directly
if __name__ == "__main__":
    import time
    import os
    start_time = time.time()
    
    output_directory, file_summary, report_path, html_report_path = run_telco_churn_analysis()
    
    execution_time = time.time() - start_time
    print(f"\nExecution completed in {execution_time:.2f} seconds")
    print(f"Results saved to: {output_directory}")
    print(f"Comprehensive report: {report_path}")
    print(f"HTML report (open in browser): {html_report_path}")
    
    # Print summary of key files for presentation
    print("\nKey files for presentation:")
    key_files = [
        "churn_distribution.csv",
        "contract_churn_rates.csv",
        "1_churn_distribution.png",
        "2_churn_by_contract.png",
        "revenue_at_risk_by_segment.png",
        "top_10_churn_factors.csv",
        "strategic_recommendations.csv",
        "Telco_Churn_Analysis_Report.md",
        "Telco_Churn_Analysis_Report.html"
    ]
    
    for file in key_files:
        matching_files = file_summary[file_summary['Filename'].str.contains(file)]
        if not matching_files.empty:
            print(f"- {matching_files.iloc[0]['Filename']} ({matching_files.iloc[0]['Path']})")