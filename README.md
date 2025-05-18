Telco Customer Churn Analysis: A Data-Driven Strategy for Revenue Retention
Predicting and preventing customer churn to save millions in annual revenue

📋 Overview
This project addresses the critical business problem of customer churn in the telecommunications industry. Using machine learning and customer segmentation, we provide actionable insights for Chief Revenue Officers to implement targeted retention strategies that can save over $170,000 annually with just a 20% reduction in churn rates.
Key Findings:

Month-to-month customers churn at 42.7% vs. 2.8% for two-year contracts
Customer segmentation reveals four distinct groups with varying churn risk (5% to 48.3%)
Predictive models achieve 80% accuracy in identifying at-risk customers
ROI of retention efforts exceeds 1,000%

🗂️ Repository Structure
telco-churn-analysis/
│
├── data/
│   └── WA_FnUseC_TelcoCustomerChurn.csv          # Raw telco customer dataset
│
├── src/
│   └── final_presentation.py                     # Main analysis script
│
├── results/
│   ├── figures/                                  # Generated visualizations
│   │   ├── 1_churn_distribution.png
│   │   ├── 2_churn_by_contract.png
│   │   ├── 7_customer_segments.png
│   │   ├── 10_feature_importance.png
│   │   └── ... (additional visualizations)
│   │
│   └── csv_results/                              # Analysis outputs in CSV format
│       ├── model_comparison.csv
│       ├── cluster_descriptions.csv
│       ├── strategic_recommendations.csv
│       └── ... (additional data files)
│
├── requirements.txt                              # Python dependencies
├── README.md                                     # This file
└── LICENSE                                       # MIT License
🚀 Quick Start
Prerequisites

Python 3.8 or higher
pip package manager

Installation

Clone the repository
bashgit clone https://github.com/YOUR_USERNAME/telco-churn-analysis.git
cd telco-churn-analysis

Create a virtual environment (recommended)
bashpython -m venv churn_env
source churn_env/bin/activate  # On Windows: churn_env\Scripts\activate

Install dependencies
bashpip install -r requirements.txt


Running the Analysis
Complete Analysis Pipeline:
bashpython src/final_presentation.py
This will:

Load and clean the telco customer data
Perform exploratory data analysis
Execute customer segmentation using K-means clustering
Build predictive models (Random Forest & Logistic Regression)
Generate business impact analysis
Create strategic recommendations
Save all visualizations and results to the results/ directory

Expected Runtime: 2-3 minutes on standard hardware
📊 Data Description
Dataset: Telco Customer Churn Dataset

Size: 7,032 customers × 21 features
Source: IBM Watson Analytics
Features:

Demographics (gender, age, family status)
Services (phone, internet, security, streaming)
Contract details (type, tenure, billing)
Financial metrics (monthly charges, total charges)
Target variable: Churn (Yes/No)



Data Quality Issues Addressed:

Converted TotalCharges from string to numeric format
Handled 11 records with missing values
Applied consistent categorical encoding
Created standardized features for clustering

🔬 Methodology
Course Methods Applied
1. Unsupervised Learning (Module 1):

K-means clustering for customer segmentation
Elbow method for optimal cluster selection
Feature scaling and standardization

2. Supervised Learning (Module 2):

Random Forest for feature importance and robust classification
Logistic Regression for interpretable probability estimates
Cross-validation for model evaluation

Analysis Pipeline

Data Preprocessing: Clean, encode, and prepare data
Exploratory Analysis: Identify key churn patterns
Customer Segmentation: Group customers by behavior
Predictive Modeling: Build churn prediction models
Business Impact Analysis: Calculate ROI and revenue impact
Strategic Recommendations: Develop actionable insights

📈 Key Results
Customer Segments Identified
SegmentDescriptionSizeChurn RatePriorityCluster 0Long-term, moderate spend24.1%24.8%ImportantCluster 1New, high-value27.1%15.4%ImportantCluster 2Mid-term, premium16.5%5.0%MonitorCluster 3New, budget-conscious32.3%48.3%Critical
Model Performance
ModelAccuracyPrecisionRecallF1-ScoreRandom Forest78.9%77.9%78.9%78.2%Logistic Regression79.9%79.4%79.9%79.6%
Business Impact

Annual Revenue Loss: $5.33M due to churn
Customer Lifetime Value: $2,101
ROI of 10% Churn Reduction: 1,178%
Potential Annual Savings: $32K-$176K depending on reduction target

🎯 Strategic Recommendations
Immediate Actions (0-3 months)

Target Cluster 3 with specialized retention programs
Convert month-to-month customers to annual contracts
Implement predictive scoring for proactive intervention

Medium-term Initiatives (3-12 months)

Enhance service value through security and tech support improvements
Deploy real-time churn dashboard for customer service teams

Long-term Strategy (12+ months)

Build comprehensive CRM with integrated churn prediction
Establish dynamic pricing based on churn risk

📚 Documentation

Detailed Analysis: Medium Article
Methodology: See docstrings in final_presentation.py
Results: Comprehensive reports in results/ directory

🔧 Customization
Running Specific Analyses
Customer Segmentation Only:
pythonfrom src.final_presentation import perform_customer_segmentation
# Modify script to run only clustering analysis
Predictive Modeling Only:
pythonfrom src.final_presentation import build_predictive_models
# Modify script to run only model training and evaluation
Parameter Tuning
Key parameters you can adjust in final_presentation.py:

n_clusters in K-means clustering (default: 4)
random_state for reproducibility (default: 42)
test_size for train-test split (default: 0.25)
Cost assumptions for ROI calculation

🤝 Contributing
This project was developed as part of INST414 coursework. While not actively maintained, feedback and suggestions are welcome:

Fork the repository
Create a feature branch
Submit a pull request with detailed description

📞 Contact
Author: Rhyss Idham

Email: rhysshaiqal2002@gmail.com


Course: INST414 - Data Science Techniques
Institution: University of Maryland
Semester: Spring 2025
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgments

Dataset: IBM Watson Analytics Telco Customer Churn Dataset
Libraries: pandas, scikit-learn, matplotlib, seaborn, numpy
Inspiration: Real-world telecommunications industry challenges

🔄 Version History

v1.0.0 (May 2025): Initial release with complete analysis pipeline
v0.5.0 (April 2025): Beta version with core functionality
v0.1.0 (March 2025): Project inception and data exploration


⭐ If you find this analysis useful, please consider starring the repository!
bash# Quick command summary for getting started:
git clone https://github.com/YOUR_USERNAME/telco-churn-analysis.git
cd telco-churn-analysis
pip install -r requirements.txt
python src/final_presentation.py
For detailed methodology and business insights, read the full analysis on Medium.
