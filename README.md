Instacart Market Basket Analysis: Reorder Prediction

This project aims to predict whether a customer will reorder a previously purchased product in their next transaction using the Instacart Market Basket Analysis dataset from Kaggle. By analyzing over 3 million grocery orders from 200,000 users, we developed a predictive pipeline to enhance user experience through personalized recommendations.

🚀 Project Overview
The core challenge of this project is to model consumer behavior and shopping rituals. Since the dataset was originally for a competition, the test labels are withheld. To build a valid end-to-end solution, we:
- Synthesized a ground-truth test set to evaluate performance.
- Engineered multi-level features (User, Product, and User-Product interactions).
- Addressed class imbalance (1:9 ratio) using cost-sensitive learning.
- Evaluated multiple GBDT models to find the most robust predictor.

📊 Evaluation MetricsGiven the highly unbalanced nature of the data, standard Accuracy is misleading. We utilized:

- PR-AUC (Precision-Recall Area Under Curve): Our primary tuning metric, as it focuses on the model's ability to catch reorders (the minority class).
- F1-Score: Used to determine the final classification threshold, balancing the trade-off between Precision and Recall.

Model Benchmarks
<pre>
  XGBoost
  Train PR-AUC:              0.450
  Validation PR-AUC:         0.428
  Test PR-AUC:               0.435

  LightGBM
  Train PR-AUC:              0.440
  Validation PR-AUC:         0.428  
  Test PR-AUC:               0.435
</pre>

🛠️ Feature Engineering

We transformed raw transaction data into predictive insights across several hierarchies:

- User Level: Average order size, purchase frequency, and overall reorder ratio.
- Product Level: Reorder frequency and department/aisle popularity.
- User-Product (The "Relationship"): The number of times a user bought a specific item, their "streak" of consecutive purchases, and the number of orders since they last bought it.

Top Predictors

Our analysis revealed that the most influential features are:
1. days_since_prior_order: Captures the customer's weekly or monthly shopping rhythm.
2. up_orders_since_last_purchase: Identifies when a user has likely moved on from a specific product.
3. up_order_rate: Measures the strength of the habit for a specific item.
  
Opted against an ensemble of trees, as the high correlation between GBDT models offered diminishing returns compared to the memory overhead required.

📁 Repository Structure data: (Not included due to size) Kaggle source files.notebooks/: Exploratory Data Analysis (EDA) and Model Training.src/: Python scripts for feature engineering and downcasting.models/: Saved model weights and evaluation plots.
