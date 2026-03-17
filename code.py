# dataset from https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis?resource=download&select=orders.csv
# to activate venv "source .venv/bin/activate" in terminal

import pandas as pd
import numpy as np
import datetime as dt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 1000)

#------------------- Importing Data from CSV ----------------

orders = pd.read_csv("/Users/darianyee/Desktop/Coding/Project 1 - InstaCart/dataset/orders.csv")
# | Column                   | Meaning                    |
# | ------------------------ | -------------------------- |
# | `order_id`               | Unique order identifier    |
# | `user_id`                | Customer identifier        |
# | `eval_set`               | `prior` / `train` / `test` |
# | `order_number`           | Order sequence per user    |
# | `order_dow`              | Day of week (0–6)          |
# | `order_hour_of_day`      | Hour order was placed      |
# | `days_since_prior_order` | Gap since last order       |

order_products_prior = pd.read_csv("/Users/darianyee/Desktop/Coding/Project 1 - InstaCart/dataset/order_products__prior.csv")
# | Column              | Meaning                              |
# | ------------------- | ------------------------------------ |
# | `order_id`          | Which order                          |
# | `product_id`        | Which product                        |
# | `add_to_cart_order` | Sequence added to cart               |
# | `reordered`         | 1 if user bought this product before |

order_products_train = pd.read_csv("/Users/darianyee/Desktop/Coding/Project 1 - InstaCart/dataset/order_products__train.csv")
# | Column              | Meaning                              |
# | ------------------- | ------------------------------------ |
# | `order_id`          | Which order                          |
# | `product_id`        | Which product                        |
# | `add_to_cart_order` | Sequence added to cart               |
# | `reordered`         | 1 if user bought this product before |

products = pd.read_csv("/Users/darianyee/Desktop/Coding/Project 1 - InstaCart/dataset/products.csv")
# | Column          | Meaning             |
# | --------------- | ------------------- |
# | `product_id`    | Product key         |
# | `product_name`  | Product description |
# | `aisle_id`      | Sub-category        |
# | `department_id` | High-level category |

aisles = pd.read_csv("/Users/darianyee/Desktop/Coding/Project 1 - InstaCart/dataset/aisles.csv")
# | Column     | Meaning    |
# | ---------- | ---------- |
# | `aisle_id` | Join key   |
# | `aisle`    | Aisle name |

departments = pd.read_csv("/Users/darianyee/Desktop/Coding/Project 1 - InstaCart/dataset/departments.csv")
# | Column          | Meaning         |
# | --------------- | --------------- |
# | `department_id` | Join key        |
# | `department`    | Department name |

#------------------- Getting User Features ----------------

# Question: Predict which products a user will reorder in their next instacart purchase
# This means we are doing a binary classification
# Unit of prediction its between (user_id and product_id)
# Output = 1 (will rorder) or 0 (will not reorder)

#1.1 Obtain the test users
test_users = orders[(orders['eval_set']=='test')]['user_id'].copy()
train_users = orders[(orders['eval_set']=='train')]['user_id'].copy()

#1.2 Obtain the prior orders for the test users
test_users_prior_orders = orders[(orders['eval_set']=='prior') & (orders['user_id'].isin(test_users))].copy()

#1.3 Obtain the most recent prior order for the test user and update eval set to 'test_synthetic'
test_user_most_recent_order_number = test_users_prior_orders.groupby(['user_id'])['order_number'].transform('max')
synthetic_test_df = test_users_prior_orders[test_users_prior_orders['order_number']==test_user_most_recent_order_number]

synthetic_test_order_ids = synthetic_test_df['order_id'].copy()
orders.loc[(orders['order_id'].isin(synthetic_test_order_ids)), 'eval_set']='test_synthetic'

#1.4 Removing the 'test_synthetic' orders from 'order_products_prior'
order_products_test = order_products_prior[order_products_prior['order_id'].isin(synthetic_test_order_ids)].copy()
order_products_prior = order_products_prior[~order_products_prior['order_id'].isin(synthetic_test_order_ids)].copy()

#2.1 Train, Validate split our train orders data

from sklearn.model_selection import train_test_split

y_train_users, y_validate_users = train_test_split(train_users, test_size=0.20, random_state=123)

orders.loc[(orders['user_id'].isin(y_validate_users)) & (orders['eval_set']=='train'), 'eval_set'] = 'validate'

validate_orders = orders[(orders['eval_set']=='validate')]['order_id']
order_products_validate = order_products_train[order_products_train['order_id'].isin(validate_orders)]
order_products_train = order_products_train[~order_products_train['order_id'].isin(validate_orders)]

#Converting data types to catagories
aisles['aisle'] = aisles['aisle'].astype('category')
departments['department'] = departments['department'].astype('category')
orders['eval_set'] = orders['eval_set'].astype('category')
products['product_name'] = products['product_name'].astype('category')

# Merging to get the product data

prior_orders = orders[orders['eval_set']=='prior'].copy()

op = prior_orders.merge(
    order_products_prior,
    how='inner',
    on='order_id'
)

#3.1.1 User Total Orders
user = op.groupby(['user_id'])['order_number'].max().to_frame('u_total_orders').reset_index()

#3.1.2 User Reorder Ratio
reorder_feature = op.groupby('user_id')['reordered'].mean().to_frame('u_reorder_ratio').reset_index()
user = user.merge(
    reorder_feature,
    how='left',
    on='user_id'
)

#3.1.3 User Number of Unique Products
unique_products_feature = op.groupby('user_id')['product_id'].nunique().to_frame('u_#unique_products').reset_index()
user = user.merge(
    unique_products_feature,
    how='left',
    on='user_id'
)

#3.1.4 User Mode Day of Week
unique_orders = op[['user_id', 'order_id', 'order_dow', 'order_hour_of_day']].drop_duplicates().copy()

dow_mode_feature = unique_orders.groupby(['user_id'])['order_dow'].agg(lambda x:x.mode().iloc[0]).to_frame('u_dow_mode').reset_index()
user = user.merge(
    dow_mode_feature,
    how='left',
    on='user_id'
)

#3.1.5 User Mode Hour of Day
hour_mode_feature = unique_orders.groupby(['user_id'])['order_hour_of_day'].agg(lambda x:x.mode().iloc[0]).to_frame('u_hr_of_day_mode').reset_index()
user = user.merge(
    hour_mode_feature,
    how='left',
    on='user_id'
)

#3.1.6 User Avg Basket Size
basket_size = op.groupby(['user_id', 'order_number'])['product_id'].count().to_frame('basket_size').reset_index()
u_avg_basket_size=basket_size.groupby(['user_id'])['basket_size'].mean().to_frame('u_avg_basket_size').reset_index()

user=user.merge(
    u_avg_basket_size,
    how='left',
    on='user_id'
)

#3.1.7 User Avg Days between orders
days_between_order = op.groupby(['user_id', 'order_id'])['days_since_prior_order'].first().to_frame('days_between_order').reset_index()
u_avg_days_between_order = days_between_order.groupby(['user_id'])['days_between_order'].mean().to_frame('u_avg_days_since_last_order').reset_index()
user=user.merge(
    u_avg_days_between_order,
    how='left',
    on='user_id'
)

#3.1.8 User Avg Basket Reorder Rate
order_reorder_rate = op.groupby(['user_id', 'order_id'])['reordered'].mean().to_frame('basket_reorder_rate').reset_index()
u_avg_reorder_rate = order_reorder_rate.groupby('user_id')['basket_reorder_rate'].mean().to_frame('u_avg_basket_reorder_rate').reset_index()

# 3. Merge into your user table
user = user.merge(
    u_avg_reorder_rate, 
    on='user_id', 
    how='left')

#3.2 Creating Product Feattures

#3.2.1 Product Total Purchases
product = op.groupby(['product_id'])['order_id'].count().to_frame('p_total_purchase').reset_index()


#3.2.2 Product Reorder Ratio
p_reorder = op.groupby(['product_id'])['reordered'].mean().to_frame('p_reorder_rate').reset_index()
product = product.merge(
    p_reorder,
    how='left',
    on='product_id'
)

#3.3 Creating User-Product Feattures

#3.3.1 User-Product Total Purchased
user_product = op.groupby(['user_id','product_id'])['order_id'].count().to_frame('up_total_purchased').reset_index()

#3.3.2 User-Product Reorder Ratio
up_stats = op.groupby(['user_id', 'product_id'])['order_number'].agg(['min', 'max']).reset_index()
up_stats.columns=['user_id', 'product_id', 'up_first_order', 'up_last_order']

user_product = user_product.merge(
    up_stats,
    how='left',
    on=['user_id', 'product_id']
)

user_product['up_reorder_ratio'] = ((user_product['up_total_purchased'] - 1) / (user_product['up_last_order'] - user_product['up_first_order'])).fillna(0)

#3.3.3 User-Product Avg days between purchase

up_avg_days_between_purchase = op.groupby(['user_id', 'product_id'])['days_since_prior_order'].mean().to_frame('up_avg_days_between_purchase').reset_index()
user_product = user_product.merge(
    up_avg_days_between_purchase,
    how='left',
    on=['user_id', 'product_id']
)

#3.3.4 User-Product Conecutive Streak

#Getting the most recent order number by user-product
op_sorted = op.sort_values(['user_id', 'product_id', 'order_number'], ascending=[True, True, False])
op_sorted['user_max_order'] = op_sorted.groupby('user_id')['order_number'].transform('max')


op_sorted['order_diff'] = op_sorted['user_max_order'] - op_sorted['order_number']
op_sorted['up_order_rank'] = op_sorted.groupby(['user_id', 'product_id']).cumcount()

op_sorted['is_streak'] = (op_sorted['order_diff'] == op_sorted['up_order_rank'])

up_streak = op_sorted[op_sorted['is_streak']].groupby(['user_id', 'product_id']).size().to_frame('up_consecutive_streak').reset_index()

user_product = user_product.merge(
    up_streak, 
    on=['user_id', 'product_id'],
    how='left'
).fillna(0)


#Merge all the fetures together from the prior data

df = user_product.merge(
    user,
    how='left',
    on='user_id'
)

df = df.merge(
    product,
    how='left',
    on='product_id'
)

df.head()

future_orders = orders[orders['eval_set'].isin(['train', 'validate', 'test_synthetic'])].copy()
future_orders.head()

df = df.merge(
    future_orders,
    how='left',
    on='user_id'
)
df.head()

#Getting only the training data

model_train_data = df[df['eval_set']=='train']
model_train_data.head()

model_train_data = model_train_data.merge(
    order_products_train[['order_id','product_id','reordered']],
    how='left',
    on=['order_id', 'product_id']
)

model_train_data['reordered'] = model_train_data['reordered'].fillna(0)

#Getting only the validate data

model_validate_data = df[df['eval_set']=='validate']
model_validate_data.head()

model_validate_data = model_validate_data.merge(
    order_products_validate[['order_id','product_id','reordered']],
    how='left',
    on=['order_id', 'product_id']
)

model_validate_data['reordered'] = model_validate_data['reordered'].fillna(0)

#Getting only the test data

model_test_synthetic_data = df[df['eval_set']=='test_synthetic']
model_test_synthetic_data.head()

model_test_synthetic_data = model_test_synthetic_data.merge(
    order_products_test[['order_id','product_id','reordered']],
    how='left',
    on=['order_id', 'product_id']
)

model_test_synthetic_data['reordered'] = model_test_synthetic_data['reordered'].fillna(0)

#################################
#Merge all the fetures together from the prior data

df = user_product.merge(
    user,
    how='left',
    on='user_id'
)

df = df.merge(
    product,
    how='left',
    on='product_id'
)

df.head()

future_orders = orders[orders['eval_set'].isin(['train', 'validate', 'test_synthetic'])].copy()
future_orders.head()

df = df.merge(
    future_orders,
    how='left',
    on='user_id'
)
df.head()

#Getting only the training data

model_train_data = df[df['eval_set']=='train']
model_train_data.head()

model_train_data = model_train_data.merge(
    order_products_train[['order_id','product_id','reordered']],
    how='left',
    on=['order_id', 'product_id']
)

model_train_data['reordered'] = model_train_data['reordered'].fillna(0)

model_train_data['reordered'].value_counts()

order_products_train['reordered'].value_counts()

#Getting only the validate data

model_validate_data = df[df['eval_set']=='validate']
model_validate_data.head()

model_validate_data = model_validate_data.merge(
    order_products_validate[['order_id','product_id','reordered']],
    how='left',
    on=['order_id', 'product_id']
)

model_validate_data['reordered'] = model_validate_data['reordered'].fillna(0)

#Getting only the test data

model_test_synthetic_data = df[df['eval_set']=='test_synthetic']
model_test_synthetic_data.head()

model_test_synthetic_data = model_test_synthetic_data.merge(
    order_products_test[['order_id','product_id','reordered']],
    how='left',
    on=['order_id', 'product_id']
)

model_test_synthetic_data['reordered'] = model_test_synthetic_data['reordered'].fillna(0)

model_test_synthetic_data.head()

import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
from sklearn.metrics import average_precision_score

# --- 1. Prepare Validation Data (Define dval and y_val here) ---
cols_to_drop = ['order_id', 'eval_set', 'reordered']

# --- 2. Prepare Training Raw Data ---
X_raw = model_train_data.drop(columns=cols_to_drop)
y_raw = model_train_data['reordered']

train_ratio_orig = (y_raw==0).sum() / (y_raw==1).sum()
print(f"Original Training ratio balance: 1:{train_ratio_orig:.2f}")

# We prepare this ONCE outside the loop to save time and memory
X_val = model_validate_data.drop(columns=cols_to_drop)
y_val = model_validate_data['reordered'] 
dval = xgb.DMatrix(X_val) # Now dval is defined and ready for the loop

# --- 3. Ratio Testing Loop ---
ratios_to_test = np.arange(7, 9.5, 0.5).tolist()
results = []

for r in ratios_to_test:
    # Apply Undersampling
    target_non_reorder_count = int((y_raw == 1).sum() * r)
    rus_test = RandomUnderSampler(sampling_strategy={0: target_non_reorder_count}, random_state=123)
    X_train_temp, y_train_temp = rus_test.fit_resample(X_raw, y_raw)
    
    dtrain_temp = xgb.DMatrix(X_train_temp, label=y_train_temp)
    
    params_temp = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 6,
        'learning_rate': 0.1,
        'tree_method': 'hist',
        'scale_pos_weight': r  
    }
    
    # Train
    model_temp = xgb.train(params_temp, dtrain_temp, num_boost_round=100)
    
    # Evaluate PR AUC
    train_probs = model_temp.predict(dtrain_temp)
    val_probs = model_temp.predict(dval) # This will now work!
    
    train_pr_auc = average_precision_score(y_train_temp, train_probs)
    val_pr_auc = average_precision_score(y_val, val_probs)
    
    results.append({
        'ratio': f"1:{r}",
        'train_pr_auc': train_pr_auc,
        'val_pr_auc': val_pr_auc
    })

# --- 4. Results ---
ratio_df = pd.DataFrame(results)
print("\n--- Summary of Ratio Testing ---")
print(ratio_df)

from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
from sklearn.metrics import f1_score, classification_report, average_precision_score

# --- 1. Data Preparation ---
cols_to_drop = ['eval_set', 'reordered', 'order_id']
X_raw = model_train_data.drop(columns=cols_to_drop, errors='ignore')
y_raw = model_train_data['reordered']

# --- 2. Calculate the 1:8 Target ---
current_reorder_count = (y_raw == 1).sum()
target_non_reorder_count = current_reorder_count * 8

# --- 3. Apply Undersampling ---
rus = RandomUnderSampler(sampling_strategy={0: target_non_reorder_count}, random_state=42)
X_train, y_train = rus.fit_resample(X_raw, y_raw)

# Convert to DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=X_train.columns.tolist())

# --- 4. Train Model (xbg) ---
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 6,
    'learning_rate': 0.1,
    'tree_method': 'hist',
    'min_child_weight': 100,
    'scale_pos_weight': 8 
}

xbg = xgb.train(params, dtrain, num_boost_round=100)

# --- 5. Evaluation Setup ---
threshold = 0.70

# --- Evaluate on Training Data ---
y_train_probs = xbg.predict(dtrain)
y_train_pred = (y_train_probs > threshold).astype(int)
train_pr_auc = average_precision_score(y_train, y_train_probs)

print(f"--- Evaluation on model_train_data (1:8 Undersampled) ---")
print(f"Train PR AUC: {train_pr_auc:.4f}")
print(f"Train F1-Score: {f1_score(y_train, y_train_pred):.4f}")

# --- Evaluate on model_validate_data ---
X_val = model_validate_data.drop(columns=cols_to_drop, errors='ignore')
y_val = model_validate_data['reordered']
dval = xgb.DMatrix(X_val)

y_val_probs = xbg.predict(dval)
y_val_pred = (y_val_probs > threshold).astype(int)
val_pr_auc = average_precision_score(y_val, y_val_probs)

print(f"\n--- Evaluation on model_validate_data ---")
print(f"Val PR AUC: {val_pr_auc:.4f}")
print(f"Val F1-Score: {f1_score(y_val, y_val_pred):.4f}")

# --- Evaluate on model_test_data ---
# I am adding this section to evaluate your final test set
X_test = model_test_synthetic_data.drop(columns=cols_to_drop, errors='ignore')
y_test = model_test_synthetic_data['reordered']
dtest = xgb.DMatrix(X_test)

y_test_probs = xbg.predict(dtest)
y_test_pred = (y_test_probs > threshold).astype(int)
test_pr_auc = average_precision_score(y_test, y_test_probs)

print(f"\n--- Evaluation on model_test_data ---")
print(f"Test PR AUC: {test_pr_auc:.4f}")
print(f"Test F1-Score: {f1_score(y_test, y_test_pred):.4f}")
print(classification_report(y_test, y_test_pred))