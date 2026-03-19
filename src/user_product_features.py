#3.3.1 User-Product Total Purchased
#3.3.2 User-Product Reorder Ratio
#3.3.3 User-Product Avg days between purchase
#3.3.4 User-Product Consecutive Streak

#Getting the most recent order number by user-product

#3.3.5 User-Product avg add to cart order

import pandas as pd 

def get_user_product_features(prior_orders):
    
    user_product_features = prior_orders.groupby(['user_id', 'product_id']).agg(
        up_total_purchases = ('order_id', 'count'),
        up_first_order = ('order_number', 'min'),
        up_last_order = ('order_number', 'max'),
        up_reorder_ratio = ('reordered', 'mean'),
        up_avg_days_between_purchase = ('days_since_prior_order', 'mean'),
        up_avg_add_to_cart_order = ('add_to_cart_order', 'mean')

    ).reset_index()
    
    user_product_features['up_reorder_ratio'] = (user_product_features['up_total_purchases'] - 1) / (user_product_features['up_last_order'] - user_product_features['up_first_order'] + 1).fillna(0)
    
    
    prior_sorted = prior_orders.sort_values(['user_id', 'product_id', 'order_number'], ascending=[True, True, False])
    prior_sorted['user_max_order'] = prior_sorted.groupby('user_id')['order_number'].transform('max')


    prior_sorted['order_diff'] = prior_sorted['user_max_order'] - prior_sorted['order_number']
    prior_sorted['up_order_rank'] = prior_sorted.groupby(['user_id', 'product_id']).cumcount()

    prior_sorted['is_streak'] = (prior_sorted['order_diff'] == prior_sorted['up_order_rank'])

    up_streak = prior_sorted[prior_sorted['is_streak']].groupby(['user_id', 'product_id']).size().to_frame('up_consecutive_streak').reset_index()

    user_product_features = user_product_features.merge(
        up_streak, 
        on=['user_id', 'product_id'],
        how='left'
    ).fillna(0)
    
    return user_product_features