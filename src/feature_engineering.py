import pandas as pd 

def get_user_features(prior_orders):
    
    #3.1.1 User total orders
    #3.1.2 User reorder ratio 
    #3.1.3 User Number of Unique Products
    #3.1.4 User Mode Day of Week
    #3.1.5 User Mode Hour of Day
    user_features = prior_orders.groupby(['user_id']).agg(
        u_total_orders = ('order_number','max'),
        u_reorder_ratio = ('reordered', 'mean'),
        u_unique_products = ('product_id', 'nunique')
    ).reset_index()
    
    #3.1.6 User Avg Basket Size
    #3.1.7 User Avg Days between orders
    #3.1.8 User Avg Basket Reorder rate
    prep_user_order_features=prior_orders.groupby(['user_id', 'order_number']).agg(
        order_dow = ('order_dow', 'first'),
        order_hour_of_day = ('order_hour_of_day', 'first'),
        basket_size = ('add_to_cart_order', 'max'),
        days_since_prior_order = ('days_since_prior_order', 'first'),
        basket_reorder_rate = ('reordered', 'mean')
    ).reset_index()
    
    user_order_features = prep_user_order_features.groupby(['user_id']).agg(
        u_dow_mode = ('order_dow', lambda x:x.mode().iloc[0]),
        u_hour_of_day_mode = ('order_hour_of_day', lambda x:x.mode().iloc[0]),
        u_avg_basket_size = ('basket_size', 'mean'),
        u_avg_days_since_prior_order = ('days_since_prior_order', 'mean'),
        u_avg_basket_reorder_rate = ('basket_reorder_rate', 'mean')
    ).reset_index()
    
    
    user_features = user_features.merge(
        user_order_features,
        how='left',
        on='user_id'
    )
    
    return user_features


def get_product_features(prior_orders):
    
    product_features = prior_orders.groupby(['product_id']).agg(
        #3.2.1 Product Total Purchases
        #3.2.2 Product Reorder Ratio
        p_total_purchases = ('product_id', 'count'),
        p_reorder_ratio = ('reordered', 'mean')
    ).reset_index()
    
    return product_features

def get_user_product_features(prior_orders):
    
    
    #3.3.1 User-Product Total Purchased
    #3.3.3 User-Product Avg days between purchase
    #3.3.5 User-Product avg add to cart order
    user_product_features = prior_orders.groupby(['user_id', 'product_id']).agg(
        up_total_purchases = ('order_id', 'count'),
        up_first_order = ('order_number', 'min'),
        up_last_order = ('order_number', 'max'),
        up_reorder_ratio = ('reordered', 'mean'),
        up_avg_days_between_purchase = ('days_since_prior_order', 'mean'),
        up_avg_add_to_cart_order = ('add_to_cart_order', 'mean')

    ).reset_index()
    
    #3.3.2 User-Product Reorder Ratio
    user_product_features['up_reorder_ratio'] = (user_product_features['up_total_purchases'] - 1) / (user_product_features['up_last_order'] - user_product_features['up_first_order'] + 1).fillna(0)
    
    #3.3.4 User-Product Consecutive Streak
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


def get_aisle_department_features(prior_orders):
    
    #3.4.1 Aisle reorder ratio
    #3.4.2 Aisle total orders
    #3.4.3 Department reorder ratio
    #3.4.4 Department total orders
    #3.4.5 User-Aisle reorder ratio
    #3.4.6 User-Aisle total orders
    #3.4.7 User-Department reorder ratio
    #3.4.8 User-Department total orders

    aisle_features = prior_orders.groupby(['aisle_id']).agg(
        a_reorder_ratio = ('reordered', 'mean'),
        a_total_orders = ('product_id', 'count')
    ).reset_index()

    user_aisle_features = prior_orders.groupby(['user_id', 'aisle_id']).agg(
        ua_reorder_ratio = ('reordered', 'mean'),
        ua_total_orders = ('product_id', 'count')
    ).reset_index()

    department_features = prior_orders.groupby(['department_id']).agg(
        d_reorder_ratio = ('reordered', 'mean'),
        d_total_orders = ('product_id', 'count')
    ).reset_index()

    user_department_features = prior_orders.groupby(['user_id', 'department_id']).agg(
        ud_reorder_ratio = ('reordered', 'mean'),
        ud_total_orders = ('product_id', 'count')
    ).reset_index()

    aisle_department_features = prior_orders.merge(
        aisle_features,
        how='left',
        on='aisle_id'
    ).merge(
        user_aisle_features,
        how='left',
        on=['user_id', 'aisle_id']
    ).merge(
        department_features,
        how='left',
        on='department_id'
    ).merge(
        user_department_features,
        how='left',
        on=['user_id', 'department_id']
        
    )

    aisle_department_features = aisle_department_features.drop(columns = ['order_id', 'reordered'], axis=1)
    aisle_department_features = aisle_department_features.drop_duplicates(subset=['user_id', 'product_id', 'aisle_id', 'department_id'])
    
    return aisle_department_features
    