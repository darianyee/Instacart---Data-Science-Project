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