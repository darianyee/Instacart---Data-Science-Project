import pandas as pd 
def get_user_features(prior_orders):
    
    #3.1.1 User total orders
    #3.1.2 User reorder ratio 
    #3.1.3 User Number of Unique Products
    user_features = prior_orders.groupby(['user_id']).agg(
        u_total_orders = ('order_number','max'),
        u_reorder_ratio = ('reordered', 'mean'),
        u_unique_products = ('product_id', 'nunique')
    )
    
    #3.1.6 User Avg Basket Size
    #3.1.7 User Avg Days between orders
    prep_user_order_features=prior_orders.groupby(['user_id', 'order_number']).agg(
        basket_size = ('add_to_cart_order', 'max'),
        basket_reorder_rate = ('reordered', 'mean')
    ).reset_index()
    
    #user_order_features = prep_user_order_features.groupby(['user_id'])
    
    #3.1.4 User Mode Day of Week
    #3.1.5 User Mode Hour of Day
    
    #3.1.8 User Avg Basket Reorder rate
    
    return prep_user_order_features