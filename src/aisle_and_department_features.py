import pandas as pd 

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

    aisle_department_features = aisle_department_features.drop(columns = ['order_id', 'reordered'])
    aisle_department_features = aisle_department_features.drop_duplicates(subset=['user_id', 'product_id', 'aisle_id', 'department_id'])
    
    return aisle_department_features
    