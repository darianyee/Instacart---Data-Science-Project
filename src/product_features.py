import pandas as pd 

def get_product_features(prior_orders):
    
    product_features = prior_orders.groupby(['product_id']).agg(
        #3.2.1 Product Total Purchases
        #3.2.2 Product Reorder Ratio
        p_total_purchases = ('product_id', 'count'),
        p_reorder_ratio = ('reordered', 'mean')
    ).reset_index()
    
    return product_features