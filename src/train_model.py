from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score
import numpy as np

def threshold_optimized_f1(y_true, y_pred, threshold_list):
    best_f1 = 0
    best_threshold = 0
    
    for t in threshold_list:
        y_pred_binary = (y_pred > t).astype(np.int8)
        current_v_f1 = f1_score(y_true, y_pred_binary)
        
        if(current_v_f1 > best_f1):
            best_threshold = t
            best_f1 = current_v_f1
            
    return best_threshold
    

def custom_f1_threshold(y_true, y_probs, threshold=0.5):
    y_pred = (y_probs>threshold).astype(int)
    return f1_score(y_true, y_pred)
    