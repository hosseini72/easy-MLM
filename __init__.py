
from model.config import *


x= knn_config()
# print(x.parameters)
x.set_config({
    'n_neighbors'=5, # int, default=5
    weights='uniform', # {'uniform', 'distance'}, callable or None, default='uniform'
    algorithm='auto', # {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
    leaf_size=30, # int, default=30
    p=2, # int, default=30
    metric='minkowski', # str or callable, default='minkowski'
    metric_params=None, # dict, default=None
    n_jobs=None # int, default=None
    })