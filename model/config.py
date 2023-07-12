from enum import Enum


class LogRegressionConfig(Enum):
    ONE= {'solver': "sag", 'penalty': "l2", 'multi_class': "auto", 'l1_ratio':0},
    TWO= {'solver': "saga", 'penalty': "l1", 'multi_class': "auto"},
    THREE= {'solver': "saga", 'penalty': "l2", 'multi_class': "auto"},
    FOUR=  {'solver': "saga", 'penalty': "elasticnet", 'multi_class': "auto"},
    FIVE=  {'solver': "liblinear", 'penalty': "l1", 'multi_class': "auto"},
    SIX=   {'solver': "liblinear", 'penalty': "l2", 'multi_class': "auto"},
    SEVEN= {'solver': "lbfgs", 'penalty': "l2", 'multi_class': "auto"},
    EIGHT= {'solver': "newton-cg", 'penalty': "l2", 'multi_class': "auto"},


   
          
class SVCConfi(Enum):
    ONE= {'kernel': "linear", 'degree': 3, 'gamma': "auto"},
    TWO= {'kernel': "poly", 'degree': 3, 'gamma': "scale"},
    THREE= {'kernel': "rbf", 'degree': 3, 'gamma': "scale"},
    FOUR=  {'kernel': "sigmoid", 'degree': 3, 'gamma': "scale"},
    FIVE=  {'kernel': "precomputed", 'degree': 3, 'gamma': "auto"},
