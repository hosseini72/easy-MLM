from enum import Enum


class LogRegressionConfig(Enum):
    ONE= {'solver': "sag", 'penalty': "l2", 'multi_class': "auto", 'l1_ratio':0},
    TWO= {'solver': "saga", 'penalty': "l1", 'multi_class': "auto"},
    THREE= {'solver': "saga", 'penalty': "l2", 'multi_class': "auto"},
    FOUR=  {'solver': "saga", 'penalty': "elasticnet", 'multi_class': "auto"},
    FIVE=  {'solver': "liblinear", 'penalty': "l1", 'multi_class': "auto"},
    SIX=   {'solver': "liblinear", 'penalty': "l2", 'multi_class': "auto"},
    SEVEN= {'solver': "lbfgs", 'penalty': "l2", 'multi_class': "auto"},
    EIGHT= {'solver': "newton-cg", 'penalty': "l1", 'multi_class': "auto"},
    # NiNE=  {'solver': "newton-cholesky", 'penalty': "l2", 'multi_class': "auto"},
    # TEN=   {'solver': "newton-cg", 'penalty': "l1", 'multi_class': "auto"},

   
          
class SVCConfi(Enum):
    ONE= {'solver': "sag", 'penalty': "l2", 'multi_class': "auto", 'l1_ratio':0},
    TWO= {'solver': "saga", 'penalty': "l1", 'multi_class': "auto"},
    THREE= {'solver': "saga", 'penalty': "l2", 'multi_class': "auto"},


