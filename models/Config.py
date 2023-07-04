# class test:
#     def p(self):
#         print('intest')

from enum import Enum


class Config_Log_Reg(Enum):
    ONE= {'solver': "sag", 'penalty': "l2", 'multi_class': "auto"},
    TWO= {'solver': "saga", 'penalty': "l1", 'multi_class': "auto"},
    THREE= {'solver': "saga", 'penalty': "l2", 'multi_class': "auto"},
    FOUR=  {'solver': "saga", 'penalty': "elasticnet", 'multi_class': "auto"},
    FIVE=  {'solver': "liblinear", 'penalty': "l1", 'multi_class': "auto"},
    SIX=   {'solver': "liblinear", 'penalty': "l2", 'multi_class': "auto"},
    SEVEN= {'solver': "lbfgs", 'penalty': "l2", 'multi_class': "auto"},
    EIGHT= {'solver': "newton-cg", 'penalty': "l1", 'multi_class': "auto"},
          
    
print(Config_Log_Reg._member_names_)
