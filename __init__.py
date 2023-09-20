
from model.config import *


x= nbc_config()
# print(x.parameters)
d=  {
     'priors': None, # array-like of shape (n_classes,), default=None
     'var_smoothing': 1e-09 # float, default=1e-9
    }
x.set_config(d)
y= x()
# print(x._custome_config)
print(y)