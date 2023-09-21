from enum import Enum
from itertools import product


# __all__= ('log_regression_config', 'svc_onfig', 'dtc_onfig',
#            'nbc_config', 'knn_config', 'mpl_config') 



class __LogRegressionConfigEnum(Enum):
    ONE= {'solver': "sag", 'penalty': "l2", 'multi_class': "auto", 'l1_ratio':0},
    TWO= {'solver': "saga", 'penalty': "l1", 'multi_class': "auto"},
    THREE= {'solver': "saga", 'penalty': "l2", 'multi_class': "auto"},
    FOUR=  {'solver': "saga", 'penalty': "elasticnet", 'multi_class': "auto"},
    FIVE=  {'solver': "liblinear", 'penalty': "l1", 'multi_class': "auto"},
    SIX=   {'solver': "liblinear", 'penalty': "l2", 'multi_class': "auto"},
    SEVEN= {'solver': "lbfgs", 'penalty': "l2", 'multi_class': "auto"},
    EIGHT= {'solver': "newton-cg", 'penalty': "l2", 'multi_class': "auto"},

          
class __SVCConfigEnum(Enum):
    ONE= {'kernel': "linear", 'degree': 3, 'gamma': "auto", 'probability': True},
    TWO= {'kernel': "poly", 'degree': 3, 'gamma': "scale", 'probability': False},
    THREE= {'kernel': "rbf", 'degree': 3, 'gamma': "scale", 'probability': False},
    FOUR=  {'kernel': "sigmoid", 'degree': 3, 'gamma': "scale", 'probability': False},
    FIVE=  {'kernel': "precomputed", 'degree': 3, 'gamma': "auto", 'probability': False}


class __DTConfigEnum(Enum):
    ONE= {'criterion' : "gini", 'max_depth': 2, },
    TWO= {'criterion' : "entropy", 'max_depth': 2,},
    THREE= {'criterion' : "gini", 'max_depth': 3, },
    FOUR= {'criterion' : "entropy", 'max_depth': 3,},


class __NBConfigEnum(Enum):
    ONE= {'priors': None, 'var_smoothing': 1e-9},


class KNNConfigEnum(Enum):
    ONE= {'n_neighbors':3, 'weights': 'uniform', 'algorithm': "auto"},
    TWO= {'n_neighbors': 4, 'weights': 'uniform', 'algorithm': "auto"},
    THREE= {'n_neighbors': 5, 'weights': 'uniform', 'algorithm': "auto"},
    FOUR= {'n_neighbors': 3, 'weights': 'distance', 'algorithm': "auto"},
    FIVE= {'n_neighbors': 4, 'weights': 'distance', 'algorithm': "auto"},
    SIX= {'n_neighbors': 5, 'weights': 'distance', 'algorithm': "auto"},



class __MLPCConfigEnum(Enum):
    ONE= {'hidden_layer_sizes':(8,), 'activation': 'identity' , 'max_iter': 500, 'alpha': 1e-4, 'solver': 'lbfgs', 'random_state': 1,  'learning_rate_init': 0.1},
    TWO= {'hidden_layer_sizes':(8,), 'activation': 'logistic' , 'max_iter': 500, 'alpha': 1e-4, 'solver': 'lbfgs', 'random_state': 1,  'learning_rate_init': 0.1},
    THREE= {'hidden_layer_sizes':(8,), 'activation': 'tanh' , 'max_iter': 500, 'alpha': 1e-4, 'solver': 'lbfgs', 'random_state': 1,  'learning_rate_init': 0.1},
    FOUR= {'hidden_layer_sizes':(8,), 'activation': 'relu', 'max_iter': 500, 'alpha': 1e-4, 'solver': 'lbfgs', 'random_state': 1,  'learning_rate_init': 0.1},
    FIVE= {'hidden_layer_sizes':(8,), 'activation': 'identity' , 'max_iter': 500, 'alpha': 1e-4, 'solver': 'adam', 'random_state': 1,  'learning_rate_init': 0.1},
    SIX= {'hidden_layer_sizes':(8,), 'activation': 'logistic' , 'max_iter': 500, 'alpha': 1e-4, 'solver': 'adam', 'random_state': 1,  'learning_rate_init': 0.1},
    SEVEN= {'hidden_layer_sizes':(8,), 'activation': 'tanh' , 'max_iter': 500, 'alpha': 1e-4, 'solver': 'adam', 'random_state': 1,  'learning_rate_init': 0.1},
    EGHIT= {'hidden_layer_sizes':(8,), 'activation': 'relu', 'max_iter': 500, 'alpha': 1e-4, 'solver': 'adam', 'random_state': 1,  'learning_rate_init': 0.1},
    NINE= {'hidden_layer_sizes':(8,), 'activation': 'identity' , 'max_iter': 500, 'alpha': 1e-4, 'solver': 'sgd', 'random_state': 1, 'learning_rate' : 'constant', 'learning_rate_init': 0.1},
    TEN= {'hidden_layer_sizes':(8,), 'activation': 'logistic' , 'max_iter': 500, 'alpha': 1e-4, 'solver': 'sgd', 'random_state': 1,'learning_rate' : 'invscaling', 'learning_rate_init': 0.1},
    ELEVEN= {'hidden_layer_sizes':(8,), 'activation': 'tanh' , 'max_iter': 500, 'alpha': 1e-4, 'solver': 'sgd', 'random_state': 1, 'learning_rate' :  'adaptive', 'learning_rate_init': 0.1},
    TWELVE= {'hidden_layer_sizes':(8,), 'activation': 'relu', 'max_iter': 500, 'alpha': 1e-4, 'solver': 'sgd', 'random_state': 1, 'learning_rate' : 'constant', 'learning_rate_init': 0.1},
    THIRTEEN= {'hidden_layer_sizes':(8,), 'activation': 'identity' , 'max_iter': 500, 'alpha': 1e-4, 'solver': 'sgd', 'random_state': 1,  'learning_rate' : 'invscaling','learning_rate_init': 0.1},
    FOURTEEN= {'hidden_layer_sizes':(8,), 'activation': 'logistic' , 'max_iter': 500, 'alpha': 1e-4, 'solver': 'sgd', 'random_state': 1, 'learning_rate' :  'adaptive', 'learning_rate_init': 0.1},
    FIFTEEN= {'hidden_layer_sizes':(8,), 'activation': 'tanh' , 'max_iter': 500, 'alpha': 1e-4, 'solver': 'sgd', 'random_state': 1, 'learning_rate' : 'constant', 'learning_rate_init': 0.1},
    SIXTEEN= {'hidden_layer_sizes':(8,), 'activation': 'relu', 'max_iter': 500, 'alpha': 1e-4, 'solver': 'sgd', 'random_state': 1,  'learning_rate' : 'invscaling', 'learning_rate_init': 0.1},
    SEVENTEEN= {'hidden_layer_sizes':(8,), 'activation': 'identity' , 'max_iter': 500, 'alpha': 1e-4, 'solver': 'sgd', 'random_state': 1, 'learning_rate' :  'adaptive', 'learning_rate_init': 0.1},
    EGHITEEN= {'hidden_layer_sizes':(8,), 'activation': 'logistic' , 'max_iter': 500, 'alpha': 1e-4, 'solver': 'sgd', 'random_state': 1, 'learning_rate' : 'constant', 'learning_rate_init': 0.1},
    NINETEEN= {'hidden_layer_sizes':(8,), 'activation': 'tanh' , 'max_iter': 500, 'alpha': 1e-4, 'solver': 'sgd', 'random_state': 1,  'learning_rate' : 'invscaling', 'learning_rate_init': 0.1},
    TWENTY= {'hidden_layer_sizes':(8,), 'activation': 'relu', 'max_iter': 500, 'alpha': 1e-4, 'solver': 'sgd', 'random_state': 1, 'learning_rate' :  'adaptive', 'learning_rate_init': 0.1},




class ModelsConfig:
    ''' 
    Abstract method class for changing default hyperparameters of classifiers
    '''
    def __init__(self, EnumClass, params) -> None:
        self.__params= params
        self._custome_config= None
        self.config= EnumClass

    def __call__(self):
        if self._custome_config is None:
            return self.config
        else:
            return self.comput_config()
    
    def set_new(self, config_dict: dict= None):
        if config_dict is None:
            raise AttributeError('a dictionary should be pass as argument')
        self._custome_config= config_dict


    def comput_config(self):
        iter_parms=[]
        iter_scalar= []
        if self._custome_config is None:
            raise AttributeError('a dictionary should be pass as argument')
        # Sorting parameters have a set of options and thoes have a scalar value
        for key,value in self._custome_config.items():
            try:
                len(value)
                iter_parms.append((key, value))
            except TypeError:
                iter_scalar.append((key, value))
        
        iter_parms= dict(iter_parms)
        iter_scalar= dict(iter_scalar)
        combinations = list(product(*iter_parms.values()))
        config_lst= []
        if len(iter_parms) == 0:
            EnumConfigClass= Enum('EnumConfigClass', zip('C', iter_scalar))  # noqa: E501
            return EnumConfigClass

        for combination in combinations:
            config= {**iter_scalar, **(dict(zip(iter_parms.keys(), combination)))}
            config_lst.append(config)

        name_list= [f'Custom_conf_{i+1}' for i in range(len(config_lst))]
        EnumConfigClass= Enum('EnumConfigClass', zip(name_list, config_lst))
        return EnumConfigClass

    

    @property
    def parameters(self):
        return self.__params
        
    
    @parameters.setter
    def parameters(self):
        raise AttributeError('Parameters can not be changed.')
    
    @parameters.deleter
    def parameters(self):
        raise AttributeError('Parameters can not be deleted.')
    

      

def log_regression_config():
    parameters= '''
        {
        'penalty': 'l2', # {'l1', 'l2', 'elasticnet', None}, default='l2
        'dual': False, # {True, False}, default=False
        'tol': 0.0001, # float, default=1e-4
        'C': 1.0, #float, default=1.0
        'fit_intercept': True, # {True, False}, default=True
        'intercept_scaling': 1, # float, default=1
        'class_weight': None, dict or 'balanced', default=None
        'random_state': None, # int, RandomState instance, default=None
        'solver': 'lbfgs', # {'lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'}, default='lbfgs'
        'max_iter': 100, # int, default=100
        'multi_class': 'auto', # {'auto', 'ovr', 'multinomial'}, default='auto'
        'verbose': 0, # int, default=0
        'warm_start': False, # {True, False}, default=False
        'n_jobs': None, # int, default=None 
        'l1_ratio': None #  float, default=None
        }
        '''  
    obj= ModelsConfig(__LogRegressionConfigEnum, parameters)
    return obj

def svc_onfig():
    parameters= '''  
        {
        'C' : float, default=1.0
        'kernel' : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}, # or callable
        'degree' : int, default=3
        'gamma' : {'scale', 'auto'}, # or float, default='scale'
        'coef0' : float, # default=0.0
        'shrinking' : {True, False}, # default=True
        'probability' : {True, False}, # default=False
        'tol' : float, # default=1e-3
        'cache_size' : 200, # float, default=200
        'class_weight' : None , # dict or 'balanced', default=None
        'verbose' : {True, False}, # default=False
        'max_iter' : -1,  # int, default=-1
        'decision_function_shape' : {'ovo', 'ovr'}, # default='ovr'
        'break_ties' : {True, False},  # default=False
        'random_state' : 15,  # int, RandomState instance or None, default=None
        }
    '''
    obj= ModelsConfig(__SVCConfigEnum, parameters)
    return obj


def dtc_onfig():
    parameters= '''
    {
    'criterion':'gini', # {"gini", "entropy", "log_loss"}, default="gini"
    'splitter': 'best',  # {"best", "random"}, default="best"
    'max_depth': None,   # int, default=None
    'min_samples_split': 2, # int or float, default=2
    'min_samples_leaf': 1,  # int or float, default=1
    'min_weight_fraction_leaf': 0.0, # int or float, default=1
    'max_features': None, # float, default=0.0
    'random_state': None, # int, RandomState instance or None, default=None
    'max_leaf_nodes': None, # int, default=None
    'min_impurity_decrease': 0.0, #  float, default=0.0
    'class_weight': None, # dict, list of dict or "balanced", default=None
    'ccp_alpha': 0.0 # non-negative float, default=0.0
    }
    '''
    obj= ModelsConfig(__DTConfigEnum, parameters)
    return obj


def nbc_config():
    parameters= '''
    {
     'priors': None, # array-like of shape (n_classes,), default=None
     'var_smoothing': 1e-09 # float, default=1e-9
    }
    '''
    obj= ModelsConfig(__NBConfigEnum, parameters)
    return obj


def knn_config():
    parameters= '''
    {
    'n_neighbors': 5, # int, default=5
    'weights': 'uniform', # {'uniform', 'distance'}, callable or None, default='uniform'
    'algorithm':'auto', # {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
    'leaf_size': 30, # int, default=30
    'p': 2, # int, default=30
    'metric': 'minkowski', # str or callable, default='minkowski'
    'metric_params': None, # dict, default=None
    'n_jobs': None # int, default=None
    }
    '''
    obj= ModelsConfig(KNNConfigEnum, parameters )
    return obj


def mpl_config():
    parameters= '''
    {
    'hidden_layer_sizes': (100,), # array-like of shape(n_layers - 2,), default=(100,)
    'activation': 'relu', # {'identity', 'logistic', 'tanh', 'relu'}, default='relu' or you can pass all acitivations
    'solver': 'adam', #  {'lbfgs', 'sgd', 'adam'}, default='adam' or you can pass all solvers
    'alpha': 0.0001, # float, default=0.0001
    'batch_size': 'auto', # int, default='auto'
    'learning_rate': 'constant', # {'constant', 'invscaling', 'adaptive'}, default='constant' or you can pass all rates
    'learning_rate_init': 0.001, # float, default=0.001
    'power_t': 0.5, # float, default=0.5
    'max_iter': 200, # int, default=200
    'shuffle': True, # {True, False}, default=True
    'random_state': None, # int, RandomState instance, default=None
    'tol': 0.0001, # float, default=1e-4
    'verbose': False, # {True, False}, default=False
    'warm_start': False, # {True, False}, default=False
    'momentum': 0.9, # float, default=0.9
    'nesterovs_momentum': True, # float, default=0.9
    'early_stopping': False, # {True, False}, default=False
    'validation_fraction': 0.1, #  float, default=0.1
    'beta_1': 0.9, # float, default=0.9
    'beta_2': 0.999, # float, default=0.999
    'epsilon': 1e-08, # float, default=1e-8
    'n_iter_no_change': 10, # int, default=10
    'max_fun': 15000 #  int, default=15000
    }
    '''  # noqa: E501
    obj= ModelsConfig(__MLPCConfigEnum, parameters)
    return obj

