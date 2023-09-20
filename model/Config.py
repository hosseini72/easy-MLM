from enum import Enum
from itertools import product

class LogRegressionConfigEnum(Enum):
    ONE= {'solver': "sag", 'penalty': "l2", 'multi_class': "auto", 'l1_ratio':0},
    TWO= {'solver': "saga", 'penalty': "l1", 'multi_class': "auto"},
    THREE= {'solver': "saga", 'penalty': "l2", 'multi_class': "auto"},
    FOUR=  {'solver': "saga", 'penalty': "elasticnet", 'multi_class': "auto"},
    FIVE=  {'solver': "liblinear", 'penalty': "l1", 'multi_class': "auto"},
    SIX=   {'solver': "liblinear", 'penalty': "l2", 'multi_class': "auto"},
    SEVEN= {'solver': "lbfgs", 'penalty': "l2", 'multi_class': "auto"},
    EIGHT= {'solver': "newton-cg", 'penalty': "l2", 'multi_class': "auto"},

          
class SVCConfigEnum(Enum):
    ONE= {'kernel': "linear", 'degree': 3, 'gamma': "auto", 'probability': True},
    TWO= {'kernel': "poly", 'degree': 3, 'gamma': "scale", 'probability': False},
    THREE= {'kernel': "rbf", 'degree': 3, 'gamma': "scale", 'probability': False},
    FOUR=  {'kernel': "sigmoid", 'degree': 3, 'gamma': "scale", 'probability': False},
    FIVE=  {'kernel': "precomputed", 'degree': 3, 'gamma': "auto", 'probability': False}


class DTConfigEnum(Enum):
    ONE= {'criterion' : "gini", 'max_depth': 2, },
    TWO= {'criterion' : "entropy", 'max_depth': 2,},
    THREE= {'criterion' : "gini", 'max_depth': 3, },
    FOUR= {'criterion' : "entropy", 'max_depth': 3,},


class NBConfigEnum(Enum):
    ONE= {'priors': None, 'var_smoothing': 1e-9},


class KNNConfigEnum(Enum):
    ONE= {'n_neighbors':3, 'weights': 'uniform', 'algorithm': "auto"},
    TWO= {'n_neighbors': 4, 'weights': 'uniform', 'algorithm': "auto"},
    THREE= {'n_neighbors': 5, 'weights': 'uniform', 'algorithm': "auto"},
    FOUR= {'n_neighbors': 3, 'weights': 'distance', 'algorithm': "auto"},
    FIVE= {'n_neighbors': 4, 'weights': 'distance', 'algorithm': "auto"},
    SIX= {'n_neighbors': 5, 'weights': 'distance', 'algorithm': "auto"},



class MLPCConfigEnum(Enum):
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
        self.params= params
        self._custome_config= None
        self.config= EnumClass

    def __call__(self, Config_dict: dict= None):
        self._custome_config= Config_dict
        if self._custome_config is None:
            return self.config
        else:
            return self.set_config()
    
    def set_config(self):
        iter_parms=[]
        iter_scalar= []
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
        parms =   '''  
        {
        C : float, default=1.0
        kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'} or callable,  
        degree : int, default=3
        gamma : {'scale', 'auto'} or float, default='scale'
        coef0 : float, default=0.0
        shrinking : {True, False}, default=True
        probability : {True, False}, default=False
        tol : float, default=1e-3
        cache_size : float, default=200
        class_weight : dict or 'balanced', default=None
        verbose : {True, False}, default=False
        max_iter : int, default=-1
        decision_function_shape : {'ovo', 'ovr'}, default='ovr'
        break_ties : {True, False}, default=False
        random_state : int, RandomState instance or None, default=None
        }
    '''
        return parms
        
    
    # @parameters.setter
    # def parameters(self):
    #     raise AttributeError('Parameters can not be changed.')
    
    # @parameters.deleter
    # def parameters(self):
    #     raise AttributeError('Parameters can not be deleted.')
    

    @property
    def param_sample(self):
        parms =     {
            'C' : 1.0,
            'kernel' : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'},  
            'degree' : 3,
            'gamma' : {'scale', 'auto'},
            'coef0' : 0.0,
            'shrinking' : {True, False},
            'probability' : {True, False},
            'tol' : 1e-3,
            'cache_size' : 200,
            'class_weight' : None,
            'verbose' : {True, False},
            'max_iter' : -1,
            'decision_function_shape' : {'ovo', 'ovr'}, 
            'break_ties' : {True, False}, 
            'random_state' : 15,
           }
    
        return parms
    
    
    @param_sample.setter
    def param_sample(self):
        raise AttributeError('Parameters sample can not be changed. '
                         'This is a sample to make neccessary changes by user. ')
    
    @param_sample.deleter
    def param_sample(self):
        raise AttributeError('Parameters sample can not be deleted. '
                         'This is a sample to make neccessary changes by user. ')
        


def log_regression_config():
    parameters= {}
    obj= ModelsConfig(LogRegressionConfigEnum, parameters)
    obj.


def svc_onfig():
    parameters= {}
    obj= ModelsConfig(SVCConfigEnum, parameters)


def dtc_onfig():
    parameters= {}
    obj= ModelsConfig(DTConfigEnum, parameters)


def nbc_config():
    parameters= {}
    obj= ModelsConfig(NBConfigEnum, parameters)


def knn_config():
    parameters= {}
    obj= ModelsConfig(KNNConfigEnum, parameters )


def mpl_config():
    parameters= {}
    obj= ModelsConfig(MLPCConfigEnum, parameters)

