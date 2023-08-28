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



   
          
class SVCConfig(Enum):
    ONE= {'kernel': "linear", 'degree': 3, 'gamma': "auto", 'probability': True},
    # TWO= {'kernel': "poly", 'degree': 3, 'gamma': "scale", 'probability': False},
    # THREE= {'kernel': "rbf", 'degree': 3, 'gamma': "scale", 'probability': False},
    # FOUR=  {'kernel': "sigmoid", 'degree': 3, 'gamma': "scale", 'probability': False},
    # FIVE=  {'kernel': "precomputed", 'degree': 3, 'gamma': "auto", 'probability': False}


class DTConfig(Enum):
    ONE= {'criterion' : "gini", 'max_depth': 2, },
    TWO= {'criterion' : "entropy", 'max_depth': 2,},
    THREE= {'criterion' : "gini", 'max_depth': 3, },
    FOUR= {'criterion' : "entropy", 'max_depth': 3,},


class NBConfig(Enum):
    ONE= {'priors': None, 'var_smoothing': 1e-9},


class KNNConfig(Enum):
    ONE= {'n_neighbors':3, 'weights': 'uniform', 'algorithm': "auto"},
    TWO= {'n_neighbors': 4, 'weights': 'uniform', 'algorithm': "auto"},
    THREE= {'n_neighbors': 5, 'weights': 'uniform', 'algorithm': "auto"},
    FOUR= {'n_neighbors': 3, 'weights': 'distance', 'algorithm': "auto"},
    FIVE= {'n_neighbors': 4, 'weights': 'distance', 'algorithm': "auto"},
    SIX= {'n_neighbors': 5, 'weights': 'distance', 'algorithm': "auto"},



class MLPCConfig(Enum):
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
