from sklearn.linear_model import LogisticRegression
from model.config import LogRegressionConfig, SVCConfi
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

from model.config import *
from model.dtsets import Iris
import pickle
import os
from abc import ABC, abstractmethod




class Model(ABC):

    def __init__(self) -> None:
        self.dataset= Iris().dataset
     
    def __configure(self):
        config= next(self.config)
        return  config.name, config.value
    
    @abstractmethod 
    def __make_model(self):
        pass
    
    def train(self):
        all_configs= len(self.config_list)
        model_counter= 0
    
        for _ in self.config_list:
            try:
                config_name, config= self.__configure()
                print(type(config))
                trained_model= self.__make_model(config[0], self.dataset[0], self.dataset[1])
                model_counter +=1
                  
            except Exception as ex:
                print('error:',ex)
                #TODO should be logged or writen in a file 
                
            else:
                file= os.path.join(self.model_dir, f'{config_name}.pickle')
                with open(file, 'bw') as file:
                        pickle.dump(trained_model, file)

        return all_configs, model_counter



class LogRegression(Model):
    '''
    Logestic Regression has below parameters and them default values:
    penalty='l2', 
    dual=False,
    tol=0.0001,
    C=1.0,
    fit_intercept=True,
    intercept_scaling=1,
    class_weight=None,
    random_state=None,
    solver='lbfgs',
    max_iter=100,
    multi_class='auto',
    verbose=0,
    warm_start=False,
    n_jobs=None,
    l1_ratio=None
    Assigned Parameters:
    In this class only below parameters are assigned and others remain default
    penalty: {'l1', 'l2', 'elasticnet'}
    solver: {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'}
    multi_class: {'auto', 'ovr', 'multinomial'}
    '''

    def __init__(self, model_dir) -> None:
        super().__init__()
        self.config_list= LogRegressionConfig.__members__
        self.config= iter(LogRegressionConfig)
        self.model_dir= model_dir
        self.dataset= Iris().dataset
     
    
    def _Model__make_model(self,conf, X_train, y_train):
        self.mdl= LogisticRegression(**conf)
        return self.mdl.fit(X_train, y_train)
    
    
class SVCModel(Model):
    '''
    This class implement different kernels of SVC.
    The Kernel is: {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}
    Below text is from sklean.svm.SVC help
    
    C-Support Vector Classification.

    Parameters
    ----------
    C : float, default=1.0
    kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'} or callable,  \
    degree : int, default=3
    gamma : {'scale', 'auto'} or float, default='scale'
    coef0 : float, default=0.0
    shrinking : bool, default=True
    probability : bool, default=False
    tol : float, default=1e-3
    cache_size : float, default=200
    class_weight : dict or 'balanced', default=None
    verbose : bool, default=False
    max_iter : int, default=-1
    decision_function_shape : {'ovo', 'ovr'}, default='ovr'
    break_ties : bool, default=False
    random_state : int, RandomState instance or None, default=None
    Attributes
    ----------
    class_weight_ : ndarray of shape (n_classes,)
    classes_ : ndarray of shape (n_classes,)
    coef_ : ndarray of shape (n_classes * (n_classes - 1) / 2, n_features)
    dual_coef_ : ndarray of shape (n_classes -1, n_SV)
    fit_status_ : int
    intercept_ : ndarray of shape (n_classes * (n_classes - 1) / 2,)
    n_features_in_ : int
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
    support_ : ndarray of shape (n_SV)
    support_vectors_ : ndarray of shape (n_SV, n_features)
    n_support_ : ndarray of shape (n_classes,), dtype=int32
    probA_ : ndarray of shape (n_classes * (n_classes - 1) / 2)
    probB_ : ndarray of shape (n_classes * (n_classes - 1) / 2)
    shape_fit_ : tuple of int of shape (n_dimensions_of_X,)
        Array dimensions of training vector ``X``.
    '''

    def __init__(self, model_dir) -> None:
        super().__init__()
        self.config_list= SVCConfig.__members__
        self.config= iter(SVCConfig)
        self.model_dir= model_dir

    def _Model__make_model(self,conf, X_train, y_train):
        self.mdl= SVC(**conf)
        return self.mdl.fit(X_train, y_train)


class DTModel(Model):
    '''
"""A decision tree classifier.

    Parameters
    ----------
    criterion : {"gini", "entropy"}, default="gini"
    splitter : {"best", "random"}, default="best"
    max_depth : int, default=None
    min_samples_split : int or float, default=2
    min_samples_leaf : int or float, default=1
    min_weight_fraction_leaf : float, default=0.0
    max_features : int, float or {"auto", "sqrt", "log2"}, default=None
    random_state : int, RandomState instance or None, default=None
    max_leaf_nodes : int, default=None
    min_impurity_decrease : float, default=0.0
    class_weight : dict, list of dict or "balanced", default=None
    ccp_alpha : non-negative float, default=0.0
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,) or list of ndarray
    feature_importances_ : ndarray of shape (n_features,)
    max_features_ : int
    n_classes_ : int or list of int
    n_features_ : int
    n_features_in_ : int
     n_outputs_ : int
    tree_ : Tree instance
    '''

    def __init__(self, model_dir) -> None:
        super().__init__()
        self.config_list= DTConfig.__members__
        self.config= iter(DTConfig)
        self.model_dir= model_dir

    def _Model__make_model(self,conf, X_train, y_train):
        self.mdl= DecisionTreeClassifier(**conf)
        return self.mdl.fit(X_train, y_train)
    

class NBModel(Model):
    '''
    Gaussian Naive Bayes (GaussianNB).
    Parameters
    ----------
    priors : array-like of shape (n_classes,)
    var_smoothing : float, default=1e-9
    Attributes
    ----------
    class_count_ : ndarray of shape (n_classes,)
        number of training samples observed in each class.
    class_prior_ : ndarray of shape (n_classes,)
        probability of each class.
    classes_ : ndarray of shape (n_classes,)
        class labels known to the classifier.
    epsilon_ : float
        absolute additive value to variances.
    n_features_in_ : int
        Number of features seen during :term:`fit`.
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
        .. versionadded:: 1.0
    sigma_ : ndarray of shape (n_classes, n_features)
        Variance of each feature per class.
        .. deprecated:: 1.0
           `sigma_` is deprecated in 1.0 and will be removed in 1.2.
           Use `var_` instead.
    var_ : ndarray of shape (n_classes, n_features)
        Variance of each feature per class.
        .. versionadded:: 1.0
    theta_ : ndarray of shape (n_classes, n_features)
        mean of each feature per class.
    '''

    def __init__(self, model_dir) -> None:
        super().__init__()
        self.config_list= NBConfig.__members__
        self.config= iter(NBConfig)
        self.model_dir= model_dir

    def _Model__make_model(self,conf, X_train, y_train):
        self.mdl= GaussianNB(**conf)
        return self.mdl.fit(X_train, y_train)
    

class MLPClassifierModel(Model):

    '''
    Multi-layer Perceptron classifier.
    This model optimizes the log-loss function using LBFGS or stochastic
    gradient descent.
    Parameters
    ----------
    hidden_layer_sizes : tuple, length = n_layers - 2, default=(100,)
    activation : {'identity', 'logistic', 'tanh', 'relu'}, default='relu'
    solver : {'lbfgs', 'sgd', 'adam'}, default='adam'
    alpha : float, default=0.0001
    batch_size : int, default='auto'
    learning_rate : {'constant', 'invscaling', 'adaptive'}, default='constant'     
    learning_rate_init : float, default=0.001      
    power_t : float, default=0.5
    max_iter : int, default=200
    shuffle : bool, default=True
    random_state : int, RandomState instance, default=None
    tol : float, default=1e-4
    verbose : bool, default=False
    warm_start : bool, default=False
    momentum : float, default=0.9
    nesterovs_momentum : bool, default=True
    early_stopping : bool, default=False
    validation_fraction : float, default=0.1
    beta_1 : float, default=0.9
    beta_2 : float, default=0.999
    epsilon : float, default=1e-8
    n_iter_no_change : int, default=10
    max_fun : int, default=15000
    Attributes
    ----------
    classes_ : ndarray or list of ndarray of shape (n_classes,)
        Class labels for each output.
    loss_ : float
        The current loss computed with the loss function.
    best_loss_ : float
        The minimum loss reached by the solver throughout fitting.
    loss_curve_ : list of shape (`n_iter_`,)
        The ith element in the list represents the loss at the ith iteration.
    t_ : int
        The number of training samples seen by the solver during fitting.
    coefs_ : list of shape (n_layers - 1,)
        The ith element in the list represents the weight matrix corresponding
        to layer i.
    intercepts_ : list of shape (n_layers - 1,)
        The ith element in the list represents the bias vector corresponding to
        layer i + 1.
    n_features_in_ : int
        Number of features seen during :term:`fit`.
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
    n_iter_ : int
        The number of iterations the solver has run.
    n_layers_ : int
        Number of layers.
    n_outputs_ : int
        Number of outputs.
    out_activation_ : str
        Name of the output activation function.
    '''
    def __init__(self, model_dir) -> None:
        super().__init__()
        self.config_list= MLPCConfig.__members__
        self.config= iter(MLPCConfig)
        self.model_dir= model_dir

    def _Model__make_model(self,conf, X_train, y_train):
        self.mdl= MLPClassifier(**conf)
        return self.mdl.fit(X_train, y_train)
    
class KNNModle(Model):
    '''
    Classifier implementing the k-nearest neighbors vote.
    Read more in the :ref:`User Guide <classification>`.
    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use by default for :meth:`kneighbors` queries.
    weights : {'uniform', 'distance'} or callable, default='uniform'
        Weight function used in prediction.  Possible values:
        - 'uniform' : uniform weights.  All points in each neighborhood
          are weighted equally.
        - 'distance' : weight points by the inverse of their distance.
          in this case, closer neighbors of a query point will have a
          greater influence than neighbors which are further away.
        - [callable] : a user-defined function which accepts an
          array of distances, and returns an array of the same shape
          containing the weights.
    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        Algorithm used to compute the nearest neighbors:
        - 'ball_tree' will use :class:`BallTree`
        - 'kd_tree' will use :class:`KDTree`
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method.
        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.
   leaf_size : int, default=30
        Leaf size passed to BallTree or KDTree.  This can affect the
        speed of the construction and query, as well as the memory
        required to store the tree.  The optimal value depends on the
        nature of the problem.
    p : int, default=2
        Power parameter for the Minkowski metric. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
    metric : str or callable, default='minkowski'
        The distance metric to use for the tree.  The default metric is
        minkowski, and with p=2 is equivalent to the standard Euclidean
        metric. For a list of available metrics, see the documentation of
        :class:`~sklearn.metrics.DistanceMetric`.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square during fit. X may be a :term:`sparse graph`,
        in which case only "nonzero" elements may be considered neighbors.
    metric_params : dict, default=None
        Additional keyword arguments for the metric function.
    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
        Doesn't affect :meth:`fit` method.
    Attributes
    ----------
    classes_ : array of shape (n_classes,)
        Class labels known to the classifier
    effective_metric_ : str or callble
        The distance metric used. It will be same as the `metric` parameter
        or a synonym of it, e.g. 'euclidean' if the `metric` parameter set to
        'minkowski' and `p` parameter set to 2.
    effective_metric_params_ : dict
        Additional keyword arguments for the metric function. For most metrics
        will be same with `metric_params` parameter, but may also contain the
        `p` parameter value if the `effective_metric_` attribute is set to
        'minkowski'.
    n_features_in_ : int
        Number of features seen during :term:`fit`.
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
    n_samples_fit_ : int
        Number of samples in the fitted data.
    outputs_2d_ : bool
        False when `y`'s shape is (n_samples, ) or (n_samples, 1) during fit
        otherwise True.
    '''

    def __init__(self, model_dir) -> None:
        super().__init__()
        self.config_list= KNNConfig.__members__
        self.config= iter(KNNConfig)
        self.model_dir= model_dir

    def _Model__make_model(self,conf, X_train, y_train):
        self.mdl= MLPClassifier(**conf)
        return self.mdl.fit(X_train, y_train)
    

