from sklearn.linear_model import LogisticRegression
from model.config import LogRegressionConfig, SVCConfi
from sklearn.svm import SVC
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
        self.mdl= LogisticRegression(penalty=conf['penalty'], solver=conf['solver'], multi_class=conf['multi_class'])
        return self.mdl.fit(X_train, y_train)
    
    
class SVCModel(Model):
    '''
    This class implement different kernels of SVC.
    The Kernel is: {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}
    Below text is from sklean.svm.SVC help
    
    C-Support Vector Classification.

    The implementation is based on libsvm. The fit time scales at least
    quadratically with the number of samples and may be impractical
    beyond tens of thousands of samples. For large datasets
    consider using :class:`~sklearn.svm.LinearSVC` or
    :class:`~sklearn.linear_model.SGDClassifier` instead, possibly after a
    :class:`~sklearn.kernel_approximation.Nystroem` transformer.

    The multiclass support is handled according to a one-vs-one scheme.

    For details on the precise mathematical formulation of the provided
    kernel functions and how `gamma`, `coef0` and `degree` affect each
    other, see the corresponding section in the narrative documentation:
    :ref:`svm_kernels`.

    Read more in the :ref:`User Guide <svm_classification>`.

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
        Hard limit on iterations within solver, or -1 for no limit.

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
        self.config_list= SVCConfi.__members__
        self.config= iter(SVCConfi)
        self.model_dir= model_dir

    def _Model__make_model(self,conf, X_train, y_train):
        self.mdl= SVC(kernel= conf['kernel'], degree= conf['degree'], gamma= conf['gamma'])
        return self.mdl.fit(X_train, y_train)

