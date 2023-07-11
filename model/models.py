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
    
    '''
    def __init__(self, model_dir) -> None:
        super().__init__()
        self.config_list= SVCConfi.__members__
        self.config= iter(SVCConfi)
        self.model_dir= model_dir

    def _Model__make_model(self):
        pass



