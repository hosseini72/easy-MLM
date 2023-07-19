from model.models import *
from model.dtsets import Iris
import os
from random import choice

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


'''
This package can be helpful for model selection
'''

class Train:
    def __init__(self):
        self.__base_dir= os.getcwdb().decode()
        self.__make_main_dir()
        self.__models= [LogRegression, SVCModel, DTModel, NBModel, MLPClassifierModel, KNNModle]
        self.dataset= Iris().dataset

    def __make_main_dir(self):
        self.__models_path= os.path.join(self.__base_dir, 'trained_models')
        try:
            os.mkdir(self.__models_path)
        except FileExistsError:
            counter= 1
            while True:
                counter+= 1
                try:
                    self.__models_path= os.path.join(self.__base_dir, f'trained_models_{counter}')
                    os.mkdir(self.__models_path)
                except FileExistsError:
                    pass
                else:
                    break
        except Exception as err:
            print(err)
            
    def __make_model_dir(self, model): 
        model_dir= os.path.join(self.__models_path, model.__name__)
        os.mkdir(model_dir)
        return model_dir


    def __run_single_model(self, model):
        model_dir= self.__make_model_dir(model=model)
        mdl_obj= model(model_dir, self.dataset)
        all_conf, conf_cuntr = mdl_obj.train()
        return all_conf, conf_cuntr


    def fit(self):
        ''' fit() trains all models and their configs once and save a pickled file in each model directory.'''
        for model in self.__models:
            self.__run_single_model(model=model)
    
    def fit_one(self, model= None):
        if not model:
            model= choice(self.__models)
        self.__run_single_model(model=model)

    @property
    def models(self):
        models= [mdl.__name__ for mdl in self.__models]
        return models
    
    @models.setter
    def models(self, item):
        raise Exception('These models are permanents and can not be changed. To run espesific model fit_one(model_name) can be used.')

    @models.deleter
    def models(self):
        raise Exception('These models are permanents and can not be deleted. To run espesific model fit_one(model_name) can be used.')

    @property
    def path(self):
        return self.__models_path
    
    @path.setter
    def path(self, new_dir):
        if os.path.isdir(new_dir):
            self.__base_dir= new_dir
        else:
            raise Exception('These models are permanents and can not be changed. To run espesific model fit_one(model_name) can be used.')

    @path.deleter
    def path(self):
        raise Exception('These models are permanents and can not be deleted. To run espesific model fit_one(model_name) can be used.')



'''
from mlxtend.plotting import plot_decision_regions
plot_decision_regions(xtr, ytr, clf= ir, legend =2 )

'''

class Evaluate:
    def __init__(self, dir) -> None:
        self.__path= dir
        self.__models_dir_list= iter(os.listdir(self.__path))

    @property
    def __model_directory(self):
        return next(self.__models_dir_list)

    def __trained_models(self, model):
        self.__trained_model_dir= os.path.join(self.__path, model)
        self.__trained_configs_list=  os.listdir(self.__trained_model_dir)
        print(self.__trained_model_dir)
        print(self.__trained_configs_list)
        
    def test(self):
        pass


    def score(self):
        model_configs= self.__trained_models(mdl)

        for _ in range(5):
            mdl= self.__model_directory
            self.__trained_models(mdl)

    def score_train(self):
        pass

    def confusion(self):
        pass

tr= Train()
tr.fit()
# print(tr.path)
# path= 'N:\MLM\trained_models'
# te= Evaluate(r'N:\MLM\trained_models')
# te.score()

'''
Accuracy
Confusion matrix
F1 score
False positive rate
False negative rate
Log loss
Negative predictive value
Precession
Recall
ROC Curve
Specificity
mean abolute error
quantile loss 
KL Divergence
Hinge loss 
'''