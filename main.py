from model.models import *
from model.dtsets import DataSet
import os
from random import choice
import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from sklearn.metrics import accuracy_score,precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import log_loss
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import hinge_loss

'''
This package can be helpful for model selection
'''

class Train:
    def __init__(self):
        self.__base_dir= os.getcwdb().decode()
        self.__make_main_dir()
        self.__models= [LogRegression, SVCModel, DTModel, NBModel, MLPClassifierModel, KNNModle]
        self.dataset= DataSet().train_dataset

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





class Evaluate:
    def __init__(self, dir) -> None:
        self.__path= dir
        self.__models_dir_list= iter(os.listdir(self.__path))
        self.data, self.target= DataSet().test_dataset

    @property
    def __model_directory(self):
        return next(self.__models_dir_list)

    def __trained_models(self, model):
        self.__trained_model_dir= os.path.join(self.__path, model)
        self.__trained_configs_list=  os.listdir(self.__trained_model_dir)
        return self.__trained_configs_list
        
    def test(self):
        result_list= []
        while True:
            try:
                mdl= self.__model_directory 
                mdl_dir= os.path.join(self.__path, mdl)
                if not os.path.isdir(mdl_dir):
                    continue
                mdl_configs_list= self.__trained_models(mdl)
                # if should check that it is a directory
                for mdl_config in mdl_configs_list:
                    with open(f'{mdl_dir}\{str(mdl_config)}', 'br') as file:
                        trained_mdl= pickle.load(file= file)
                    result= self.__evaluate(model_name=mdl, config_name=mdl_config, model= trained_mdl)
                    result_list.append(result)
                # else:
                #     print('mdl changed from ', mdl)
            except StopIteration:
                break
        evaluation_df= pd.DataFrame(result_list)
        print(self.__path)
        evaluation_df.to_csv(f'{self.__path}\MLM.csv')
        return evaluation_df

    def __evaluate(self, model_name, config_name, model):
        true_labels= self.target 
        predicted_labels= model.predict(self.data)
        predicted_probabilities= model.predict_proba(self.data)
        predicted_scores= model.score(self.data, self.target)
        result= {}
        try: # in the binary case
            conf_matrix= confusion_matrix(true_labels, predicted_labels)
            tn, fp, fn, tp= conf_matrix.ravel()
            result= {
                'model': model_name,
                'config': config_name.split('.')[0],
                # 'train score': model.score(X, y),
                'predict score': predicted_scores,
                'Accuracy': (tp + tn)/(tp + tn + fp + fn),
                'Precesion': tp/(tp + fp)  if (tp + fp) != 0 else 0.0,
                'Recall': tp/(tp + fn) if (tp + fn) != 0 else 0.0,
                'Specificity': tn/(tn+fp) if (tn + fn) != 0 else 0.0,
                'F1 score': 2*tp/(2*tp+fp+fn),
                'Negative predictive value': tn / (tn + fn) if (tn + fn) != 0 else 0.0,
                'False positive rate': fp / (fp + tn) if (fp + tn) != 0 else 0.0,
                'False negative rate': fn/(fn + tp) if (fn + tp) != 0 else 0.0,
                'mean abolute error': mean_absolute_error(true_labels, predicted_labels),
                'Log loss': log_loss(true_labels, predicted_probabilities),
                # 'Hinge loss': hinge_loss(true_labels, predicted_scores)
                # quantile loss 
                # KL Divergence
            }

        except: # in non binary case
            conf_matrix= multilabel_confusion_matrix
            result= {'model': model_name,
                    'config': config_name,
                    # 'train score': model.score(X, y),
                    'predict score': predicted_scores,
                    'Accuracy': (tp+tn)/(tp+tn+fp+fn),
                    'Precesion': precision_score(true_labels, predicted_labels),
                    'Recall': tp/(tp+fn),
                    'Specificity': tn/(tn+fp),
                    'F1 score': 2*tp/(2*tp+fp+fn),
                    'Negative predictive value': tn / (tn + fn),
                    'False positive rate': fp/(fp + tn),
                    'False negative rate': fn/(fn + tp),
                    'mean abolute error': mean_absolute_error(true_labels, predicted_labels),
                    'Log loss': log_loss(true_labels, predicted_probabilities),
                    'Hinge loss': hinge_loss(true_labels, predicted_scores)}
        return result
    

    def score(self):
        # model_configs= self.__trained_models(mdl)

        for _ in range(5):
            mdl= self.__model_directory
            self.__trained_models(mdl)

    def score_train(self):
        pass


    def roc_curve(self):
        
        # from sklearn.metrics import roc_curve
        # fpr, tpr, thresholds = roc_curve(true_labels, predicted_probabilities)
        pass


# tr= Train()
# tr.fit()
# print(tr.path)
# path= 'N:\MLM\trained_models'
te= Evaluate(r'N:\MLM\trained_models')
te.test()
# print(te.data.shape, te.target.shape)



'''
from mlxtend.plotting import plot_decision_regions
plot_decision_regions(xtr, ytr, clf= ir, legend =2 )


# predict 
P = model.predict_proba(testX)
clf.predict(xtest)
clf.score(X, y)
'''