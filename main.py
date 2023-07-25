from model.models import *
from model.dtsets import Iris
import os
from random import choice
import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
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
        return self.__trained_configs_list
        
    def test(self):
        result_list= []
        while True:
            try:
                mdl= self.__model_directory 
                mdl_dir= os.path.join(self.__path, mdl)
                mdl_configs_list= self.__trained_models(mdl)
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
        evaluation_df.to_csv(f'{self.__path}\MLM.csv')
        return evaluation_df


    



    def __evaluate(self, model_name, config_name, model):
        true_labels= None #TODO true labels for test data ytest
        predicted_labels= model.predict(xtest)
        predicted_probabilities= model.predict_proba(xtest)
        predicted_scores= model.score(X, y)

        conf_matrix= confusion_matrix(true_labels, predicted_labels)
        tn, fp, fn, tp= conf_matrix.ravel()
        result= {'model': model_name,
                'config': config_name,
                'train score': model.score(X, y),
                'predict score': predicted_scores,
                'Accuracy': (tp+tn)/(tp+tn+fp+fn),
                'Precesion': tp/(tp+fp),
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

    def confusion(self):
        pass

# tr= Train()
# tr.fit()
# print(tr.path)
# path= 'N:\MLM\trained_models'
te= Evaluate(r'N:\MLM\trained_models_2')
print(te.test())

'''
conf_matrix= confusion_matrix(true_labels, predicted_labels)
tn, fp, fn, tp= conf_matrix.ravel()
Accuracy= (tp+tn)/(tp+tn+fp+fn)
Precesion= tp/(tp+fp) 
Recall= tp/(tp+fn)
Specificity= tn/(tn+fp)
F1_score= 2*tp/(2*tp+fp+fn)
Negative_predictive_value= tn / (tn + fn)
False_positive_rate= fp/(fp + tn)
False_negative_rate= fn/(fn + tp)
logloss = log_loss(true_labels, predicted_probabilities)
mean_abolute_error= mean_absolute_error(true_labels, predicted_labels)
hinge_loss_value = hinge_loss(true_labels, predicted_scores)


from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(true_labels, predicted_probabilities)

'''
cols= ['model', 'config', 'score','Accuracy', 'Precesion', 'Recall', 'Specificity', 'F1 score', 'Negative predictive value',
       'False positive rate', 'False negative rate', 'mean abolute error', 'Log loss',  'Hinge loss'  ]



'''
#Accuracy
#Confusion matrix
#F1 score
#False positive rate
#False negative rate
#Log loss
#Negative predictive value
##Precesion
#Recall
#ROC Curve
#Specificity
#mean abolute error
quantile loss 
KL Divergence
#Hinge loss 
'''

'''
from mlxtend.plotting import plot_decision_regions
plot_decision_regions(xtr, ytr, clf= ir, legend =2 )


trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.3, random_state=1)
# predict 
P = model.predict_proba(testX)
clf.predict(xtest)
clf.score(X, y)
'''