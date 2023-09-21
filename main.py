from model.models import LogRegression, DTModel, MLPClassifierModel, KNNModle,  NBModel, SVCModel  # noqa: E501
from model.config import *
from sklearn.naive_bayes import GaussianNB
from model.dtsets import DataSet
from scipy import sparse 
from random import choice
from typing import Union, List, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from sklearn.metrics import accuracy_score,precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import log_loss
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import hinge_loss

'''
This package can be helpful for model selection
'''

class __Train:
    config_dict= {

    }
    def __init__(self, base_dir, models_obj):
        self.__base_dir= base_dir
        self.__models_obj= models_obj
        self.dataset= None

    def set_dataset(self, *, data_address, label_address, test_size=0.2 ):
        self.__dataset_object= DataSet(data_address, label_address, test_size)
        self.__make_main_dir()


    def __make_main_dir(self):
        file_name= self.__dataset_object.dataset_name
        self.__models_path= os.path.join(self.__base_dir, file_name)  # noqa: E501
        try:
            os.mkdir(self.__models_path)
        except FileExistsError:
            counter= 1
            while True:
                counter+= 1
                try:
                    self.__models_path= os.path.join(self.__base_dir, f'{file_name}_{counter}')  # noqa: E501
                    os.mkdir(self.__models_path)
                    #TODO should change the name of created file name, in evaluation it may not find the file to run test  # noqa: E501
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


    def __run_single_model(self, model, model_config):
        if self.__dataset_object is None:
            raise ValueError('''The dataset is not provided. Please make sure to pass a 
                             valid dataset by calling "set_dataset" method.''')
        self.dataset= self.__dataset_object.train_dataset
        model_dir= self.__make_model_dir(model=model)
        mdl_obj= model(model_dir, self.dataset, model_config)  #TODO here pass the config class 
        all_conf, conf_cuntr = mdl_obj.train()
        print(f' The {model.__name__} trainning was just fineshed and the next model is going to be started...')  # noqa: E501
        return all_conf, conf_cuntr



    def fit(self):
        ''' fit() trains all models and their configs once and save a pickled file in each model directory.'''  # noqa: E501
        for model, model_config in self.__models_obj.items():
            self.__run_single_model(model, model_config)  # noqa: E999
    
    def fit_one(self, model= None):
        if not model:
            model= choice(self.__models_obj)
        self.__run_single_model(model, self.__models_obj.get(model))


    @property
    def path(self):
        return self.__models_path
    
    @path.setter
    def path(self, new_dir):
        if os.path.isdir(new_dir):
            self.__base_dir= new_dir
        else:
            raise Exception('These models are permanents and can not be changed. To run espesific model fit_one(model_name) method can be used.')  # noqa: E501

    @path.deleter
    def path(self):
        raise Exception('These models are permanents and can not be deleted. To run espesific model fit_one(model_name) method can be used.')  # noqa: E501

    # passing dataset object to evaluation
    @property
    def dataset_obj(self):
        return self.__dataset_object



class __Evaluate:
    def __init__(self, *, models_dir,  data_obj) -> None:
        self.__path= models_dir
        self.__models_dir_list= iter(os.listdir(self.__path))
        self.data, self.target= data_obj.test_dataset

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
                    with open(f'{mdl_dir}\{str(mdl_config)}', 'br') as file: # noqa:E999
                        trained_mdl= pickle.load(file= file)
                    result= self.__evaluate(model_name=mdl, config_name=mdl_config, model= trained_mdl)  # noqa: E501
                    result_list.append(result)
                # else:
                #     print('mdl changed from ', mdl)
            except StopIteration:
                break
        evaluation_df= pd.DataFrame(result_list)
        print(self.__path)
        evaluation_df.to_csv(f'{self.__path}\MLM.csv')  #TODO \MLM.csv to >> {mdl}.csv
        return evaluation_df

    def __evaluate(self, model_name, config_name, model):
        ##### tempory 
        if isinstance(model,GaussianNB) and isinstance(self.data, sparse._csr.csr_matrix):  # noqa: F405
            self.data= self.data.toarray()
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
                # 'Log loss': log_loss(true_labels, predicted_probabilities),
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
    

class MLM:
    __default_models= ['LogReg', 'DecisionTree', 'MLP', 'KNN',  'NaiveBase', 'SVC']  # noqa: E501
    def __init__(self,*, data_dir: Union[str, Path], 
                 data_set_list: Union[List, Tuple], models_list: List= [] ,
                  label_dataset, result_path : Union[str, Path]= None) -> None:
        # data_dir is the direcotory that datasets are in.
        if data_dir is None or not (os.path.exists(data_dir) and os.path.isdir(data_dir)):  # noqa: E501:
            raise ValueError('directory path of data sets should be passed.')
        else:
            self.data_addr= data_dir
        # data_set_list is the list of datasets, if there is one dataset, it can be passed in list or tuple with single element  # noqa: E501
        if data_set_list is None:
            raise ValueError('datasets names should be passed in a list or a tuple.'
                             'if there is one dataset, its name should be passed in '
                             'a single member tuple or list.')
        else:
            self.data_set_list= data_set_list
        # label_dataset is the name of label which are in other datasets direcotry
        if label_dataset is None:
            raise ValueError('Label data set in mandetory for training and the name of '
                              'label data set should be passed.')
        else:
            self.label_dt= label_dataset
        # models_list contain model(s) which are trained.
        if MLM.__check_models_list(models_list) is None:
            self.models_lst= MLM.__default_models # noqa: E501
        else:
            self.models_lst= models_list

        # base_dir is the direcotry that trained models will be stored in. 
        if result_path is None or not (os.path.exists(result_path) and os.path.isdir(result_path)):  # noqa: E501
            self.base_dir= os.getcwdb().decode() 
        else:
            self.base_dir= result_path
        # config parameters
        self.LogReg= log_regression_config()
        self.DecisionTree= dtc_onfig()
        self.MLP= mpl_config()
        self.KNN= knn_config()
        self.NaiveBase= nbc_config()
        self.SVC= svc_onfig()
        self.models_config= {
            'LogReg':  self.LogReg, 
            'DecisionTree': self.DecisionTree, 
            'MLP': self.MLP, 
            'KNN':  self.KNN, 
            'NaiveBase': self.NaiveBase,
            'SVC':  self.SVC
        }
        
    @staticmethod
    def __check_models_list(mdl_lst):
        if mdl_lst is not None:  
            for mdl in mdl_lst:
                if mdl not in  MLM.__default_models:
                    raise ValueError(f"the {mdl} is not one of models of this class."
                                     "the models are:"
                                     "{MLM.__default_models}")
                    return None   
        return mdl_lst
    
    @staticmethod
    def __model_objects(mdl_lst):
        mdl_dict= {
            'LogReg': LogRegression, 
            'DecisionTree': DTModel,
            'MLP': MLPClassifierModel,
            'KNN': KNNModle,
            'NaiveBase': NBModel,
            'SVC': SVCModel,
            }
        mdl_obj= [mdl_dict.get(mdl) for mdl in mdl_lst]
        return mdl_obj
    
    def run(self):
        for dataset in self.dataset_lst:
            dt_adrs= os.path.join(self.data_addr, dataset)
            lbl_adrs= os.path.join(self.data_addr, self.label_dt)
            tr= __Train(self.base_dir, MLM.__model_objects(self.models_lst), self.models_config)  # noqa: E501
            tr.set_dataset(data_address= dt_adrs,
                        label_address=lbl_adrs)  
            tr.fit()
            obj= tr.dataset_obj
            dt_name= obj.dataset_name
            mdl_adr= os.path.join( self.base_dir, dt_name)
            te= __Evaluate(models_dir=mdl_adr, data_obj= obj)
            te.test()
            del tr
            del te

    @property
    def models(self):
        print(f' there is/are {len(self.models_lst)} model(s):')
        for i, model in enumerate(self.models_lst, start=1):
            print(f'{i}: {model}')
        return self.models_lst
    
    @models.setter
    def models(self, item):
        if MLM.__check_models_list(item) is not None:
            self.models_lst= item

    @models.deleter
    def models(self):
        raise Exception('These models are permanents and can not be deleted. To run espesific model fit_one(model_name) can be used.')  # noqa: E501



