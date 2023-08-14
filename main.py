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
        self.__models= [LogRegression, DTModel, MLPClassifierModel, KNNModle]#  NBModel, , SVCModel] # noqa: E501
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


    def __run_single_model(self, model):
        if self.__dataset_object is None:
            raise ValueError('''The dataset is not provided. Please make sure to pass a 
                             valid dataset by calling "set_dataset" method.''')
        self.dataset= self.__dataset_object.train_dataset
        model_dir= self.__make_model_dir(model=model)
        mdl_obj= model(model_dir, self.dataset)
        all_conf, conf_cuntr = mdl_obj.train()
        print(f' The {model.__name__} trainning was just fineshed and the next model is going to be started...')
        return all_conf, conf_cuntr



    def fit(self):
        ''' fit() trains all models and their configs once and save a pickled file in each model directory.'''  # noqa: E501
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

    # passing dataset object to evaluation
    @property
    def dataset_obj(self):
        return self.__dataset_object



class Evaluate:
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
                    with open(f'{mdl_dir}\{str(mdl_config)}', 'br') as file:  # noqa: E999
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




# [LogRegression, DTModel, NBModel, MLPClassifierModel, KNNModle ]

def run(data_addr,  dataset_lst, label_dt):
    for dataset in dataset_lst:
        dt_adrs= os.path.join(data_addr, dataset)
        lbl_adrs= os.path.join(data_addr, label_dt)
        print(dt_adrs)
        print(lbl_adrs)
        tr= Train()
        tr.set_dataset(data_address= dt_adrs,
                    label_address=lbl_adrs)  # noqa: E501
        tr.fit()
        obj= tr.dataset_obj
        dt_name= obj.dataset_name
        mdl_adr= os.path.join(r'N:\MLM', dt_name)
        print(mdl_adr)
        te= Evaluate(models_dir=mdl_adr, data_obj= obj)
        te.test()
        print('*'* 500)
        del tr
        del te


g_lst= [
    #'gossipcop_bow.npz',
    #'gossipcop_one_gram.npz',
    #'gossipcop_bigram.npz',
    #'gossipcop_trigram.npz',
    #'gossipcop_one_to_trigram.npz',
    #'gossipcop_TFIDF.npz',
    #'gossipcop_W2V.npz',
    #'gossipcop_bow_w2v.npz',
    #'gossipcop_enhanc_bow_w2v.npz',
    #'gossipcop_TFIDF_w2v.npz',
    #'gossipcop_enhance_TFIDF_w2v.npz'
 ]
 
p_lst=[
    #'PolitiFact_bow.npz',
    #'PolitiFact_one_gram.npz',
    #'PolitiFact_bigram.npz',
    #'PolitiFact_trigram.npz',
    #'PolitiFact_one_to_trigram.npz',
    #'PolitiFact_TFIDF.npz',
    #'PolitiFact_W2V.npz',
    #'PolitiFact_bow_w2v.npz',
    #'PolitiFact_enhanc_bow_w2v.npz',
    #'PolitiFact_TFIDF_w2v.npz',
    #'PolitiFact_enhance_TFIDF_w2v.npz'
 ]
L_lst= [
   # 'Liar_bow.npz',
   # 'Liar_one_gram.npz',
    'Liar_bigram.npz',
    'Liar_trigram.npz',
    'Liar_one_to_trigram.npz',
    'Liar_TFIDF.npz',
    'Liar_W2V.npz',
    'Liar_bow_w2v.npz',
    'Liar_enhanc_bow_w2v.npz',
    'Liar_TFIDF_w2v.npz',
    'Liar_enhance_TFIDF_w2v.npz'
    ]

dt_lsts= [g_lst,p_lst,L_lst]
lbl_lst=['gossipcop_label.csv', 'PolitiFact_label.csv', 'Liar_label.csv']

for dt_lst, lbl in zip(dt_lsts, lbl_lst):
    run(data_addr=r'O:\Second Semister\dissertation\dis-dataset\GossioCop\train_data',
    dataset_lst =dt_lst  , 
    label_dt= lbl
    )

# tr= Train()
# tr.set_dataset(data_address= r'O:\Second Semister\dissertation\dis-dataset\GossioCop\train_data\gossipcop_bow.npz',
#             label_address=r'O:\Second Semister\dissertation\dis-dataset\GossioCop\train_data\gossipcop_label.csv')  # noqa: E501
# tr.fit()
# obj= tr.dataset_obj
# te= Evaluate(models_dir=r'N:\MLM', data_obj= obj)
# te.test()
# print('*'* 500)

import os 
os.system("shutdown /s /t 200")
'''

['gossipcop_bow.npz',
 'gossipcop_one_gram.npz',
 'gossipcop_bigram.npz',
 'gossipcop_trigram.npz',
 'gossipcop_one_to_trigram.npz',
 'gossipcop_TFIDF.npz',
 'gossipcop_W2V.npz',
 'gossipcop_bow_w2v.npz',
 'gossipcop_enhanc_bow_w2v.npz',
 'gossipcop_TFIDF_w2v.npz',
 'gossipcop_enhance_TFIDF_w2v.npz']
'''

'''
['PolitiFact_bow.npz',
 'PolitiFact_one_gram.npz',
 'PolitiFact_bigram.npz',
 'PolitiFact_trigram.npz',
 'PolitiFact_one_to_trigram.npz',
 'PolitiFact_TFIDF.npz',
 'PolitiFact_W2V.npz',
 'PolitiFact_bow_w2v.npz',
 'PolitiFact_enhanc_bow_w2v.npz',
 'PolitiFact_TFIDF_w2v.npz',
 'PolitiFact_enhance_TFIDF_w2v.npz']
'''

'''
['Liar_bow.npz',
 'Liar_one_gram.npz',
 'Liar_bigram.npz',
 'Liar_trigram.npz',
 'Liar_one_to_trigram.npz',
 'Liar_TFIDF.npz',
 'Liar_W2V.npz',
 'Liar_bow_w2v.npz',
 'Liar_enhanc_bow_w2v.npz',
 'Liar_TFIDF_w2v.npz',
 'Liar_enhance_TFIDF_w2v.npz']
'''


'''
from mlxtend.plotting import plot_decision_regions
plot_decision_regions(xtr, ytr, clf= ir, legend =2 )


# predict 
P = model.predict_proba(testX)
clf.predict(xtest)
clf.score(X, y)
'''