# from sklearn.datasets import load_iris
# from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from scipy.sparse import load_npz

import pandas as pd
import numpy as np




def extention(adrs):
    result= str(adrs).rsplit('.')
    return result

def file_name(adrs):
    lst= extention(adrs)[0]
    return lst.split('\\')[-1]

class DataSet:
    def __init__(self, dt_adrs, label_adrs ,  test_size) -> None:
        self.data_dir= dt_adrs
        self.label_adrs= label_adrs
        self.test_size= test_size
        self.data_extention= extention(dt_adrs)[-1]
        self.label_extention= extention(label_adrs)[-1]
        self.split_lst= None
        self.__dataset_name= file_name(dt_adrs)


    def __read_csv(self, addrss):
        ''' reads pandas DataFrame'''
        dataset= pd.read_csv(addrss)

        return dataset


    def __read_numpy(self, addrss):
        ''' reads numpy array and sparse matrix'''
        dataset= np.load(addrss)
        return dataset

    def __read_sparse(self, addrss):
        dataset= load_npz(addrss) 
        return dataset
        



    def test_train(self):
        ''' 
         loads and Splits data set to test and train 
        '''

        # load data 
        if self.data_extention == 'csv':
            self.data= self.__read_csv( self.data_dir)
        elif self.data_extention == 'npz':
            self.data= self.__read_sparse(self.data_dir)
        elif self.data_extention == 'npy':
            self.data= self.__read_numpy(self.data_dir)
        #load label
        if self.label_extention == 'csv':
            self.label= self.__read_csv( self.label_adrs)
        elif self.label_extention == 'npz':
            self.label= self.__read_sparse(self.label_adrs)
        elif self.label_extention == 'npy':
            self.label= self.__read_numpy(self.label_adrs)   
        #TODO then can be added to shrink dataset
        # # decrease the dataset for svm ######********************************************************************************************************
        self.data= self.data[:3500,:]  
        self.label= self.label.head(3500)
        from sklearn.feature_selection import VarianceThreshold
        t= .8*(1-0.8)
        v = VarianceThreshold(threshold= t)
        self.data= v.fit_transform(self.data)
        self.label= v.fit_transform(self.label)


        # split data to train and test  

        if self.split_lst is None:
           self.split_lst = list(train_test_split(self.data, self.label, test_size=self.test_size, random_state=10))  # noqa: E501


    @property
    def train_dataset(self):
        if self.split_lst is None:
            self.test_train()
        data= self.split_lst[0]
        label= self.split_lst[2]# .values.ravel() # in case of shrinking 
        return data, label
    
    @property
    def test_dataset(self):
        if self.split_lst is None:
            self.test_train()
        data= self.split_lst[1]
        label= self.split_lst[3]# .values.ravel() 
        return data, label


    @property
    def dataset_name(self):
        return  self.__dataset_name



