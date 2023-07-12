from model.models import *
from model.dtsets import Iris
import os

class MLM:
    def __init__(self):
        self.base_dir= os.getcwdb().decode()
        self.__make_main_dir()
        self.models= [LogRegression, SVCModel]

    def __make_main_dir(self):
        self.path= os.path.join(self.base_dir, 'trained_models')
        try:
            os.mkdir(self.path)
        except FileExistsError:
            counter= 1
            while True:
                counter+= 1
                try:
                    self.path= os.path.join(self.base_dir, f'trained_models_{counter}')
                    os.mkdir(self.path)
                except FileExistsError:
                    pass
                else:
                    break
        except Exception as err:
            print(err)
            
    def __make_model_dir(self, model): # model= class.__name__
        model_dir= os.path.join(self.path, model.__name__)
        os.mkdir(model_dir)
        return model_dir


    def __run_single_model(self, model):
        model_dir= self.__make_model_dir(model=model)
        mdl_obj= model(model_dir)
        all_conf, conf_cuntr = mdl_obj.train()
        return all_conf, conf_cuntr


    def fit(self):

        for model in self.models:
            self.__run_single_model(model=model)

    def predict(self):
        pass

    def score(self):
        pass

    def confusion(self):
        pass


        
#TODO before calling any model should the directory should be created
# ob= Log_Regression()
# ob.train()


ob= MLM()
ob.fit()



'''
from mlxtend.plotting import plot_decision_regions
plot_decision_regions(xtr, ytr, clf= ir, legend =2 )

'''