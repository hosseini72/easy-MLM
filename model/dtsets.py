from sklearn.datasets import load_iris
from sklearn.datasets import make_classification



def split_dataset(dataset):
    pass
# trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.3, random_state=1)


class DataSet:

    def test_train(self):
        ''' 
        Splits data set to test and train 
        '''
        pass

    @property
    def train_dataset(self):
        # X, y= load_iris(return_X_y= True)
        # print(X)
        # result= {'X_train': X, 'y_train':y}
        iris= load_iris()
        data= iris.data
        target= iris.target
        target[target==2] = 1

        return data, target
    
    @property
    def test_dataset(self):
        iris= load_iris()
        data= iris.data
        target= iris.target
        target[target==2] = 1
        return data, target



class Random_dataset:

    @property
    def dataset(self):
        X,y= make_classification(n_samples= 100, n_features=3, n_informative=2, n_redundant=2, random_state=5)
        return {'X_train': X, 'y_train':y}
    

