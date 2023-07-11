from sklearn.datasets import load_iris
from sklearn.datasets import make_classification




def dt():
    return load_iris(return_X_y= True)


class Iris:

    @property
    def dataset(self):
        # X, y= load_iris(return_X_y= True)
        # print(X)
        # result= {'X_train': X, 'y_train':y}
        iris = load_iris()
        data = iris.data
        target = iris.target
        return data, target


class Random_dataset:

    @property
    def dataset(self):
        X,y= make_classification(n_samples= 100, n_features=3, n_informative=2, n_redundant=2, random_state=5)
        return {'X_train': X, 'y_train':y}
    

