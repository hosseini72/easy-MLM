from sklearn.datasets import load_iris
from sklearn.datasets import make_classification

class Iris:

    @property
    def dataset(self):
        X, y= load_iris(return_X_y= True)
        return X, y 


class Random_dataset:

    @property
    def dataset(self):
        X,y= make_classification(n_samples= 100, n_features=3, n_informative=15, n_redundant=5, random_state=5)
        return X,y