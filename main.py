from models.models import Log_Regression
from dtsets import Iris

X,y = Iris().dataset



ob= Log_Regression()
mod= ob.configure()
mod.fit(X)

res= mod.predict(y)
print(res)

