"""Usage example of CCEL estimators, i'll try to add more examples later"""

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier

from CCEL import CCELClassifier

data = datasets.load_digits()

c = CCELClassifier(tmax=150, po=0.5, pf=0.07, pm=0.4, N2=5, N1=100, N0=20, d1=5, eps1=0.005,
                   d2=10, eps2=0.05, d3=15, eps3=0.01, min_features=2, min_samples=0.5,
                   min_estimators=2,
                   base_estimator=DecisionTreeClassifier(max_depth=4), random_state=1)

c.fit(data.data, data.target)
print(c.score(data.data, data.target))