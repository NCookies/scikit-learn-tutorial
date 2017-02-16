import numpy as np
from sklearn import random_projection

from sklearn import datasets
from sklearn.svm import SVC


rng = np.random.RandomState(0)
X = rng.rand(10, 2000)
X = np.array(X, dtype='float32')

print X.dtype       # float32

# type casting
# fit_transform() changes float32(X) to float64(X_new)
transformer = random_projection.GaussianRandomProjection()
X_new = transformer.fit_transform(X)
print X_new.dtype   # float64


iris = datasets.load_iris()
clf = SVC()

print clf.fit(iris.data, iris.target)
"""
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
"""

print list(clf.predict(iris.data[:3]))
"""
[0, 0, 0]
"""

print clf.fit(iris.data, iris.target_names[iris.target])
"""
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
"""

print list(clf.predict(iris.data[:3]))
"""
['setosa', 'setosa', 'setosa']
"""
