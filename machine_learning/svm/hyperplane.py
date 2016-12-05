import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

np.random.seed(50)
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
y = [0] * 20 + [1] * 20
y = np.array(y)
clf = svm.SVC(kernel="linear").fit(X, y)
w = clf.coef_[0]
b = clf.intercept_ / w[1]
a = w[0] / w[1]
xx = np.linspace(-5, 5)

yy = -a * xx - b
fig = plt.figure(figsize=(10, 10))
plt.plot(xx, yy, 'k-')

b_down = clf.support_vectors_[0]
yy_down = -a * xx + b_down[1] + a * b_down[0]
plt.plot(xx, yy_down, 'k--')

b_up = clf.support_vectors_[-1]
yy_up = -a * xx + b_up[1] + a * b_up[0]
plt.plot(xx, yy_up, 'k--')

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=80, facecolors='none')
plt.axis('tight')
plt.show()