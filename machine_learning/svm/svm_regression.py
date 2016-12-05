import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# generating data
X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel()

# add noise to y
y[::5] += (0.5 - np.random.rand(8))

svr_rbf = svm.SVR(kernel='rbf', C=1e3, gamma=0.1).fit(X, y)
svr_lin = svm.SVR(kernel='linear', C=1e3).fit(X,y)
svr_poly = svm.SVR(kernel='poly', C=1e3, degree=3).fit(X, y)

y_rbf = svr_rbf.predict(X)
y_lin = svr_lin.predict(X)
y_poly = svr_poly.predict(X)


plt.scatter(X, y, color='darkorange', label='data', cmap=plt.cm.Paired)
lw = 2
plt.hold('on')
plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')
plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')
plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.axis('tight')
plt.show()
