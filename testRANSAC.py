import numpy as np
from matplotlib import pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D

from sklearn import linear_model, datasets

#n_samples = 1000
#n_outliers = 50

n_samples = 50
n_outliers = 5

X, y, coef = datasets.make_regression(n_samples=n_samples, n_features=1,
        n_informative=1, noise=10, coef=True, random_state=0)

np.random.seed(2)
X[:n_outliers] = 3 + 0.5 * np.random.normal(size=(n_outliers, 1))
y[:n_outliers] = -3 + 10 * np.random.normal(size=n_outliers)

lr = linear_model.LinearRegression()
lr.fit(X, y)

ransac = linear_model.RANSACRegressor()
start = time.time()
ransac.fit(X, y)
print('Fit time: {}'.format(time.time() - start))
print(ransac.estimator_.coef_)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

line_X = np.arange(X.min(), X.max())[:, np.newaxis]
line_y = lr.predict(line_X)
line_y_ransac = ransac.predict(line_X)

"""plt.scatter(X[inlier_mask], y[inlier_mask], color='yellowgreen', marker='.', label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask], color='gold', marker='.', label='Outliers')
plt.plot(line_X, line_y, color='navy', label='Linear regressor')
plt.plot(line_X, line_y_ransac, color='cornflowerblue', label='RANSAC regressor')
plt.legend(loc='lower right')
plt.xlabel('Input')
plt.ylabel('Response')
plt.show()"""
d = datasets.load_iris()
x = d.data
y = d.target
xt = x[:,(0,1)]
r = linear_model.RANSACRegressor()
r.fit(xt, y)
print(r.estimator_.intercept_, r.estimator_.coef_)

#t = np.linspace(2, 5, 50)
#x, y = np.meshgrid(t, t)
#y_pred = r.predict(xt)
y_pred = np.dot(xt, r.estimator_.coef_) + r.estimator_.intercept_ + 3

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(x[:,0], x[:,1], y, '.')
ax.plot3D(x[:,0], x[:,1], y_pred, '.')
plt.show()
