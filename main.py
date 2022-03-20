from datetime import datetime
import numpy as np
import numpy.random as rnd
import numpy.linalg as la
import matplotlib
import matplotlib.pyplot as plt
import os
from scipy.stats import multivariate_normal
import tikzplotlib


def create_directory(name):
    date_str = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    path = os.path.join('results', name, date_str)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def scm(X):
    _, N = X.shape
    return (1/N) * X @ X.T


def Tyler(X):
    p, N = X.shape
    sigma = scm(X)
    for iter_ in range(20):
        sigma_inv = la.inv(sigma)
        tmp = np.einsum('ji, jl, li -> i', X, sigma_inv, X).reshape((-1, 1))
        tmp = 1 / tmp
        sigma = (1/N) * X @ (tmp * X.T)
    return sigma


# parameters simulation
N = 50
N_outliers = 10

# seed
rnd.seed(3)

# matplotlib backend
matplotlib.use('Agg')

# path
path = create_directory('SCM_vs_Tyler')

# data generation
mean = np.zeros((2))
cov = np.array([[1, 0.9], [0.9, 1]])
X_data = rnd.multivariate_normal(mean=mean, cov=cov, size=N).T
mean = np.array([-5, 4])
cov = np.array([[0.3, -0.1], [-0.1, 0.3]])
X_outliers = rnd.multivariate_normal(mean=mean, cov=cov, size=N_outliers).T
X = np.concatenate([X_data, X_outliers], axis=1)

for estimator in ['SCM', 'Tyler']:
    # estimation
    if estimator == 'SCM':
        cov_est = scm(X)
    elif estimator == 'Tyler':
        cov_est = Tyler(X)
    # trace normalization
    cov_est = (cov_est / np.trace(cov_est)) * np.trace(scm(X))

    # meshgrid
    N_POINTS = 300
    MAX = 5
    x = np.linspace(-MAX, MAX, N_POINTS)
    y = np.linspace(-MAX, MAX, N_POINTS)
    x, y = np.meshgrid(x, y)
    pos = np.array([x.flatten(), y.flatten()]).T

    # plot
    plt.plot(X_data[0, :], X_data[1, :], 'b.')
    plt.plot(X_outliers[0, :], X_outliers[1, :], 'r.')
    mean = np.zeros((2))
    dist = multivariate_normal(mean, cov_est)
    plt.contour(x, y, dist.pdf(pos).reshape((N_POINTS, N_POINTS)),
                colors=['green'], alpha=0.8)
    plt.axis('equal')
    path_fig = os.path.join(path, estimator)
    plt.savefig(path_fig + '.png')
    tikzplotlib.save(path_fig + '.tex')
    plt.close('all')
