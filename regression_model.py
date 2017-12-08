'''
Cosa.py
'''
import numpy as np
from scipy.optimize import fmin_cg


def simple_fit(ui_matrix, r_matrix, num_features, lam):
    """
    Ajusta el modelo de regresion dada una matriz de calificaciones y
    una mascara de calificados
    """
    y = ui_matrix
    r = r_matrix
    num_items, num_users = y.shape

    theta0 = np.random.rand(num_users, num_features)
    x0 = np.random.rand(num_items, num_features)

    def fold_matrices(x_matrix, theta_matrix):
        return np.concatenate([x_matrix.flatten(), theta_matrix.flatten()])

    def unfold_vector(x):
        x_matrix = np.reshape(x[:x0.size],
                              x0.shape)
        theta_matrix = np.reshape(x[x0.size:],
                                  theta0.shape)
        return x_matrix, theta_matrix

    def unfold_parameter(f):
        def wrapper(x):
            return f(*unfold_vector(x))

        return wrapper

    @unfold_parameter
    def optimization_target(x, theta):
        differences = np.multiply((np.dot(x, theta.T) - y), r)
        square_error = (0.5) * np.sum(differences**2)
        regularization = (lam / 2) * (np.sum(x**2) + np.sum(x**2))

        return square_error + regularization

    @unfold_parameter
    def gradient(x, theta):
        differences = np.multiply((np.dot(x, theta.T) - y), r)
        x_grad = np.dot(differences, theta) + lam * x
        theta_grad = np.dot(x.T, differences).T + lam * theta

        return fold_matrices(x_grad, theta_grad)

    init_fold = fold_matrices(x0, theta0)
    result = fmin_cg(f=optimization_target, x0=init_fold, fprime=gradient)

    x, theta = unfold_vector(result)

    return x, theta


def fit_user(ratings, mask, features, means=None, lam=1):
    if means is None:
        means = np.zeros(ratings.shape)

    theta0 = np.random.rand(features.shape[1])

    def optimization_target(theta):
        differences = np.multiply((np.dot(features, theta.T) - ratings), mask)
        return 0.5 * np.sum(differences**2) + 0.5 * lam * np.sum(theta**2)

    def gradient(theta):
        differences = np.multiply((np.dot(features, theta.T) - ratings), mask)
        return np.dot(features.T, differences) + lam * theta

    theta = fmin_cg(f=optimization_target, x0=theta0, fprime=gradient)

    return theta


def normalized_fit(y, *args):
    means = np.nanmean(y, axis=1)
    y = y - means.reshape(-1, 1)

    r = -(np.isnan(y).astype(int) - 1)
    y = np.nan_to_num(y)

    x, theta = simple_fit(y, r, *args)

    return x, theta, means


def get_score(user, item, x, theta, means=None):
    offset = 0 if means is None else means[item]

    user_theta = theta[user]
    item_x = x[item]

    return np.dot(user_theta.T, item_x) + offset
