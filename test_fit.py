"""
Ajuste de modelo de ejemplo
"""
import numpy as np
import pandas as pd
from regression_model import normalized_fit
from math import sqrt
import pickle

ratings1 = pd.read_csv('datasets/jester/UserRatings1.csv')
ratings2 = pd.read_csv('datasets/jester/UserRatings2.csv')

user_ratings = pd.merge(ratings1, ratings2, on='JokeId')

# tomamos solo un subconjunto de los datos
user_ratings = user_ratings.loc[:, 'User1':].sample(20000, axis=1)


UI_MATRIX = user_ratings.as_matrix()

NUM_FEATURES = 500
LAMBDA = 1

X, THETA, MEANS = normalized_fit(UI_MATRIX, NUM_FEATURES, LAMBDA)

PREDICTIONS = np.dot(X, THETA.T) + MEANS.reshape(-1, 1)

R_MASK = np.isfinite(UI_MATRIX)

ERROR = sqrt(np.mean((PREDICTIONS[R_MASK] - UI_MATRIX[R_MASK])**2))

print(ERROR)

with open('data/jester_fit.p', 'wb') as fp:
    pickle.dump((UI_MATRIX, X, THETA, MEANS), fp)
