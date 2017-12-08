import numpy as np
from math import floor
from regression_model import fit_user
from sklearn.cluster import KMeans

example = np.asarray([[5, 5, 0, 0],
                      [5, np.nan, np.nan, 0],
                      [np.nan, 4, 0, np.nan],
                      [0, 0, 5, 4],
                      [0, 0, 5, np.nan]])

features = np.asarray([[0.9, 0],
                       [1, 0.01],
                       [0.99, 0],
                       [0.1, 1.0],
                       [0, 0.9]])


class SingleUserItemRecommender:
    def __init__(self, features, relevance, means=None, lam=1):
        if means is None:
            means = np.zeros(features.shape[0])
        self.features = features
        self.means = means
        self.lam = lam
        self.theta = None
        self.relevance = relevance
        self.make_clusters()

    def make_clusters(self):
        n_clusters = min(8, self.means.size)
        kmeans = KMeans(n_clusters)

        clusters = kmeans.fit_predict(self.features)

        self.cluster_mask = np.asarray([clusters == i for i in range(n_clusters)])

    def get_best(self, masks=None, reverse=False):
        if masks is None:
            masks = np.ones(self.features.shape[0], dtype=np.bool)

        masked_array = np.copy(self.relevance)

        if reverse:
            masked_array = -masked_array

        masked_array[self.rated_mask.astype(np.bool)] = -np.inf
        masked_array = np.tile(masked_array, (masks.shape[0], 1))
        masked_array[~masks] = -np.inf
        sorted_array = np.argsort(masked_array, axis=1)

        return sorted_array[:, -1]

    def fit(self, ratings):
        rated_mask = np.isfinite(ratings).astype(int)
        ratings = np.nan_to_num(ratings)
        theta = fit_user(ratings, rated_mask, self.features, self.means, self.lam)

        self.theta = theta
        self.rated_mask = rated_mask

        return self.predict_ratings()

    def predict_ratings(self):
        if self.theta is None:
            raise ValueError('No se ha ajustado el modelo')
        return np.dot(self.features, self.theta.T) + self.means

    def initial_recommendations(self):
        return self.get_best(self.cluster_mask)

    def recommend_good(self):
        predicted_ratings = self.predict_ratings()

        cutoff = max(floor((self.rated_mask == 0).size * 0.5), 1)

        predicted_ratings[self.rated_mask.astype(np.bool)] = -np.inf

        indices = np.argsort(predicted_ratings)

        choices = indices[-cutoff:]
        index = np.random.choice(choices)

        return index
