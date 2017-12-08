import pickle
import numpy as np
import pandas as pd
from recommend import SingleUserItemRecommender

joke_text = pd.read_csv('datasets/jester/JokeText.csv').JokeText
print(joke_text[3])

with open('data/jester_fit.p', 'rb') as fp:
    ui, x, theta, means = pickle.load(fp)

print(theta.shape)
print(x.shape)

new_user = np.zeros(x.shape[0])
new_user[:] = np.nan

popularity = np.sum(np.isfinite(ui).astype(float), axis=1)

recommender = SingleUserItemRecommender(x, popularity, means)
recommender.fit(new_user)

prediction = recommender.predict_ratings()
queue = list(recommender.initial_recommendations())

test_user = ui[:, np.random.choice(np.arange(theta.shape[0]))]
recommender.fit(test_user)
print(recommender.get_best())

while False:
    if not queue:
        queue.append(recommender.get_best())
        print(queue[0])
    next = queue.pop(0)
    print(joke_text[next])
    print('Rate this joke')
    choice = float(input())

    choice = choice * 2 - 10

    print('predicted {}'.format((prediction[next] + 10) / 2))

    new_user[next] = choice

    recommender.fit(new_user)
    prediction = recommender.predict_ratings()
