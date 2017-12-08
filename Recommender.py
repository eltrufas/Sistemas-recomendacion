
class Recommender:
    def __init__(self, users, items, ratings):
        self.ratings = ratings
        self.users = users
        self.items = items

    def score(self, user_item_matrix):
        raise NotImplementedError

    
