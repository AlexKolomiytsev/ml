import numpy as np
from sklearn.naive_bayes import GaussianNB

def test():
    # features
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    # labels
    Y = np.array([1, 1, 1, 2, 2, 2])
    # classifier
    clf = GaussianNB()

    clf.fit(X, Y)

    prediction = clf.predict([[-0.8, -1]])

    print(prediction)