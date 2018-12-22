import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve


def plot_curve(classifier, X, y, title="",
               cv=None, train_sizes=None, error = True, ylim=None):

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training set")
    if error:
        plt.ylabel("Error")
    else :
        plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        classifier, X, y, cv=cv, train_sizes=train_sizes)
    if error:
        train_scores = 1-train_scores
        test_scores = 1-test_scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid(True)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std,
                     alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std,
                     alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training error")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Test error")
    plt.legend(loc="best")
    return plt
