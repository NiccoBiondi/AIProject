import mnist_reader
import plot_curve as pc
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit
plt.rcParams['figure.figsize'] = (10.0, 8.0)

print("The dataset is loading")
X_train, y_train = mnist_reader.load_mnist("C:\\fashion-mnist-master\\data\\fashion", kind='train')
X_test, y_test = mnist_reader.load_mnist("C:\\fashion-mnist-master\\data\\fashion", kind='t10k')
X = np.concatenate((X_train, X_test))
y = np.concatenate((y_train, y_test))
print(X.shape, y.shape)

print("Random Forest is running")
title = "Random Forest"
cv = ShuffleSplit(n_splits=10, test_size=10000, random_state=0)
#Random Forest without pruning
classifier = RandomForestClassifier(n_estimators=30, criterion='gini')
#Random Forest with pruning
#classifier = RandomForestClassifier(n_estimators=30, criterion='gini', min_samples_leaf=2)
pc.plot_curve(classifier, X, y, title=title,
              cv=cv, train_sizes=np.linspace(0.05, 1.0, 10), error=True)

plt.show()
