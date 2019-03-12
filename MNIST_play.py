from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Load data from https://www.openml.org/d/554
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)


# testing whether the data was loaded
some_digit = X[3600]
# some_dig_img = some_digit.reshape(28, 28)
# plt.imshow(some_dig_img, cmap = matplotlib.cm.binary, interpolation="nearest")
# plt.axis("off")
# plt.title(y[3600])
# plt.show()

# data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# converting data to integers
y_train = y_train.astype(np.int8)

# stochastic gradient descent
sgd_clf = SGDClassifier(random_state=42)

# random forest
rfc = RandomForestClassifier()

# bagging with decision trees
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=500,
    max_samples=100, bootstrap=True, n_jobs=-1)

print("Random forest classifier: {}".format(cross_val_score(rfc, X_train, y_train, cv=3, scoring="accuracy")))
print("Bagging classifier: {}".format(cross_val_score(bag_clf, X_train, y_train, cv=3, scoring="accuracy")))
print("Stochastic gradient descent classifier: {}".format(cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")))


