from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Load data from https://www.openml.org/d/554
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist['data'], mnist['target']

print("Data successfully fetched")

# testing whether the data was loaded
some_digit = X[3600]
# some_dig_img = some_digit.reshape(28, 28)
# plt.imshow(some_dig_img, cmap = matplotlib.cm.binary, interpolation="nearest")
# plt.axis("off")
# plt.title(y[3600])
# plt.show()

# test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]



# 4/6 : 1/6 : 1/6 train, validation and test split
# X_train, X_val, X_test = np.split(X, [int(4/6*len(X)), int(5/6*len(X))])
# y_train, y_val, y_test = np.split(y, [int(4/6*len(y)), int(5/6*len(y))])

print("Split successful")

# split check
# print(X_train.shape, X_val.shape, X_test.shape)
# print(y_train.shape, y_val.shape, y_test.shape)

# converting data to integers
y_train = y_train.astype(np.int8)

# stochastic gradient descent
sgd_clf = SGDClassifier(random_state=42)

# random forest
rfc = RandomForestClassifier()

# bagging with decision trees
# bag_clf = BaggingClassifier(
#     DecisionTreeClassifier(), n_estimators=500,
#     max_samples=100, bootstrap=True, n_jobs=-1)


gbrt = GradientBoostingClassifier(max_depth=2, n_estimators=80, verbose=1)
print("Classifier created")
# gbrt.fit(X_train, y_train)
# print("Training complete")

print("Gradient Boosting classifier accuracy: {}".format(cross_val_score(gbrt, X_train, y_train, cv=3, scoring="accuracy")))

# # reshaping the X test set, so the predict method doesn't cry
# # yes, it is far from optimized solution
# reshaped_X_test = [0] * len(X_test)
# i = 0
# while i < len(X_test):
#     reshaped_X_test[i] = np.reshape(X_test[40], (1, -1))
#     i += 1
#
#
# # print("Random forest classifier: {}".format(cross_val_score(rfc, X_train, y_train, cv=3, scoring="accuracy")))
# # # print("Bagging classifier: {}".format(cross_val_score(bag_clf, X_train, y_train, cv=3, scoring="accuracy")))
# # print("Stochastic gradient descent classifier: {}".format(cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")))
# #print("Gradient Boosting classifier accuracy: {}".format(accuracy_score(y_test, gbrt.predict(X_test))))
#
# print(f"{y_test[50]} : {gbrt.predict(reshaped_X_test[50])}")






