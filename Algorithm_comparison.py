from sklearn.datasets import load_digits
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib
import matplotlib.pyplot as plt


# loading data
X, y = load_digits(return_X_y=True)
print("Data successfully fetched")

# test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Split successful")


# stochastic gradient descent
sgd = SGDClassifier()

# random forest
rfc = RandomForestClassifier()

# gradient boosting
gb = GradientBoostingClassifier()

# parameter responsible for density of points
# smaller the step, more data points on the plot
step = 50

# creating empty lists for data
data_sgd = [0] * (int(len(y_train) / step) + 1)
data_rfc = [0] * (int(len(y_train) / step) + 1)
data_gb = [0] * (int(len(y_train) / step) + 1)

for i in range(step, len(y_train), step):
    # data split
    X_t, y_t = X_train[:i], y_train[:i]

    # SGD
    sgd.fit(X_t, y_t)
    data_sgd[int(i/step)] = sgd.score(X_test, y_test)

    # RFC
    rfc.fit(X_t, y_t)
    data_rfc[int(i/step)] = rfc.score(X_test, y_test)

    # GB
    gb.fit(X_t, y_t)
    data_gb[int(i/step)] = gb.score(X_test, y_test)

x_vals = [i for i in range(0, step*len(data_sgd), step)]

# plotting the data
plt.plot(x_vals, data_sgd, label='SGD')
plt.plot(x_vals, data_rfc, label='RFC')
plt.plot(x_vals, data_gb, label='GB')

ax = plt.subplot(111)
ax.legend(loc='lower right')
plt.xlabel("Number of data samples")
plt.ylabel("Score")
plt.grid()
plt.title(f"Step = {step}")

plt.show()


