# import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# load the iris dataset
iris = load_iris()

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

# create a k-NN classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)

# fit the classifier to the training data
knn.fit(X_train, y_train)

# make predictions on the testing data
y_pred = knn.predict(X_test)

# print the accuracy score
print("Accuracy:", knn.score(X_test, y_test))
