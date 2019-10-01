from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np

iris_dataset = load_iris()
# print(iris_dataset.data)
# print(len(iris_dataset.target))
X_train, X_test, y_train, y_test = train_test_split(iris_dataset.data, iris_dataset.target, random_state=0)
model = DecisionTreeClassifier()
mymodel = model.fit(X_train, y_train)
X_New = np.array([[6.0, 3.23, 4.5, 2.0]])
print(mymodel.score(X_test, y_test))
