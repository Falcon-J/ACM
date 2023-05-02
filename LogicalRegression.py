from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# load iris dataset from sklearn
iris = datasets.load_iris()

# create a pandas dataframe with the iris data
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['class'] = iris.target


x = iris.data
y = iris.target

# prepare inputs and outputs
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42)


# logistic regression

model = LogisticRegression()
model.fit(x, y.ravel())

# accuracy
score = model.score(x, y)
print("Accuracy: {:.6f}%".format(score * 100))

# make predictions
expected = y
predicted = model.predict(x)

# performance metrics
print()
print("Classification Report : ")
print()
print(metrics.classification_report(expected, predicted))
print()
print("Confusion Matrix : ")
print()
print(metrics.confusion_matrix(expected, predicted))
print()
# pair plot
sns.set(style="darkgrid", color_codes=True)
sns.pairplot(iris_df, hue='class', palette="Set2")
plt.show()
