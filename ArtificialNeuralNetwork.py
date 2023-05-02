
from keras.models import Sequential
from keras.layers import Dense
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Loading Dataset
data = datasets.load_iris()

x = data.data
y = data.target

# Split Dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Building the Model
model = Sequential()
model.add(Dense(100, input_shape=(4,), activation="relu"))
model.add(Dense(3, activation='softmax'))

# Compile the Model
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Fit the Model
history = model.fit(x_train, y_train, epochs=20)

# Evaluate the Model
model.evaluate(x_test, y_test)

# Plot the accuracy and loss over epochs
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.title('Model Accuracy and Loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy / Loss')
plt.legend(['Accuracy', 'Loss'], loc='upper right')
plt.show()
