# ACM
 AI/ML



**Logistic Regression on Iris dataset**



Overview: 

This code demonstrates the use of logistic regression for classification of the iris dataset. 

Code Explanation:

Import necessary libraries: 
This code imports several packages including metrics, LogisticRegression from sklearn.linear_model, pandas, matplotlib.pyplot, seaborn, and datasets from sklearn.

Load and prepare the iris dataset: 
The code loads the iris dataset from sklearn and converts it into a pandas dataframe. The 'class' column is then mapped from numerical values to their corresponding class names. Finally, the input and output variables are prepared.

Visualize the dataset: The code creates a pair plot of the iris dataset using seaborn.

Logistic Regression: The code defines a logistic regression model, fits it to the data, and calculates the accuracy score.

Predictions and performance metrics: The code makes predictions using the model and calculates the classification report and confusion matrix.

The output of the code includes the accuracy score, the classification report, and the confusion matrix. These metrics are used to evaluate the performance of the logistic regression model.









**Artificial Neural Network on Iris dataset**



Overview :

This code builds a neural network model using the Keras library to classify iris flowers based on their sepal and petal measurements. It uses the Iris dataset, which is a commonly used dataset in machine learning and is available in the scikit-learn library. The model is trained on a subset of the dataset and its accuracy and loss are plotted over epochs.


Code Explanation:

Importing the necessary libraries: 
The code starts by importing the required libraries Keras, sklearn(scikit-learn), and matplotlib.

Loading the Iris dataset: 
The Iris dataset is loaded using the load_iris() function from scikit-learn, and the input features and target labels are stored in x and y variables, respectively.

Splitting the dataset:
The dataset is split into training and testing sets using the train_test_split() function from scikit-learn. 20% of the data is reserved for testing.

Building the neural network model: The neural network model is built using the Sequential API from Keras. The model has two dense layers, with 100 units in the first layer and 3 units in the output layer. The input shape is set to (4,), which corresponds to the number of input features.


Compiling the model: 
The model is compiled using the compile() function, which sets the optimizer, loss function, and evaluation metric. In this case, the Adam optimizer is used, the loss function is set to sparse_categorical_crossentropy, and the metric is set to accuracy.


Fitting the model: 
The model is trained using the fit() function, which takes in the training data, labels, and number of epochs. The history variable is used to store the accuracy and loss values of the model during training.



Evaluating the model:
 The model is evaluated on the testing data using the evaluate() function, which returns the test loss and accuracy.

Plotting the accuracy and loss: 
The accuracy and loss of the model during training are plotted using the plot() function from matplotlib. The x-axis shows the epoch number, and the y-axis shows the accuracy and loss values.




