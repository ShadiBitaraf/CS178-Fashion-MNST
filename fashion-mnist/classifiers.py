import numpy as np
import pandas as pd
import tensorflow as tf


from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV



#kNN Classifiier:

# Load the Fashion-MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

# Flatten the image data
train_images = train_images.reshape(-1, 28 * 28)
test_images = test_images.reshape(-1, 28 * 28)
# Create a kNN classifier
#Using RandomizedSearchCV to perform random search with cross-validation, trying different k values and selecting the best one based on the accuracy score. 

# Define the parameter grid for random search
param_grid = {'n_neighbors': np.arange(1, 21)}

# Create a kNN classifier
knn = KNeighborsClassifier()

# Perform random search for best k value
random_search = RandomizedSearchCV(knn, param_grid, cv=5, n_iter=10, scoring='accuracy')
random_search.fit(train_images, train_labels)

# Print the best k value and its corresponding test accuracy
best_k = random_search.best_params_['n_neighbors']
best_accuracy = random_search.best_score_
print(f"Best k: {best_k}")
print(f"Best Accuracy: {best_accuracy}")

# Create a kNN classifier with the best k value
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(train_images, train_labels)

# Make predictions on the test data
test_predictions = best_knn.predict(test_images)

# Calculate accuracy on the test data
test_accuracy = accuracy_score(test_labels, test_predictions)
print(f"kNN Classifier test Accuracy for k={best_k}: {test_accuracy}")


#Random Forest Classifier:

# Create a Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100)

# Fit the classifier to the training data
rf_classifier.fit(train_images, train_labels)

# Make predictions on the test data
test_predictions = rf_classifier.predict(test_images)

# Calculate accuracy on the test data
test_accuracy = accuracy_score(test_labels, test_predictions)
print("Random Forest Classifier test Accuracy:", test_accuracy)