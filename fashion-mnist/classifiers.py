import numpy as np
import pandas as pd
import torch # pytorch
import tensorflow as tf


from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score



#kNN Classifiier:

# Load the Fashion-MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

# Flatten the image data
train_images = train_images.reshape(-1, 28 * 28)
test_images = test_images.reshape(-1, 28 * 28)

# Create a kNN classifier
k_values = [3, 5, 7, 9, 11]
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1) 

    # Fit the classifier to the training data
    knn.fit(train_images, train_labels)

    # Make predictions on the test data
    test_predictions = knn.predict(test_images)

    # Calculate accuracy on the test data
    test_accuracy = accuracy_score(test_labels, test_predictions)
    print(f"kNN Classifier test Accuracy for k={k}: {test_accuracy}")


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