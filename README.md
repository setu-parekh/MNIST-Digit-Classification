# MNIST Digit Classification
Building Naive Bayes classification model to classify digits from MNIST database.

## Summary
* [Introduction & General Information](#introduction--general-information)
* [Objectives](#objectives)
* [Data Used](#data-used)
* [Approach & Methodology](#approach--methodology)
* [Run Locally](#run-locally)
* [Conclusion](#conclusion)

## Introduction & General Information
- We are using Naive Bayes Classification technique to classify the handwritten digits in MNIST database.
- Naive Bayes Classifier is a probabilistic machine learning model. It works on the principle of Bayes Theorem.
- Bayes Theorem states that P(y|X) = (P(X|y) * P(y)) / P(X).
  - Here y is the class label (digit 0 or 1), X represents the image features.
- Using this theorem, we can find the class probability of the data for given set of features. It is assumed at all features are independent of each other.

## Objectives
- For the purpose of this project, we are considering only a subset of MNIST dataset. We filter the training and testing sets to obtain the images of digits 0 and 1 only.
- The main objective of this project is to apply the concept of Naive Bayes Classification technique on the subset of MNIST dataset and evaulate the performance of the classifier.

## Data Used
(Data Source: http://yann.lecun.com/exdb/mnist/)
- We are using MNIST dataset containing images of handwritten digits from 0 - 9.
- Each image is of size 28 pixels x 28 pixels, making 784 pixels in total. This dataset is divided into training and testing sets. The training set consists of 60,000 images and testing set consists of 10,000 images.
- We build the model based on the training dataset and use testing dataset to evaluate the performance of the classification model.

## Approach & Methodology
- Loading and Filtering MNIST Dataset to obtain the images of digits 0 and 1.
- Convert the given dataset in 3D Numpy format to 2D pandas dataframe. Now each row is representing a single image in pandas dataframe.
- Each image consists of two features namely, average brightness and standard deviation. Perform feature extraction for each image from train and test datasets of digits 0 and 1.
- Compute density parameters - Mean and Variance for each of the extracted features.
- Apply Naive Baye's classifier technique to classify the image as 0 or 1 on the test dataset by using the density parameters computed above.
- Evaluate the performance of the classifier by computing the accuracy of predictions.

## Run Locally
- Make sure Python 3 is installed. Reference to install: [Download and Install Python 3](https://www.python.org/downloads/)
- Clone the project: `git clone https://github.com/setu-parekh/MNIST-Digit-Classification.git`
- Route to the cloned project: `cd MNIST-Digit-Classification`
- Install necessary packages: `pip install -r requirements.txt`
- Run Jupyter Notebook: `jupyter notebook`
- Select the notebook to open: `naive_bayes_classifier.ipynb`

## Conclusion
- Naive Bayes Classification Model is able to classify digit 0 with 91.4 % accuracy and digit 1 with 92.4 % accuracy.




