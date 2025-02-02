import numpy as np
import time 
import pandas as pd
from sklearn.preprocessing import StandardScaler


import os
import csv

def save_results_to_csv(filename, dataset_name, accuracy, time_taken, method):
    file_exists = os.path.isfile(filename)
    
    # Open the file in append mode
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write the header only if the file does not exist
        if not file_exists:
            writer.writerow(["Dataset Name", f"{method} Accuracy", f"{method} Time"])

        # Write the results
        writer.writerow([dataset_name, accuracy, time_taken])


class LogisticRegression:
    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        time_s = time.time()
        for epoch in range(self.n_iters):
            # approximate y with linear combination of weights and x, plus bias
            linear_model = np.dot(X, self.weights) + self.bias
            # apply sigmoid function
            y_predicted = self._sigmoid(linear_model)

            loss = self._binary_cross_entropy(y, y_predicted)
            # if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{self.n_iters}, Loss: {loss:.4f}")

            # compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        
        return time.time() - time_s
    

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def _binary_cross_entropy(self, y_true, y_pred):
        # Binary cross-entropy loss
        n_samples = len(y_true)
        return -1 / n_samples * np.sum(
            y_true * np.log(y_pred + 1e-15) + (1 - y_true) * np.log(1 - y_pred + 1e-15)
        )

one_zero_dataset = ['A5a.csv', 'A6a.csv', 'A7a.csv', 'A8a.csv', '2dplanes.csv', 'A9a.csv', 'adult.csv', 'W8a.csv']
mixed_dataset = ['ijcnn.csv', 'covtype.csv']
continues_dataset = ['magic_gamma_telescope.csv', 'fried.csv', 'Run_or_walk_information.csv', 'hepmass.csv', 'susy.csv']

all_datasets = one_zero_dataset + mixed_dataset + continues_dataset

# Testing
if __name__ == "__main__":
    # Imports
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    # bc = datasets.load_breast_cancer()
    # X, y = bc.data, bc.target

    for dataset_name in all_datasets:
        dataset_path = f"D:\\datasts\\large datasets\\binary\\{dataset_name}"
        df = pd.read_csv(dataset_path, header=None)

        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        # Apply StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=1234
        )

        regressor = LogisticRegression(learning_rate=0.1, n_iters=10)

        execution_time = regressor.fit(X_train, y_train)

        print(f"Execution time of fit function of GD: {execution_time:.4f} seconds")
        
        predictions = regressor.predict(X_test)
        acc = accuracy(y_test, predictions)
        print("LR classification accuracy:", acc)


        # Save results to CSV
        save_results_to_csv(
            "results_GD2.csv", 
            dataset_name,	 
            acc, 
            execution_time, 
            "GD"
        )