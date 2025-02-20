import numpy as np
import csv
from sklearn.model_selection import KFold

def euclidean_distance(row1, row2):
    r1 = np.array(row1, dtype=float)
    r2 = np.array(row2, dtype=float)
    return np.sqrt(np.sum((r1 - r2) ** 2))

def read_data(filename):
    data = []
    with open(filename, 'r') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            if row[-1] in ['yes', 'no']:
                data.append([float(item) for item in row[:-1]] + [row[-1]])
            else:
                data.append([float(item) for item in row])
    return data

    return data

def get_neighbors(training_data, test_instance, k):
    distances = []
    for t_row in training_data:
        dist = euclidean_distance(test_instance, t_row[:-1])
        distances.append((t_row, dist))
    distances.sort(key=lambda x: x[1])
    return [dist[0] for dist in distances[:k]]

def predict_classification(neighbors):
    votes = {}
    for n in neighbors:
        response = n[-1]
        votes[response] = votes.get(response, 0) + 1
    sorted_votes = sorted(votes.items(), key=lambda item: item[1], reverse=True)
    return sorted_votes[0][0]

def k_fold_cross_validation(data, k, num_folds=10):
    kf = KFold(n_splits=num_folds)
    accuracies = []

    for train_index, test_index in kf.split(data):
        train_set = [data[i] for i in train_index]
        test_set = [data[i] for i in test_index]
        predictions = []

        for test_instance in test_set:
            neighbors = get_neighbors(train_set, test_instance[:-1], k)
            prediction = predict_classification(neighbors)
            predictions.append(prediction)

        correct = sum(1 for x, y in zip(predictions, test_set) if x == y[-1])
        accuracy = correct / len(test_set)
        accuracies.append(accuracy)

    return np.mean(accuracies)

# Load data
filename = 'pima.csv'  # Change to 'occupancy.csv' for the other dataset
data = read_data(filename)

# Perform 10-fold cross-validation for k=1 and k=5
accuracy_k1 = k_fold_cross_validation(data, k=1)
accuracy_k5 = k_fold_cross_validation(data, k=5)

print("10-fold Cross-validation Accuracy for k=1:", accuracy_k1)
print("10-fold Cross-validation Accuracy for k=5:", accuracy_k5)



#Naive Bayes

import numpy as np
import csv
from math import sqrt, pi, exp
from sklearn.model_selection import KFold

# Read CSV file
def read_data(filename):
    data = []
    with open(filename, 'r') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)  # Skip the header if necessary
        for row in csv_reader:
            if row[-1] in ['yes', 'no']:
                data.append([float(item) for item in row[:-1]] + [row[-1]])
            else:
                data.append([float(item) for item in row])
    return data

# Separate data by class
def separate_by_class(data):
    separated = {}
    for vec in data:
        class_val = vec[-1]
        if class_val not in separated:
            separated[class_val] = []
        separated[class_val].append(vec[:-1])
    return separated

# Summarize dataset by mean, standard deviation, and count
def summarize_dataset(dataset):
    summaries = [(np.mean(col), np.std(col), len(col)) for col in zip(*dataset)]
    return summaries

# Summarize the dataset by class
def summarize_by_class(data):
    separated = separate_by_class(data)
    summaries = {}
    for class_val, rows in separated.items():
        summaries[class_val] = summarize_dataset(rows)
    return summaries

# Calculate Gaussian probability
def calculate_probability(x, mean, stdev):
    if stdev == 0:
        stdev = 0.0001  # To prevent division by zero
    exponent = exp(-((x-mean)**2 / (2 * stdev**2)))
    return (1 / (sqrt(2 * pi) * stdev)) * exponent

# Calculate probabilities for each class
def calculate_class_probabilities(summaries, row):
    total_rows = sum([summaries[label][0][2] for label in summaries])
    probabilities = {}
    for class_val, class_summaries in summaries.items():
        probabilities[class_val] = summaries[class_val][0][2] / float(total_rows)
        for i in range(len(class_summaries)):
            mean, stdev, count = class_summaries[i]
            probabilities[class_val] *= calculate_probability(row[i], mean, stdev)
    return probabilities

def k_fold_cross_validation(data, num_folds=10):
    kf = KFold(n_splits=num_folds)
    accuracies = []
    for train_index, test_index in kf.split(data):
        train_set = [data[i] for i in train_index]
        test_set = [data[i] for i in test_index]
        summaries = summarize_by_class(train_set)
        predictions = []
        for row in test_set:
            probabilities = calculate_class_probabilities(summaries, row[:-1])
            best_label = max(probabilities, key=probabilities.get)
            predictions.append(best_label == row[-1])
        accuracy = sum(predictions) / len(predictions)
        accuracies.append(accuracy)
    return np.mean(accuracies)


filename = 'pima.csv'  
data = read_data(filename)


accuracy = k_fold_cross_validation(data)
print("Naive Bayes:", accuracy)
