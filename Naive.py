import numpy as np
import csv
from math import sqrt, pi, exp

def read_data(filename):
    data = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if row[-1] in ['yes', 'no']:
                data.append([float(item) for item in row[:-1]] + [row[-1]])
            else:
                data.append([float(item) for item in row])
    return data

def separate_by_class(data):
    separated = {}
    for i in range(len(data)):
        vector = data[i]
        class_value = vector[-1]
        if (class_value not in separated):
            separated[class_value] = []
        separated[class_value].append(vector[:-1])
    return separated

def summarize_dataset(dataset):
    summaries = [(np.mean(column), np.std(column), len(column)) for column in zip(*dataset)]
    return summaries

def summarize_by_class(data):
    separated = separate_by_class(data)
    summaries = {}
    for class_value, rows in separated.items():
        summaries[class_value] = summarize_dataset(rows)
    return summaries

def calculate_probability(x, mean, stdev):
    if stdev == 0:
        stdev = 0.0001  # To prevent division by zero
    exponent = exp(-((x-mean)**2 / (2 * stdev**2)))
    return (1 / (sqrt(2 * pi) * stdev)) * exponent

def calculate_class_probabilities(summaries, row):
    total_rows = sum([summaries[label][0][2] for label in summaries])
    probabilities = {}
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = summaries[class_value][0][2] / float(total_rows)
        for i in range(len(class_summaries)):
            mean, stdev, count = class_summaries[i]
            probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
    return probabilities

def classify_nb(training_filename, testing_filename):
    training_data = read_data(training_filename)
    testing_data = read_data(testing_filename)
    summaries = summarize_by_class(training_data)
    predictions = []
    for row in testing_data:
        probabilities = calculate_class_probabilities(summaries, row)
        best_label, best_prob = None, -1
        for class_value, probability in probabilities.items():
            if best_label is None or probability > best_prob:
                best_label, best_prob = class_value, probability
            elif probability == best_prob:
                best_label = 'yes'  # Tie breaking
        predictions.append(best_label)
    return predictions
