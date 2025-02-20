import numpy as np
import csv

def euclidean_distance(row1, row2):
    row1 = np.array(row1, dtype=float)
    row2 = np.array(row2, dtype=float)
    return np.sqrt(np.sum((row1 - row2) ** 2))

def read_data(filename):
    data = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            # Check if the last element is a class label (yes or no)
            if row[-1] in ['yes', 'no']:
                data.append([float(item) for item in row[:-1]] + [row[-1]])
            else:
                data.append([float(item) for item in row])
    return data

def get_neighbors(training_data, test_instance, k):
    distances = []
    for train_row in training_data:
        dist = euclidean_distance(test_instance, train_row[:-1])  # Ignore the class label
        distances.append((train_row, dist))
    distances.sort(key=lambda x: x[1])
    return [dist[0] for dist in distances[:k]]

def predict_classification(neighbors):
    votes = {}
    for neighbor in neighbors:
        response = neighbor[-1]  # The class label is the last element in the neighbor array
        votes[response] = votes.get(response, 0) + 1
    
    # Sort votes by count and predict 'yes' in case of a tie
    sorted_votes = sorted(votes.items(), key=lambda item: item[1], reverse=True)
    if len(sorted_votes) > 1 and sorted_votes[0][1] == sorted_votes[1][1]:
        return 'yes'
    return sorted_votes[0][0]

def classify_nn(training_filename, testing_filename, k):
    training_data = read_data(training_filename)
    testing_data = read_data(testing_filename)
    predictions = []
    for test_instance in testing_data:
        neighbors = get_neighbors(training_data, test_instance, k)
        prediction = predict_classification(neighbors)
        predictions.append(prediction)
    return predictions

