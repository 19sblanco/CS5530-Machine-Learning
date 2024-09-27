import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import random

# random.seed(1)
data_type = "tfidf"
data_paths = [
    "data/splits/" + data_type + "/split0.csv",
    "data/splits/" + data_type + "/split1.csv",
    "data/splits/" + data_type + "/split2.csv",
    "data/splits/" + data_type + "/split3.csv",
    "data/splits/" + data_type + "/split4.csv",
]

def find_hyper_parameter_4(p=True):
    learning_rates = [1, 0.1, 0.01]
    best_learning_rate = None
    best_performace = None
    for rate in learning_rates:
        average_performace = cross_validation(rate)
        if best_performace is None or average_performace > best_performace:
            best_performace = average_performace
            best_learning_rate = rate

    if p:
        print("\tBest learning rate:", best_learning_rate)
        print("\tBest performance:", best_performace)
    return best_learning_rate


def cross_validation(rate):
    performaces = []
    for i in range(len(data_paths)):
        train_data, dev_data = split_data(i)
        w, b =  perceptron_average(train_data, rate)
        error = validate_data(w, b, dev_data)
        performaces.append(error)

    return np.mean(performaces)


def validate_data(w, b, dev_data):
    labeled_data = format_data(dev_data)
    correct = 0
    for y_i, x_i in labeled_data:
        y_hat = np.dot(w, x_i) + b
        if y_hat < 0: guess = -1
        else: guess = 1

        if guess == y_i:
            correct += 1

    return correct / len(labeled_data)


def split_data(fold_num, all_data=False):
    train_data = None
    dev_data = None
    if not all_data:
        for i in range(len(data_paths)):
            if i == fold_num:
                dev_data = pd.read_csv(data_paths[i])
            else:
                if train_data is None:
                    train_data = pd.read_csv(data_paths[i])
                else:
                    train_data = pd.concat([train_data, pd.read_csv(data_paths[i])])
        return train_data, dev_data
    else:
        for i in range(len(data_paths)):
            if train_data is None:
                train_data = pd.read_csv(data_paths[i])
            else:
                train_data = pd.concat([train_data, pd.read_csv(data_paths[i])])
        return train_data


def read_data_from_csv(file_path):
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        data = [row for row in reader]
    return data


def format_data(dataFrame):
    labels = np.array(list(dataFrame['label'].iloc))
    data_points = dataFrame.drop('label', axis=1).values
    labeled_data = list(zip(labels, data_points))
    return labeled_data


def perceptron_average(train_data, rate, epochs=10, dev_data=None):
    labeled_data  = format_data(train_data)
    w, b = initilize_w(labeled_data[0][1])
    w_cum, b_cum = w, b
    evals = []
    for epoch in range(epochs):
        random.shuffle(labeled_data)
        for example in labeled_data:
            y_i = example[0]
            x_i = example[1]
            if y_i * np.dot(w, x_i) + b <= 0:
                w = w + rate * y_i * x_i
                b = b + rate * y_i

            w_cum += w
            b_cum += b
        if dev_data is not None:
            evals.append((validate_data(w_cum, b_cum, dev_data), w, b))

    if dev_data is None:
        return w_cum, b_cum
    else:
        return w_cum, b_cum, evals


def initilize_w(sample_point):
    smallest = -0.01
    largest = 0.01
    size_of_arary = sample_point.shape
    w = np.random.uniform(smallest, largest, size_of_arary)
    b = random.uniform(smallest, largest)
    return w, b


def test():
    n_epochs = 20
    # learning_rate = find_hyper_parameter_4()
    learning_rate = 1

    # train_data = pd.read_csv("drive/MyDrive/splits/" + data_type + "/split3.csv")
    # test_data = pd.read_csv("drive/MyDrive/splits/" + data_type + "/split4.csv")

    train_data = pd.read_csv("data/" + data_type + "/" + data_type + ".train.csv")
    test_data = pd.read_csv("data/" + data_type + "/" + data_type + ".test.csv")


    w, b = perceptron_average(train_data, learning_rate, epochs=n_epochs)

    test_accuracy = validate_data(w, b, test_data)
    print("test:", test_accuracy)


def submission():
    n_epochs = 20
    # learning_rate = find_hyper_parameter_4()
    learning_rate = 1

    train_path = "data/" + data_type + "/" + data_type + ".train.csv"
    test_path = "data/" + data_type + "/" + data_type + ".eval.anon.csv"

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    labeled_data = format_data(test_data)

    w, b = perceptron_average(train_data, learning_rate, epochs=n_epochs)

    with open("data/submissions/perceptron0.csv", "w") as submission_file:
        submission_file.write("example_id,label\n")
        i = 0
        for y_i, x_i in labeled_data:
            y_hat = np.dot(w, x_i) + b
            # if y_hat < 0: guess = -1
            # else: guess = 1
            if y_hat < 0: guess = 1
            else: guess = -1

            submission_file.write(str(i) + "," + str(guess) + "\n")
            i += 1




# test()
submission()