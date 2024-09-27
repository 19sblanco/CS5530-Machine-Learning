import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import random

random.seed(1)
folds = 5

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
    return best_learning_rate, None


def cross_validation(rate):
    performaces = []
    for i in range(folds):
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
    data_paths = [
        "hw2_data/CVSplits/train0.csv",
        "hw2_data/CVSplits/train1.csv",
        "hw2_data/CVSplits/train2.csv",
        "hw2_data/CVSplits/train3.csv",
        "hw2_data/CVSplits/train4.csv",
    ]
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
    global perceptron_average_updates
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


def main():
    n_epochs = 20
    learning_rate, margin = find_hyper_parameter_4(p=False)
    train_data = split_data(None, all_data=True)
    dev_data = pd.read_csv("hw2_data/diabetes.dev.csv")
    test_data = pd.read_csv("hw2_data/diabetes.test.csv")
    w, b, evals = perceptron_average(train_data, learning_rate, epochs=n_epochs, dev_data=dev_data)

    dev_accuracy_list = [item[0] for item in evals]
    test_accuracy = validate_data(w, b, test_data)
    print("dev accuracys:", dev_accuracy_list)
    print("test:", test_accuracy)

    # plot
    x = list(range(n_epochs))

    plt.plot(x, dev_accuracy_list)
    plt.xlabel("epoch number")
    plt.ylabel("accuracy")
    plt.title(perceptron_average)
    plt.show()


    


            


def majority_base_line():
    # find most common label and its count in training set
    # find number of examples with that label for dev and test set
    training_data = split_data(None, all_data=True)
    training_labels = training_data["label"].tolist()

    # find most common label
    count_1 = 0
    count_neg1 = 0
    for label in training_labels:
        if label == 1: count_1 += 1
        else: count_neg1 += 1
    if count_1 > count_neg1:
        most_common_label = 1
    else:
        most_common_label = -1

    # eval on dev and test set
    dev_data = pd.read_csv("hw2_data/diabetes.dev.csv")
    dev_labels = dev_data["label"].tolist()
    
    dev_label_count = dev_labels.count(most_common_label)
    print("accuracy on dev", dev_label_count / len(dev_data))

    test_data = pd.read_csv("hw2_data/diabetes.test.csv")
    test_labels = dev_data["label"].tolist()
    
    test_label_count = test_labels.count(most_common_label)
    print("accuracy on test", test_label_count / len(test_data))
    




main()
# majority_base_line()