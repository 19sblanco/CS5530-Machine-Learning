
from random import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import *

# set random seeds for repeatable experiements
# seed(1)
# np.random.seed(1)


"""
prepare data

impliment SVM
    graph the change and let that inform your threshold
    test with some small vectors on paper

impliment logistic regression
    test with some small vectors

impliment cross validation
"""

def prepare_data(df):
    data_points = df.iloc[:, 1:].to_numpy()
    labels = df.iloc[:, 0].to_numpy()
    return list(zip(data_points, labels))

def precision(TP, FP):
    if TP + FP == 0:
        return 0
    return TP / (TP + FP)

def recall(TP, FN):
    if TP + FN == 0:
        return 0
    return TP / (TP + FN)

def F1(TP, FP, FN):
    p = precision(TP, FP)
    r = recall(TP, FN)
    if p+r == 0:
      return 0
    return 2 * ((p*r) / (p+r))

def get_change(old, new):
    diff = old - new
    norm = np.linalg.norm(diff)
    return norm

def get_data_for_fold(i):
    train_data, dev_data = split_data(i)
    train_data = prepare_data(train_data)
    dev_data = prepare_data(dev_data)
    return train_data, dev_data

"""
Evaluate the model (w, b) on the val_data data set using the F1 metric
parameters:
    w: weight vector of model
    b: bias of model
    val_data: data set which model will be evaluated on
"""
def test(w, b, val_data):
    TP = 0
    FP = 0
    FN = 0
    for x_i, y_i in val_data:
        y_pred = np.dot(w, x_i) + b
        if y_pred > 0 and y_i == 1:
            TP += 1
        elif y_pred > 0 and y_i == -1:
            FP += 1
        elif y_pred < 0 and y_i == 1:
            FN += 1

    F1_result = F1(TP, FP, FN)
    precision_result = precision(TP, FP)
    recall_result = recall(TP, FN)
    return F1_result, precision_result, recall_result


"""
implementation of the stochastic sub-gradient decent for SVM

parameters:
    s = training set {(x_i, y_i)}
    gamma: initial learning rate
    C: regularization / tradeoff parameter
"""
def svm(s, gamma_0, c, epoch=10, plot=False):
    losses = []
    d = len(s[0][0])
    w = np.zeros(d)
    b = 0
    for t in range(epoch):
        shuffle(s)
        gamma_t = gamma_0 / (1 + t)
        for i in range(len(s)):
            x_i = s[i][0]
            y_i = s[i][1]
            if y_i * (np.dot(w, x_i) + b) <= 1:
                w = (1-gamma_t) * w + gamma_t * c * y_i * x_i
                b = (1-gamma_t) * b + gamma_t * c * y_i
            else:
                w = (1-gamma_t) * w
                b = (1-gamma_t) * b

        if plot == True:
            avg_loss = 0
            for x_i, y_i in s:
                loss = 0
                result = y_i * (np.dot(w, x_i) + b)
                if result <= 1:
                    loss = result
                avg_loss += loss
            avg_loss = avg_loss / len(s)
            losses.append(avg_loss)
    if plot:
          x_points = [i for i in range(len(losses))]
          plt.plot(x_points, losses, color="blue")
          plt.xlabel("epoch")
          plt.ylabel("loss")
          plt.show()


    return w, b


"""
calculate the gradient of the logistic regression
function with normalization given in the hw
    parameters:
        y_i - true label
        x_i - input data point
        w - weight vector
"""
def logistic_gradient(y_i, x_i, w, b):
    inside = -y_i * np.dot(w, x_i)
    p1 = 1 / (1 + exp(inside))
    p2 = exp(inside)
    p3 = -y_i * x_i
    grad_w = p1 * p2 * p3

    inside_b = -y_i * b
    b1 = 1 / (1+exp(inside))
    b2 = exp(inside)
    b3 = -y_i
    grad_b = b1 * b2 * b3

    return grad_w, grad_b



"""
parameters:
    initial_lr: initial learning rate
    C: regularization / tradeoff parameter
"""
def logistic_regression(s, gamma_0, c, epoch=10, plot=False):
    losses = []
    d = len(s[0][0])
    w = np.zeros(d)
    b = 0
    for t in range(epoch):
        shuffle(s)
        gamma_t = gamma_0 / (1 + t)
        for x_i, y_i in s:
            grad_w, grad_b = logistic_gradient(y_i, x_i, w, b)
            w = w - gamma_t * grad_w
            b = b - gamma_t * grad_b

        if plot == True:
            avg_loss = []
            for x_i, y_i in s:
                loss = 0

                result = 1 / (1 + exp(-np.dot(w, x_i) + b))
                # result = np.dot(w, x_i) + b

                if result >= .5:
                    if y_i == 0: # misclassification
                      loss = result - .5
                else:
                    if y_i == 1:
                      loss = 5. - result
                avg_loss.append(loss)

            avg = sum(avg_loss) / len(avg_loss)
            losses.append(avg_loss)
    if plot:
          x_points = [i for i in range(len(losses))]
          plt.plot(x_points, losses, color="blue")
          plt.xlabel("epoch")
          plt.ylabel("loss")
          plt.show()


    return w, b


"""
parameters:
    fold_num: the index of the fold for the dev_data, the rest as the training set
    all_data: mark as true to collect all data into one training set
"""
def split_data(fold_num, data_set="bag-of-words", all_data=False, test_data=False):
    if data_set == "bag-of-words":
        data_paths = [
            "data/splits/bag-of-words/split0.csv",
            "data/splits/bag-of-words/split1.csv",
            "data/splits/bag-of-words/split2.csv",
            "data/splits/bag-of-words/split3.csv",
            "data/splits/bag-of-words/split4.csv",
        ]
    elif data_set == "glove":
        data_paths = [
            "data/splits/glove/split0.csv",
            "data/splits/glove/split1.csv",
            "data/splits/glove/split2.csv",
            "data/splits/glove/split3.csv",
            "data/splits/glove/split4.csv",
        ]
    elif data_set == "tfidf":
        data_paths = [
            "data/splits/tfidf/split0.csv",
            "data/splits/tfidf/split1.csv",
            "data/splits/tfidf/split2.csv",
            "data/splits/tfidf/split3.csv",
            "data/splits/tfidf/split4.csv",
        ]
    else:
        print("error: split_data, param data_set")
        exit()

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
        if not test_data:
            for i in range(len(data_paths)):
                if train_data is None:
                    train_data = pd.read_csv(data_paths[i])
                else:
                    train_data = pd.concat([train_data, pd.read_csv(data_paths[i])])
            return prepare_data(train_data)
        else:
            if data_set == "bag-of-words":
                test_path = "data/bag-of-words/bow.test.csv"
            elif data_set == "glove":
                test_path = "data/glove/glove.test.csv"
            elif data_set == "tfidf":
                test_path = "data/tfidf/tfidf.test.csv"

            test_data = pd.read_csv(test_path)
            return prepare_data(test_data)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def get_split_train_data(data_set):
    if data_set == "bag-of-words":
      data_paths = [
          "data/splits/bag-of-words/split0.csv",
          "data/splits/bag-of-words/split1.csv",
      ]
    elif data_set == "glove":
        data_paths = [
            "data/splits/glove/split0.csv",
            "data/splits/glove/split1.csv",
        ]
    elif data_set == "tfidf":
        data_paths = [
            "data/splits/tfidf/split0.csv",
            "data/splits/tfidf/split1.csv",
        ]
    train_data = pd.read_csv(data_paths[0])
    test_data = pd.read_csv(data_paths[1])
    return prepare_data(train_data), prepare_data(test_data)



"""
for a given algorithm find the hyper parameters for initial learning
rate and Cs (tradeoff parameter)
parameters:
    algorithm: algorithm to perform cross validation on
    initial_learning_rates: a list of initial learning rates
    Cs: a list of tradeoff parameters
"""
def cross_validation(algorithm, initial_learning_rates, Cs, epochs, data):
    best_avg_f1_result = None
    best_avg_precision_result = None
    best_avg_recall_result = None

    best_initial_learning_rate = None
    best_c = None
    best_epoch = None

    train_data, dev_data = get_split_train_data(data)
    for rate in initial_learning_rates:
        for c in Cs:
            for epoch in epochs:
                print(rate, c, epoch)
                F1_results = []
                precision_results = []
                recall_results = []
                for i in range(5):

                    w, b = algorithm(train_data, rate, c, epoch=epoch)

                    F1_result, precision_result, recall_result = test(w, b, dev_data)
                    F1_results.append(F1_result)
                    precision_results.append(precision_result)
                    recall_results.append(recall_result)


                avg_F1_result = sum(F1_results) / len(F1_results)
                if best_avg_f1_result == None or avg_F1_result > best_avg_f1_result:
                    best_avg_f1_result = avg_F1_result
                    best_avg_precision_result = sum(precision_results) / len(precision_results)
                    best_avg_recall_result = sum(recall_results) / len(recall_results)
                    best_initial_learning_rate = rate
                    best_c = c
                    best_epoch = epoch

    print("=== Best ===")
    print(best_avg_f1_result)
    print(best_avg_precision_result)
    print(best_avg_recall_result)
    print(best_initial_learning_rate)
    print(best_c)
    print(best_epoch)


def experiment_logistic_regression_project():
  initial_learning_rates = [10**0, 10**-1, 10**-2, 10**-3, 10**-4, 10**-5]
  Cs = [10**-1, 10**0, 10**1, 10**2, 10**3, 10**4]
  epochs = [10, 25, 50]
  data_set = "tfidf"
  cross_validation(svm, initial_learning_rates, Cs, epochs, data_set)
"""
=== Best ===
1.0
1.0
1.0
1
0.1
10
"""

def run_logistic_regression(attempt):
    rate = 1
    c = .1
    epoch = 10

    train_path = "data/tfidf/tfidf.train.csv"
    t_data = pd.read_csv(train_path)
    train_data = prepare_data(t_data)

    e_path = "data/tfidf/tfidf.eval.anon.csv"
    e_data = pd.read_csv(e_path)
    eval_data = prepare_data(e_data)

    w, b = logistic_regression(train_data, c, epoch)

    with open("data/submissions/LG_" + str(attempt) + ".csv", "w") as submission_file:
        submission_file.write("example_id,label\n")
        i = 0
        for x_i, y_i in eval_data:
          y_hat = np.dot(w, x_i) + b
          y_hat = sigmoid(y_hat)
          if y_hat < .5: guess = -1
          else: guess = 1

          submission_file.write(str(i) + "," + str(guess) + "\n")
          i += 1




# experiment_logistic_regression_project()
attempt = 0
run_logistic_regression(attempt)


