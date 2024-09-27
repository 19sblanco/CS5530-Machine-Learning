from lib import * 


"""
for a given algorithm find the hyper parameters for initial learning
rate and Cs (tradeoff parameter)
parameters:
    algorithm: algorithm to perform cross validation on
    initial_learning_rates: a list of initial learning rates
    Cs: a list of tradeoff parameters
"""
def cross_validation(algorithm, initial_learning_rates, Cs):
    best_avg_f1_result = None
    best_avg_precision_result = None
    best_avg_recall_result = None

    best_initial_learning_rate = None
    best_c = None

    for rate in initial_learning_rates:
        for c in Cs:
            F1_results = []
            precision_results = []
            recall_results = []
            for i in range(5):
                train_data, dev_data = get_data_for_fold(i)
                w, b = algorithm(train_data, rate, c)

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

    print("=== Best ===")
    print(best_avg_f1_result)
    print(best_avg_precision_result)
    print(best_avg_recall_result)
    print(best_initial_learning_rate)
    print(best_c)

    """
=== Best ===
0.2208497684631568
0.6416422965785478
0.143760742249118
0.1
10
    """


def experiment_svm():
    initial_learning_rates = [10**0, 10**-1, 10**-2, 10**-3, 10**-4]
    Cs = [10**1, 10**0, 10**-1, 10**-2, 10**-3, 10**-4]
    cross_validation(svm, initial_learning_rates, Cs)

def run_svm():
    C = 10
    learning_rate = .1
    train_data = split_data(0, all_data=True)
    test_data = split_data(0, all_data=True, test_data=True)
    epoch = 30

    w, b = svm(train_data, learning_rate, C, epoch=epoch, plot=True)
    f1, precision, recall = test(w, b, test_data)

    print(f1)
    print(precision)
    print(recall)
    """
0.44469783352337516
0.38562953197099537
0.5251346499102334
    """

def experiment_logistic_regression():
  initial_learning_rates = [10**0, 10**-1, 10**-2, 10**-3, 10**-4, 10**-5]
  Cs = [10**-1, 10**0, 10**1, 10**2, 10**3, 10**4]

  cross_validation(logistic_regression, initial_learning_rates, Cs)
  """
=== Best ===
0.4057440960363877
0.6888358482460057
0.287648501856026
0.01 lr
1000 c
  """
def run_logistic_regression():
  learning_rate = .01
  c = 1000
  train_data = split_data(0, all_data=True)
  test_data = split_data(0, all_data=True, test_data=True)
  epoch = 50

  w, b = logistic_regression(train_data, learning_rate, c, epoch=epoch, plot=True)
  f1, precision, recall = test(w, b, test_data)

  print(f1)
  print(precision)
  print(recall)





# experiment_svm()
print("=== SVM ===")
run_svm()
# experiment_logistic_regression()
print("=== logistic regression ===")
run_logistic_regression()
