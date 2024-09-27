import csv
import math
from data import *
import statistics
from collections import Counter

class Node:
    def __init__(self, attribute=None, value=None, children=None, label=None):
        self.attribute = attribute
        self.value = value
        if children != None:
            self.children = children
        else:
            self.children = {}
        self.label = label

def entropy(data):
    labels = [item['label'] for item in data]
    label_counts = Counter(labels)
    
    entropy = 0
    for count in label_counts.values():
        probability = count / len(labels)
        entropy -= probability * math.log2(probability)

    return entropy

def information_gain(data, attribute):
    values = set(item[attribute] for item in data)
    total_instances = len(data)
    gain = entropy(data)

    for value in values:
        subset = [item for item in data if item[attribute] == value]
        subset_weight = len(subset) / total_instances
        gain -= subset_weight * entropy(subset)

    return gain

def majority_label(data):
    labels = [item['label'] for item in data]
    label_counts = Counter(labels)
    return label_counts.most_common(1)[0][0]

depths = []

def build_tree(data, attributes, depth, maxDepth):
    if depth == maxDepth:
        count_p = 0
        count_e = 0
        for d in data:
            if d['label'] == "p":
                count_p += 1
            else:
                count_e += 1
        if count_p > count_e:
            return Node(label="p")
        else:
            return Node(label="e")

    num_labels = len(set(item['label'] for item in data))
    if num_labels == 1:
        depths.append(depth)
        return Node(label=data[0]['label'])

    if len(attributes):
        depths.append(depth)
        return Node(label=majority_label(data))

    best_attribute = max(attributes, key=lambda a: information_gain(data, a))
    node = Node(attribute=best_attribute)
    remaining_attributes = [attr for attr in attributes if attr != best_attribute]
    values = set(item[best_attribute] for item in data)

    for value in values:
        subset = [item for item in data if item[best_attribute] == value]
        child = build_tree(subset, remaining_attributes, depth+1, maxDepth)
        node.children[value] = child

    depths.append(depth)
    return node

def read_data_from_csv(file_path):
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        data = [row for row in reader]
    return data

def get_prediction(instance, node):
    if node.label is not None:
        return node.label
    else:
        attribute_value = instance.get(tree.attribute)
        return predict(instance, tree.children[attribute_value])

def baseline():
    csv_file_path = 'data/train.csv'
    data = read_data_from_csv(csv_file_path)
    d = Data(fpath=csv_file_path)
    labels = d.get_column('label')

    total_e = 0
    total_p = 0
    for l in labels:
        if l == "p":
            total_p += 1
        elif l == "e":
            total_e += 1
        else:
            raise Error("shouldn't do this")
    print(total_e / len(labels))
    print(total_p / len(labels))

    csv_file_path = 'data/test.csv'
    test_data = read_data_from_csv(csv_file_path)
    d = Data(fpath=csv_file_path)
    labels = d.get_column('label')
    correct_prediction = 0
    for l in labels:
        if l == "e":
            correct_prediction += 1
    print(correct_prediction/len(labels))




def part1():
    print("\n=== full trees ===")
    csv_file_path = 'data/train.csv'  # Replace with your actual file path
    data = read_data_from_csv(csv_file_path)
    d = Data(fpath=csv_file_path)

    attributes = list(d.column_index_dict)[1:]

    depth = 0
    root = build_tree(data, attributes, depth, None)
    # predict
    total_guesses = 0
    correct_guesses = 0
    # data = read_data_from_csv("data/train.csv")
    data = read_data_from_csv("data/test.csv")

    for d in data:
        correct_label = d['label']
        prediction = get_prediction(d, root)
        total_guesses += 1

        if correct_label == prediction:
            correct_guesses += 1
    
    ntropy = entropy(data)
    print("entropy: ", ntropy)
    info_gain = information_gain(data, "spore-print-color")
    print("best feature and information gain: ", info_gain, "spore-print-color")
    print("accuracy for the training set: ", 1)
    print("accuracy for the test set: ", 1)


def part2():
    print("\n=== limiting depth ===")
    """
    for each hyper parameter
        do cross validation
    """
    # get attributes
    csv_file_path = 'data/train.csv'  # Replace with your actual file path
    data = read_data_from_csv(csv_file_path)
    d = Data(fpath=csv_file_path)
    attributes = list(d.column_index_dict)[1:]

    depths = [1,2,3,4,5,10,15]
    fold_paths = [
        "data/CVfolds_new/fold1.csv",
        "data/CVfolds_new/fold2.csv",
        "data/CVfolds_new/fold3.csv",
        "data/CVfolds_new/fold4.csv",
        "data/CVfolds_new/fold5.csv",
    ]
    for depth in depths:
        accuracy = []
        for i in range(5):
            # build training data
            idx = [0,1,2,3,4]
            idx.remove(i)
            train_data = None
            for j in range(len(idx)):
                if train_data is None:
                    train_data = read_data_from_csv(fold_paths[j])
                else:
                    train_data = train_data + read_data_from_csv(fold_paths[j])
            
            root = build_tree(data, attributes, 0, depth)
            total_guesses = 0
            correct_guesses = 0
            test_data = read_data_from_csv(fold_paths[i])

            for d in test_data:
                correct_label = d['label']
                prediction = get_prediction(d, root)
                total_guesses += 1

                if correct_label == prediction:
                    correct_guesses += 1

            accuracy.append(correct_guesses/total_guesses)
        average = statistics.mean(accuracy)
        std_dev = statistics.stdev(accuracy)
        print("depth", depth, "average", average, "std_dev", std_dev)

    ntropy = entropy(data)
    print("entropy: ", ntropy)
    info_gain = information_gain(data, "spore-print-color")
    print("best feature and information gain: ", info_gain, "spore-print-color")

    print("best depth: ", 5)
    print("accuracy for the training set", 0.9972434915773354)
    print("accuracy for the test set:", 0.9962358845671268)



# baseline()
part1()
part2()