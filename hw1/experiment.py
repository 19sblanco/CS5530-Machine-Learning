from data import *
from copy import *
from math import *

def baseline():
    train_total_lines = 0
    train_p_label = 0
    train_e_label = 0
    with open("data/data/train.csv", "r") as file:
        for i, line in enumerate(file):
            if i == 0: continue
            train_total_lines += 1
            if line[0] == "p":
                train_p_label += 1
            elif line[0] == "e":
                train_e_label += 1
            else:
                print("somehting is wrong")
                exit()
    print(train_total_lines )
    print("P train amount: ", train_p_label, train_p_label/train_total_lines )
    print("E train amount: ", train_e_label, train_e_label/train_total_lines )
    
    test_total_lines = 0
    test_p_label = 0
    test_e_label = 0
    with open("data/data/test.csv", "r") as file:
        for i, line in enumerate(file):
            if i == 0: continue
            test_total_lines += 1
            if line[0] == "p":
                test_p_label += 1
            elif line[0] == "e":
                test_e_label += 1
            else:
                print("somehting is wrong")
                exit()
    print(test_total_lines )
    print("P test amount: ", test_p_label, test_p_label/test_total_lines )
    print("E test amount: ", test_e_label, test_e_label/test_total_lines )


def full_trees():
    data = Data(fpath="data/data/train.csv")
    attributes = list(data.column_index_dict.keys())[1:]
    maxDepth = None
    currDepth = 0
    maxDepth = None
    most_common_label = None

    n = id3(data, attributes, currDepth, maxDepth, most_common_label)


class Node():
    Attribute = None
    children = None 
    label = None

    def __init__(self):
        self.children = {}
    
    def add_child(self, value, node):
        self.children[value] = node


def id3(data, attrs, currDepth, maxDepth, most_common_label):
    """
    TODO: before you code ensure that this works logically
    note: 
        * data is a rowwise subset of the data
        * col is a set of attributes that are up for consideration for data


    if depth exceeded:
        return most common label of data
    if all examples in data have the same label:
        return a node with that as the label (name attribute?)

    otherwise:
        A = best attribute for the rowwise subset of the data called "data"
        root node named A
        for each value v that A can take:
            sv = {value: subset of data where A=v}
            append to root node
            if sv is empty:
                create leaf node with most common label in the whole dataset
            else:
                new_attr = attrs - {A}
                id3(sv, new_attrs, currDepth+=1, maxDepth, most_common_label)
    """
    is_same, label = is_same_label(data)
    if is_same:
        n = Node()
        n.label = label
        return n

    A, infoGain = best_attribute(data, attrs)
    root_node = Node()
    for v in data.get_attribute_possible_vals(A):
        """
        starting at the root node
        what is the most important attribute?
        split the data base on that attribute
        each split is where that attribute = some value
        take a subset of the data and update the column list
        * information gain should be calculated with only columns in the list

        def get_row_subset(self, attribute_name, attribute_value, data = None):
        """
        s_v = data.get_row_subset(A, v)
        if is_empty(s_v):
            n = Node()
            n.label = most_common_label
            root_node.add_child(v, n)
        else:
            new_attrs = [x for x in attrs if x != A]
            n = id3(s_v, new_attrs, currDepth+1, maxDepth, most_common_label)
            root_node.add_child(v, n)

    return root_node




        

    pass


def is_same_label(data):
    keys_list = list(data.column_index_dict.keys())
    labels = data.get_column([keys_list[0]])

    if "p" not in labels:
        return [True, "e"]
    elif "e" not in labels:
        return [True, "p"]
    else:
        return [False, None]


def is_empty(data):
    keys_list = list(data.column_index_dict.keys())
    labels = data.get_column([keys_list[0]])
    if len(labels) == 0:
        return True
    else:
        return False


def getLabels(data):
    keys_list = list(data.column_index_dict.keys())
    labels = data.get_column([keys_list[0]])
    total_p = 0
    total_e = 0
    for i in range(len(labels)):
        if labels[i] == "p":
            total_p += 1
        elif labels[i] == "e":
            total_e += 1
        else:
            raise ValueError("Something went wrong")

    p_ratio = total_p / len(labels)
    e_ratio = total_e / len(labels)
    if p_ratio == 0 or e_ratio == 0:
        total_entropy = 0
    else:
        total_entropy = -p_ratio * log(p_ratio, 2) - e_ratio * log(e_ratio, 2)


    return (total_p, total_e, p_ratio, e_ratio, total_entropy, len(labels))



def best_attribute(data, col_list):

    keys_list = list(data.column_index_dict.keys())
    lables = data.get_column('label')
    if len(lables == 0):
        print("0", keys_list)
    total_p = 0
    total_e = 0
    for i in range(len(lables)):
        if lables[i] == "p":
            total_p += 1
        elif lables[i] == "e":
            total_e += 1
        else:
            raise ValueError("Something went wrong")
    total_size = len(lables)
    p_ratio = total_p / len(lables)
    e_ratio = total_e / len(lables)
    total_entropy = -p_ratio * log(p_ratio, 2) - e_ratio * log(e_ratio, 2)

    # find attribute with most information gain
    max = [None, -1]
    for attribute in col_list:
        subset_entropy = 0
        for value in data.get_attribute_possible_vals(attribute):
            subset = data.get_row_subset(attribute, value)
            (sub_p, sub_e, subp_ratio, sube_ratio, entropy, subset_size) = getLabels(subset)
            weighted_entropy = entropy * (subset_size / total_size)
            subset_entropy += weighted_entropy
        information_gain = total_entropy - subset_entropy
        if information_gain < 0:
            raise ValueError("Something went wrong")
        if information_gain >= max[1]:
            max[0] = attribute
            max[1] = information_gain
    return max



def limiting_depth():
    pass


class kTree():
    root = None

    def __init__(self):
        pass



def test():
    # this will give you the "best attribute"
    d = Data(fpath="data/data/train.csv")
    # def is_same_label(data):
    print(getLabels(d))





# baseline()
# test()
full_trees()
