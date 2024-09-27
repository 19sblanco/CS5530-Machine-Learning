import csv
import math
import statistics
from collections import Counter
import copy
import numpy as np


class Attribute:
	pass

class Data:

	def __init__(self, *, fpath = "", data = None):

		if not fpath and data is None:
			raise Exception("Must pass either a path to a data file or a numpy array object")

		self.raw_data, self.attributes, self.index_column_dict, \
		self.column_index_dict = self._load_data(fpath, data)

	def _load_data(self, fpath = "", data = None):

		if data is None:
			data = np.loadtxt(fpath, delimiter=',', dtype = str)

		header = data[0]
		index_column_dict = dict(enumerate(header))

		#Python 2.7.x
		# column_index_dict = {v: k for k, v in index_column_dict.items()}

		#Python 3+
		column_index_dict = {v: k for k, v in index_column_dict.items()}

		data = np.delete(data, 0, 0)

		attributes = self._set_attributes_info(index_column_dict, data)

		return data, attributes, index_column_dict, column_index_dict

	def _set_attributes_info(self, index_column_dict, data):
		attributes = dict()

		for index in index_column_dict:
			column_name = index_column_dict[index]
			if column_name == 'label':
				continue
			attribute = Attribute()
			attribute.name = column_name
			attribute.index = index - 1
			attribute.possible_vals = np.unique(data[:, index])
			attributes[column_name] = attribute

		return attributes

	def get_attribute_possible_vals(self, attribute_name):
		"""

		Given an attribute name returns the all of the possible values it can take on.

		Args:
		    attribute_name (str)

		Returns:
		    TYPE: numpy.ndarray
		"""
		return self.attributes[attribute_name].possible_vals

	def get_row_subset(self, attribute_name, attribute_value, data = None):
		"""

		Given an attribute name and attribute value returns a row-wise subset of the data,
		where all of the rows contain the value for the given attribute.

		Args:
		    attribute_name (str):
		    attribute_value (str):
		    data (numpy.ndarray, optional):

		Returns:
		    TYPE: numpy.ndarray
		"""
		if not data:
			data = self.raw_data

		column_index = self.get_column_index(attribute_name)
		new_data = copy.deepcopy(self)
		new_data.raw_data = data[data[:, column_index] == attribute_value]
		return new_data

	def get_column(self, attribute_names, data = None):
		"""

		Given an attribute name returns the corresponding column in the dataset.

		Args:
		    attribute_names (str or list)
		    data (numpy.ndarray, optional)

		Returns:
		    TYPE: numpy.ndarray
		"""
		if not data:
			data = self.raw_data

		if type(attribute_names) is str:
			column_index = self.get_column_index(attribute_names)
			return data[:, column_index]

		column_indicies = []
		for attribute_name in attribute_names:
			column_indicies.append(self.get_column_index(attribute_name))

		return data[:, column_indicies]


	def get_column_index(self, attribute_name):
		"""

		Given an attribute name returns the integer index that corresponds to it.

		Args:
		    attribute_name (str)

		Returns:
		    TYPE: int
		"""
		return self.column_index_dict[attribute_name]

	def __len__(self):
		return len(self.raw_data)
	

import csv
import math
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
        count_1 = 0
        count_0 = 0
        for d in data:
            if int(d['label']) >= 1:
                count_1 += 1
            else:
                count_0 += 1
        if count_1 > count_0:
            return Node(label="1")
        else:
            return Node(label="0")

    num_labels = len(set(item['label'] for item in data))
    if num_labels == 1:
        depths.append(depth)
        if int(data[0]['label']) >= 1:
            return Node(label="1")
        else:
            return Node(label="0")

    # if len(attributes):
    #     depths.append(depth)
    #     return Node(label=majority_label(data))

    best_attribute = max(attributes, key=lambda a: information_gain(data, a))
    node = Node(attribute=best_attribute)
    remaining_attributes = [attr for attr in attributes if attr != best_attribute]
    values = set(1.0 if float(item[best_attribute]) >= 1.0 else float(item[best_attribute]) for item in data)

    for value in values:
        if value == 0:
            # Select items where the attribute is exactly 0
            subset = [item for item in data if float(item[best_attribute]) == 0]
        else:
            # Select items where the attribute is 1 or greater
            subset = [item for item in data if float(item[best_attribute]) >= 1]


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
        attribute_value = instance.get(node.attribute)
        if int(float(attribute_value)) > 1:
            attribute_value = 1.0
        try:
            return get_prediction(instance, node.children[float(attribute_value)])
        except:
            key_types = {type(key) for key in node.children.keys()}

            print("key_types", key_types)
            print("node.children[attribute_value]")
            print(node.children)
            print("attribute_value", attribute_value)
            print("attribute_value", type(attribute_value))
            print("=====")
            exit()


def project():
    train_path = "data/splits/bag-of-words/split0.csv"
    test_path = "data/bag-of-words/bow.test.csv"
    # test_path = "data/bag-of-words/bow.test.csv"


    data = read_data_from_csv(train_path)
    attributes = list(Data(fpath=train_path).column_index_dict)[1:]

    print("build_tree")
    root = build_tree(data, attributes, depth=0, maxDepth=5)
    submission(root)

    # print("eval_tree")
    # print(eval_tree(root, test_path))



"""

"""
def submission(root):
    """
    open a file for submission "drive/MyDrive/decision_tree0.csv"
    open bow.eval.anon.csv
    print("example_id,label")
    loop:
      prediction = get_prediction(datapoint, root)
      print(i,prediction)

    """
    eval_data = read_data_from_csv("data/bag-of-words/bow.eval.anon.csv")
    with open("data/submissions/decision_tree0.csv", "w") as submission_file:
        submission_file.write("example_id,label\n")
        for i in range(len(eval_data)):
            prediction = get_prediction(eval_data[i], root)
            submission_file.write(str(i) + "," + prediction + "\n")






"""
return the accuracy of a decision tree
"""
def eval_tree(root, test_path):
    total_guesses = 0
    correct_guesses = 0
    test_data = read_data_from_csv(test_path)
    for d in test_data:
        correct_label = d['label']
        prediction = get_prediction(d, root)
        total_guesses += 1

        if correct_label == prediction:
            correct_guesses += 1

    return correct_guesses / total_guesses




def baseline(train_path, test_path):
    csv_file_path = train_path
    # data = read_data_from_csv(csv_file_path)
    d = Data(fpath=csv_file_path)
    train_labels = d.get_column('label')
    total_1 = 0
    total_0 = 0
    for l in train_labels:
        if l == "0":
            total_0 += 1
        elif l == "1":
            total_1 += 1
        else:
            print("something went wrong")
            exit()
    # print(total_0 / len(labels))
    # print(total_1 / len(labels))

    csv_file_path = test_path
    d = Data(fpath=csv_file_path)
    test_labels = d.get_column('label')
    if total_0 > total_1:
        majority_label = "0"
    else:
        majority_label = "1"
    correct_prediction = 0
    for l in test_labels:
        if l == majority_label:
            correct_prediction += 1
    print(correct_prediction/len(test_labels))



project()


    