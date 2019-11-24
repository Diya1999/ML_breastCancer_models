# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 22:23:40 2019

@author: Hp
"""

from random import randrange
from csv import reader
#loading the dataset 
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset
 
#To calculate the groups for the proposed split in data
def split_test(index, value, dataset):
    left = []
    right = []
    for col in dataset:
        if col[index] < value:
            left.append(col)
        else:
            right.append(col)
    return left, right
#To calculate the gini index of a proposed split 
def gini(groups, classes):
    n_instances = 0 
    for group in groups:
        n_instances = float(n_instances + len(group))
    gini_val = 0.0
    for group in groups:
        size = float(len(group))
        if(size==0):
            continue
        score = 0.0
        for class_val in classes:
            g = ([row[-1] for row in group].count(class_val)) / size
            score = score + (g*g)
        gini_val = gini_val + (1.0 - score)* (size/ n_instances)
    return gini_val 
#calculate the optimal split
def get_split(dataset, n_features):
    class_mid = set(row [-1] for row in dataset)
    class_val = list(class_mid)
    opt_index, opt_value, opt_score, opt_groups = 9999, 9999, 9999, None
    feat = list()
    while (len(feat) < n_features):
        col = randrange(len(dataset[0])-1)
        if col not in feat:
            feat.append(col)
    for col in feat:
        for row in dataset:
            groups = split_test(col, row[col], dataset) 
            gini_val = gini(groups, class_val)
            if(gini_val < opt_score):
                opt_index, opt_value, opt_score, opt_groups = col, row[col], gini_val, groups
        return {'index':opt_index, 'value':opt_value, 'groups':opt_groups}
#to get the most probable class of the terminal node 
def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)
#to get splits of the data 
def split(node, min_size, n_features, depth):
    left, right = node['groups']
    del(node['groups'])
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    if( len(left) <= min_size):
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left, n_features)
        split(node['left'], min_size, n_features, depth+1)
    if( len(right) <= min_size):
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right, n_features)
        split(node['right'], min_size, n_features, depth+1)

def build_tree(train, min_size, n_features):
    root = get_split(train, n_features)
    split(root, min_size, n_features, 1)
    return root
    
def predict(node, row):
    #print(node)
    #print(node['index'])
    #print(row[node['index']])
    if(row[int(node['index'])] < node['value']):
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

def sampling(dataset, ratio):
    sample = list()
    size = round(len(dataset)*ratio)
    while(len(sample) < size ):
        random_pick = randrange(len(dataset))
        sample.append(dataset[random_pick])
    return sample
def random_forest_prediction(trees, row):
    predictions = []
    for tree in trees:
        #print(tree)
        prediction = predict(tree,row)
        predictions.append(prediction)
    max_prediction = max(set(predictions), key = predictions.count)
    return (max_prediction) 
def create_random_forests(train, test, min_size, sample_size, n_trees, n_features):
    forest = list()
    predictions =  []
    for i in range(n_trees):
        sample = sampling(train, sample_size)
        tree = build_tree(sample, min_size, n_features)
        print(i)
        print('\n')
        print(tree)
        forest.append(tree)
    for row in test:
        prediction = random_forest_prediction(forest, row)
        predictions.append(prediction)
    #print(predictions)
    return(predictions)
def accuracy_metric(actual, predicted):
    correct = 0
    #print(predicted)
    for i in range(len(actual)):
        if (actual[i] == predicted[i]):
            correct += 1
    return correct / float(len(actual)) * 100.0
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for j in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split
def evaluation_algorithm(dataset, algorithm, n_folds, min_size, sample_size, n_trees, n_features):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            #row_copy[-1] = 0
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, min_size, sample_size, n_trees, n_features)
        #print(predicted)
        actual = [row[-1] for row in fold]
        #print(actual)
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores
if __name__ == "__main__":
    filename = 'ml_data.csv'
    dataset = load_csv(filename)
    n_folds = 5
    min_size = 1
    sample_size = 1.0
    n_features = int((len(dataset[0])-1)/2)
    #print(n_features)
    for n_trees in [1]:
        scores = evaluation_algorithm(dataset, create_random_forests, n_folds, min_size, sample_size, n_trees, n_features)
        print('Trees: %d' % n_trees)
        print('Scores: %s' % scores)
        mean_accuracy = (sum(scores)/float(len(scores)))
        print('Mean Accuracy: %.3f%%' % (mean_accuracy))

