# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 22:23:40 2019

@author: Hp
"""
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt 
from random import randrange
from csv import reader
#calculate the optimal split
def get_split(dataset, n_features):
    class_mid = set(row [-1] for row in dataset)
    classes = list(class_mid)
    opt_index, opt_value, opt_score, opt_groups = 999, 999, 999, None
    feat = list()
    while (len(feat) < n_features):
        col = randrange(len(dataset[0])-1)
        if col not in feat:
            feat.append(col)
    for col in feat:
        for row in dataset:
            groups = split_test(col, row[col], dataset)
            n_instances = 0 
            count = 0
            #finding gini index 
            for group in groups:
                n_instances = float(n_instances + len(group))
            gini_val_mid = 0.0
            for group in groups:
                size = int(len(group))
                if(size==0):
                    continue
                score = 0.0
                for class_val in classes:
                    for row in group:
                        if (row[-1] == class_val):
                            count = count + 1 
                g = count / size 
                score = score + (g*g)
                gini_val_mid = gini_val_mid + (1.0 - score)* (size/ n_instances)
            gini_val = gini_val_mid 
            #gets optimal column to split and the point at which column should be split 
            if(gini_val < opt_score):
                opt_index, opt_value, opt_score, opt_groups = col, row[col], gini_val, groups
        return {'index':opt_index, 'value':opt_value, 'groups':opt_groups}
#To calculate the groups for the proposed split in data
def split_test(index, value, dataset):
    left = []
    right = []
    groups = []
    for col in dataset:
        if col[index] < value:
            left.append(col)
        else:
            right.append(col)
    groups.append(left)
    groups.append(right)
    return groups
#to get splits of the data 
def create_tree(node, min_size, n_features, depth):
    left, right = node['groups']
    del(node['groups'])
    if not left or not right:
        #to get the most probable class of the terminal node
        outcomes = [row[-1] for row in (left+right)]
        node['left'] = node['right'] = max(set(outcomes), key = outcomes.count)
        return
    if( len(left) <= min_size):
        #to get the most probable class of the terminal node
        outcomes = [row[-1] for row in left]
        node['left'] = max(set(outcomes), key = outcomes.count)
    else:
        node['left'] = get_split(left, n_features)
        #split the node future 
        create_tree(node['left'], min_size, n_features, depth+1)
    if( len(right) <= min_size):
        #to get the most probable class of the terminal node
        outcomes = [row[-1] for row in (left+right)]
        node['right'] = max(set(outcomes), key = outcomes.count)
    else:
        node['right'] = get_split(right, n_features)
        #split the node future
        create_tree(node['right'], min_size, n_features, depth+1)

def random_forests(train, test, min_size, n_trees, n_features):
    forest = list()
    predictions =  []
    #Creating trees 
    for i in range(n_trees):
        tree = get_split(train, n_features)
        create_tree(tree, min_size, n_features, 1)
        forest.append(tree)
    #Testing the trees  
    for row in test:
        predictions_mid = []
        for tree in forest:
            prediction = predict(tree,row)
            predictions_mid.append(prediction)
        max_prediction = max(set(predictions_mid), key = predictions_mid.count)
        prediction = max_prediction
        predictions.append(prediction)
    return(predictions)
#traversing the trees to predict the otucome to for the give test row 
def predict(node, row):
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
def evaluation_algorithm(dataset,  n_folds, min_size, n_trees, n_features):
    #splitting the data into folds 
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for j in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    folds = dataset_split
    scores = list()
    #creating test and train datasets
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
        #getting the predictions for the algorithm 
        predicted = random_forests(train_set, test_set, min_size, n_trees, n_features)
        #print(predicted)
        actual = list()
        for row in fold:
            actual.append(row[-1])
        accuracy = accuracy_score(actual, predicted)
        accuracy = accuracy *100
        scores.append(accuracy)
    return scores
#driver code 
if __name__ == "__main__":
    dataset = []
    with open('ml_data.csv', 'r') as f:
        csv_reader = reader(f)
        for row in csv_reader:
            dataset.append(row)
    fold_size = 7
    min_termination_size = 1
    sum_scores = 0.0
    n_features = int((len(dataset[0])-1)/2)
    n_trees = 1
    score = []
    n_trees_list = [1]
    #n_trees_list = [1,3,5]
    #n_trees_list = [1,2,3,4,5,6,7,8,9,10]
    mean_accuracies = list()
    for n_trees in n_trees_list:
        scores = evaluation_algorithm(dataset,fold_size, min_termination_size, n_trees, n_features)
        print('Trees: %d' % n_trees)
        for score in scores:  
            sum_scores = sum_scores + score
        mean_accuracy = (sum_scores/7)
        mean_accuracies.append(mean_accuracy)
        sum_scores = 0 
        print('Mean Accuracy: %.2f%%' % (mean_accuracy))
    plt.plot(n_trees_list, mean_accuracies, color='black', linestyle='dashed', linewidth = 3)
    plt.xlabel('Number of trees')
    plt.ylabel('Accuracy')
    plt.show()
    plt.savefig('random_forest.png')


 
