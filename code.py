# Backprop on the Vowel Dataset
from random import seed
from random import randrange
from random import random
from csv import reader
from math import exp
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd

def minmax(dataset):
        minmax = list()
        stats = [[min(column), max(column)] for column in zip(*dataset)]
        return stats
      

 # Rescale dataset columns to the range 0-1
def normalize(dataset, minmax):
        for row in dataset:
                for i in range(len(row)-1):
                        row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
                        
# Convert string column to float
def column_to_float(dataset, column):
        for row in dataset:
                try:
                        row[column] = float(row[column])
                except ValueError:
                        print("Error with row",column,":",row[column])
                        pass

# Convert string column to integer
def column_to_int(dataset, column):
        class_values = [row[column] for row in dataset]
        unique = set(class_values)
        lookup = dict()
        for i, value in enumerate(unique):
                lookup[value] = i
        for row in dataset:
                row[column] = lookup[row[column]]
        return lookup
      
      
# Find the min and max values for each column

 
# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
        dataset_split = list()
        dataset_copy = list(dataset)
        fold_size = int(len(dataset) / n_folds)
        for i in range(n_folds):
                fold = list()
                while len(fold) < fold_size:
                        index = randrange(len(dataset_copy))
                        fold.append(dataset_copy.pop(index))
                dataset_split.append(fold)
        return dataset_split

# Calculate accuracy percentage
def accuracy_met(actual, predicted):
        correct = 0
        for i in range(len(actual)):
                if actual[i] == predicted[i]:
                        correct += 1
        return correct / float(len(actual)) * 100.0
      
# Evaluate an algorithm using a cross validation split
def run_algorithm(dataset, algorithm, n_folds, *args):
        
        folds = cross_validation_split(dataset, n_folds)
        #for fold in folds:
                #print("Fold {} \n \n".format(fold))
        scores = list()
        for fold_i, fold in enumerate(folds):
                #print("Test Fold {} \n \n".format(fold))
                train_set = list(folds)
                train_set.remove(fold)
                train_set = sum(train_set, [])
                test_set = list()
                for row in fold:
                        row_copy = list(row)
                        test_set.append(row_copy)
                        row_copy[-1] = None
                predictions_train,predictions_test,epoch_error = algorithm(train_set, test_set, *args)
                # ploting epoch error curve
                # taking average of each row i.e. all classes average
                mean = [sum(err)/len(err) for err in epoch_error]
                plt.plot(range(1,len(epoch_error)+1), mean,label=f"fold {fold_i}")
                actual = [row[-1] for row in fold]
                accuracy = accuracy_met(actual, predictions_test)
                cm = confusion_matrix(actual, predictions_test)
                print('\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in cm]))
                #confusionmatrix = np.matrix(cm)
                FP = cm.sum(axis=0) - np.diag(cm)
                FN = cm.sum(axis=1) - np.diag(cm)
                TP = np.diag(cm)
                TN = cm.sum() - (FP + FN + TP)
                print('False Positives\n {}'.format(FP))
                print('False Negetives\n {}'.format(FN))
                print('True Positives\n {}'.format(TP))
                print('True Negetives\n {}'.format(TN))
                TPR = TP/(TP+FN)
                print('Sensitivity \n {}'.format(TPR))
                TNR = TN/(TN+FP)
                print('Specificity \n {}'.format(TNR))
                Precision = TP/(TP+FP)
                print('Precision \n {}'.format(Precision))
                Recall = TP/(TP+FN)
                print('Recall \n {}'.format(Recall))
                Acc = (TP+TN)/(TP+TN+FP+FN)
                print('Áccuracy \n{}'.format(Acc))
                Fscore = 2*(Precision*Recall)/(Precision+Recall)
                print('FScore \n{}'.format(Fscore))
                k=cohen_kappa_score(actual, predictions_test)
                print('Çohen Kappa \n{}'.format(k))
                scores.append(accuracy)
        # labeling plot
        plt.xlabel("# of Epoch")
        plt.ylabel("Error")
        plt.title("Errors vs # of Epoch")
        plt.legend()
        plt.show()
        return scores
      
      
# Calculate neuron activation for an input
def activate(weights, inputs):
        activation = weights[-1]
        for i in range(len(weights)-1):
                activation += weights[i] * inputs[i]
        return activation
# Transfer neuron activation
def transfer(activation,K):
        return sum(np.array(K)*activation**np.arange(len(K)))
  
# Calculate the derivative of an neuron output
# derivative of function, f(x) = K0 + K1*x is f'(x) = K1
def transfer_derivative(activation,K):
        # calculating derivative for f(x) = K0 + K1*x + K2* x^2 + ...
        K1 = K[1:]
        return sum(np.arange(1,len(K))*np.array(K1)*activation**np.arange(len(K1)))
      
# Forward propagate input to a network output
def forward_propagate(network, row, K):
        inputs = row
        for layer in network:
                new_inputs = []
                for neuron in layer:
                        activation = activate(neuron['weights'], inputs)
                        neuron['activation'] = activation
                        neuron['output'] = transfer(activation, K)
                        new_inputs.append(neuron['output'])
                inputs = new_inputs
        return inputs

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected, K):
        for i in reversed(range(len(network))):
                layer = network[i]
                errors = list()
                if i != len(network)-1:
                        for j in range(len(layer)):
                                error = 0.0
                                for neuron in network[i + 1]:
                                        error += (neuron['weights'][j] * neuron['delta'])
                                errors.append(error)
                else:
                        for j in range(len(layer)):
                                neuron = layer[j]
                                errors.append(expected[j] - neuron['output'])
                for j in range(len(layer)):
                        neuron = layer[j]
                        neuron['error'] = errors[j]
                        neuron['delta'] = neuron['error'] * transfer_derivative(neuron['activation'], K)
                        
   
# Update network weights with error
def update_weights(network, row, l_rate, K):
        # updation for each layer in network
        for i in range(len(network)):
                inputs = row[:-1]                
                if i != 0:
                        inputs = [neuron['output'] for neuron in network[i - 1]]
                # updation of each neuron in layaer
                for neuron in network[i]:
                        # each weight in neuron
                        for j in range(len(inputs)):
                                temp = l_rate * neuron['delta'] * inputs[j] + mu * neuron['prev'][j]
                                
                                neuron['weights'][j] -= temp
                                #print("neuron weight{} \n".format(neuron['weights'][j]))
                                neuron['prev'][j] = temp
                        # updation for bias term
                        temp = l_rate * neuron['delta'] + mu * neuron['prev'][-1]
                        neuron['weights'][-1] -= temp
                        neuron['prev'][-1] = temp
                        
        
        # Updation for K
        # loop for each term in K
        for i in range(len(K)):
            # finding dk
            dki = 0
            # loop for each node in first layer
            for neuron in network[0]:
                dki += neuron['error']*neuron['activation']**i
            # average of dk
            dki /= len(network[0])
            # update K also adding regularisation parameter
            K[i] -= l_rate*dki +mu
            
        
                                

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs, K):
        # epoch_error list
        epoch_error = list()
        for epoch in range(n_epoch):
                
                for row in train:
                        outputs = forward_propagate(network, row,K)
                        #print(network)
                        expected = [0 for i in range(n_outputs)]
                        expected[row[-1]] = 1
                        #print("expected row{}\n".format(expected))
                        backward_propagate_error(network, expected,K)
                        update_weights(network, row, l_rate,K)
                # error for each output neuron
                errors = list()
                for neuron in network[-1]:
                    errors.append(neuron['error'])
                # appending to epoch error
                epoch_error.append(errors)
                
        # return it
        return epoch_error
                        
                

# Initialize a network
# Network with only one hidden layer
def initialize_network(n_inputs, n_hidden, n_outputs):
        network = list()
        # defining and adding one hidden layer to network
        hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)], 'prev':[0 for i in range(n_inputs+1)]} for i in range(n_hidden)]        
        network.append(hidden_layer)
        # defining and adding output layer to network
        output_layer = [{'weights':[random() for i in range(n_hidden + 1)],'prev':[0 for i in range(n_hidden+1)]} for i in range(n_outputs)]
        network.append(output_layer)
        # returning network
        return network

# Make a prediction with a network
def predict(network, row, K):
        outputs = forward_propagate(network, row, K)
        return outputs.index(max(outputs))

# Backpropagation Algorithm With Stochastic Gradient Descent
def back_propagation(train, test, l_rate, n_epoch, n_hidden, K):
        n_inputs = len(train[0]) - 1
        n_outputs = len(set([row[-1] for row in train]))
        network = initialize_network(n_inputs, n_hidden, n_outputs)
        epoch_error = train_network(network, train, l_rate, n_epoch, n_outputs,K)
        #print("network {}\n".format(network))
        # train predictions
        predictions_train = list()
        for row in train:
                prediction = predict(network, row,K)
                predictions_train.append(prediction)
        # Test predictions
        predictions_test = list()
        for row in test:
                prediction = predict(network, row,K)
                predictions_test.append(prediction)
        return(predictions_train,predictions_test,epoch_error)


# Working on Banknote dataset
print("Working on Banknote dataset...")
# reading data
banknote_dataset = pd.read_csv('data-banknote-authentication.csv',skiprows=[1,2])
dataset = banknote_dataset.values.tolist()

for i in range(len(dataset[0])-1):
        column_to_float(dataset, i)
# convert class column to integers
column_to_int(dataset, len(dataset[0])-1)
# normalize input variables
mm = minmax(dataset)
normalize(dataset, mm)
# evaluate algorithm
n_folds = 5
l_rate = 0.01
mu=0.001
n_epoch = 10
n_hidden = 20
K = [random()*1e-6 for i in range(2)]
# printing initial K
print("Initial K:",K)
# run algorithm
scores = run_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden,K)

print("Final K:",K)

print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
      
# Loading Data Set
# Iris Data Set
from sklearn.datasets import load_iris
X,y = load_iris(return_X_y=True)
dataset = np.hstack((X,y.reshape(-1,1)))
dataset = dataset.tolist()


for i in range(len(dataset[0])-1):
        column_to_float(dataset, i)
# convert class column to integers
column_to_int(dataset, len(dataset[0])-1)
# normalize input variables
mm = minmax(dataset)
normalize(dataset, mm)
# evaluate algorithm
n_folds = 5
l_rate = 0.01
mu=0.001
n_epoch = 10
n_hidden = 20
K = [random()*1e-6 for i in range(2)]
# printing initial K
print("Initial K:",K)
# run algorithm
scores = run_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden,K)

print("Final K:",K)

print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

