#!/usr/bin/python

import math
import os
import pickle
import sys

import numpy as np

import sklearn.metrics
import sklearn.preprocessing

import ml_metrics

import hyper_classifier


class DataSet(object):
	'''
	TODO DOCUMENTATION
	'''

	def __init__(self, name):
		'''
		TODO DOCUMENTATION
		'''
		self.name = name
		self.data = np.array(np.load('data_sets/{:}/data.npy'.format(self.name)), dtype=np.float32)
		self.target = np.load('data_sets/{:}/target.npy'.format(self.name))
		self.trials = pickle.load(open('data_sets/{:}/trials.pickle'.format(self.name)))


assert(len(sys.argv) == 3)
data_set_name = sys.argv[1]
print('Data set: {:}'.format(data_set_name))
algorithm_name = sys.argv[2]
print('Algorithm: {:}'.format(algorithm_name))

print('Loading {:} data set...'.format(data_set_name))
data_set = DataSet(data_set_name)

classifier = None
if algorithm_name == 'svm':
	print('Creating SVM hyper classifier...')
	classifier = hyper_classifier.SVMHyperClassifier()
elif algorithm_name == 'knn':
	print('Creating KNN hyper classifier...')
	classifier = hyper_classifier.KNNHyperClassifier()
elif algorithm_name == 'rf':
	print('Creating random forest hyper classifier...')
	classifier = hyper_classifier.RandomForestHyperClassifier()
elif algorithm_name == 'lr':
	print('Creating logistic regression hyper classifier...')
	classifier = hyper_classifier.LogisticRegressionHyperClassifier()
elif algorithm_name == 'pr':
	print('Creating perceptron hyper classifier...')
	classifier = hyper_classifier.PerceptronHyperClassifier()

assert(not classifier == None)

for trial in data_set.trials:
	print('Running trial {:}...'.format(trial['index']))

	print('Creating trial data...')
	training_indices = np.array(trial['training'])
	testing_indices = np.array(trial['testing'])
	preprocessed_data = classifier.data_preprocessing(data_set.data)
	training_data = preprocessed_data[training_indices]
	training_target = data_set.target[training_indices]
	testing_data = preprocessed_data[testing_indices]
	testing_target = data_set.target[testing_indices]

	trial_dump_path = 'runs/{:}/{:}/{:}'.format(data_set_name, trial['index'], algorithm_name)
	if not os.path.exists(trial_dump_path):
		os.makedirs(trial_dump_path)

	print('Applying classifier to trial...')
	classifier.apply(trial_dump_path, training_data, training_target, testing_data, testing_target, context={'data_set': data_set_name, 'trial_index': trial['index'], 'algorithm': algorithm_name, 'dump_path': trial_dump_path})

	print('Clearing trial data from memory...')
	del(training_data)
	del(training_target)
	del(testing_data)
	del(testing_target)

