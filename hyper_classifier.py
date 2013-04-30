#!/usr/bin/python

import math
import os
import pickle
import sys

import numpy as np

import sklearn.preprocessing
import sklearn.metrics
import sklearn.svm
import sklearn.neighbors
import sklearn.ensemble
import sklearn.linear_model

import ml_metrics


class ClassifierRun(object):
	'''
	TODO DOCUMENTATION
	'''
	
	def __init__(self, name, parameterization):
		'''
		TODO DOCUMENTATION
		'''
		self.name = name
		self.parameterization = parameterization
		self.classifier = None
	
	def run(self, hyper_classifier, training_data, training_target, testing_data, testing_target):
		'''
		TODO DOCUMENTATION
		'''
		results = {'name': self.name, 'parameterization': self.parameterization, 'exception': None}
		try:
			self.classifier = hyper_classifier.make_classifier(training_data, training_target, **self.parameterization)
			self.classifier.fit(training_data, training_target)
			results['predicted'] = self.classifier.predict(testing_data)
		except MemoryError as e:
			raise e
		except Exception as e:
			print(repr(e))
			results['exception'] = e
		else:
			# attempt to save memory
			del(self.classifier)
			self.classifier = None

			results['ml_metric_ce'] = ml_metrics.ce(testing_target, results['predicted'])
			results['ml_metric_rmse'] = ml_metrics.rmse(testing_target, results['predicted'])
			results['sklearn_metric_accuracy'] = sklearn.metrics.accuracy_score(testing_target, results['predicted'])
			results['sklearn_metric_f1'] = sklearn.metrics.f1_score(testing_target, results['predicted'])
			results['sklearn_metric_precision'] = sklearn.metrics.precision_score(testing_target, results['predicted'])
			results['sklearn_metric_recall'] = sklearn.metrics.recall_score(testing_target, results['predicted'])

			results['ml_metric_auc'] = {}
			results['sklearn_metric_auc'] = {}
			for label in set(testing_target):
				binary_testing_target = np.array(map(lambda x: 1 if x == label else 0, testing_target))
				binary_predicted = np.array(map(lambda x: 1 if x == label else 0, results['predicted']))
				results['ml_metric_auc'][label] = ml_metrics.auc(binary_testing_target, binary_predicted)
				results['sklearn_metric_auc'][label] = sklearn.metrics.auc_score(binary_testing_target, binary_predicted)

		return results


class HyperClassifier(object):
	'''
	TODO DOCUMENTATION
	'''

	def __init__(self, name):
		'''
		TODO DOCUMENTATION
		'''
		self.name = name
		self.runs = []
	
	def data_preprocessing(self, data):
		return data

	def apply(self, dump_path, training_data, training_target, testing_data, testing_target, context={}):
		'''
		TODO DOCUMENTATION
		'''
		for index, run in enumerate(self.runs):
			dump_filename = '{:}/{:}.pickle'.format(dump_path, run.name)
			results = None
		
			print('')
			if os.path.exists(dump_filename):
				print('Found cached result "{:}", loading...'.format(dump_filename))
				with open(dump_filename, 'r') as f:
					results = pickle.load(f)
			else:
				print('No cached result "{:}", building and running classifier...'.format(dump_filename))

			if results == None:
				results = run.run(self, training_data, training_target, testing_data, testing_target)
			
			results.update(context)

			print('run: {:}'.format(results['name']))
			print('parameters: {:}'.format(results['parameterization']))
			if results['exception'] == None:
				print('ml ce: {:}'.format(results['ml_metric_ce']))
				print('ml rmse: {:}'.format(results['ml_metric_rmse']))
				print('sk acc: {:}'.format(results['sklearn_metric_accuracy']))
				print('sk f1: {:}'.format(results['sklearn_metric_f1']))
				print('sk prec: {:}'.format(results['sklearn_metric_precision']))
				print('sk recl: {:}'.format(results['sklearn_metric_recall']))
				for label in results['ml_metric_auc'].keys():
					print('label: {:}'.format(label))
					print('ml auc: {:}'.format(results['ml_metric_auc'][label]))
					print('sk auc: {:}'.format(results['sklearn_metric_auc'][label]))
		
			if not os.path.exists(dump_filename):
				print('Saving result "{:}"...'.format(dump_filename))
				with open(dump_filename, 'w') as f:
					pickle.dump(results, f)


class SVMHyperClassifier(HyperClassifier):
	'''
	TODO DOCUMENTATION
	'''

	def __init__(self):
		'''
		TODO DOCUMENTATION
		'''
		super(SVMHyperClassifier, self).__init__('svm')
		
		#C_values = np.logspace(-9, 5, 15)
		C_values = np.logspace(-9, 5, 6)
		
		#for C in C_values:
		#	self.runs.append(ClassifierRun('{:03d}'.format(len(self.runs)), {'kernel': 'linear', 'C': C}))
		
		for degree in [2, 3]:
			for C in C_values:
				run = ClassifierRun(
					'kernel(poly)-degree({:d})-C({:e})'.format(degree, C),
					{'kernel': 'poly', 'C': C, 'degree': degree})
				self.runs.append(run)
		
		#for width in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2]:
		for gamma in [0.001, 0.01, 0.1, 1, 2]:
			for C in C_values:
				run = ClassifierRun(
					'kernel(rbf)-gamma({:})-C({:e})'.format(gamma, C),
					{'kernel': 'rbf', 'C': C, 'gamma': gamma})
				self.runs.append(run)
	
	def data_preprocessing(self, data):
		print('Scaling data for SVM...')
		return sklearn.preprocessing.scale(data)

	def make_classifier(self, training_data, training_target, **kwargs):
		return sklearn.svm.SVC(**kwargs)


class KNNHyperClassifier(HyperClassifier):
	'''
	TODO DOCUMENTATION
	'''

	def __init__(self):
		'''
		TODO DOCUMENTATION
		'''
		super(KNNHyperClassifier, self).__init__('knn')

		for neighbors in [5, 40, 200, 500, 1000]:
			for weights in ['uniform', 'distance']:
				run = ClassifierRun(
					'neighbors({:d})-weights({:})'.format(neighbors, weights),
					{'n_neighbors': neighbors, 'weights': weights})
				self.runs.append(run)
			
			for width in np.arange(0.1, 820 + 1, (820 - 0.1) / 9):
				run = ClassifierRun(
					'neighbors({:d})-weights(gaussian)-width({:e})'.format(neighbors, width),
					{'n_neighbors': neighbors, 'width': width})
				self.runs.append(run)
	
	def data_preprocessing(self, data):
		print('Scaling data for kNN...')
		return sklearn.preprocessing.scale(data)

	def make_classifier(self, training_data, training_target, **kwargs):
		n_neighbors = kwargs['n_neighbors']
		if kwargs.has_key('weights'):
			return sklearn.neighbors.KNeighborsClassifier(
				n_neighbors=n_neighbors,
				weights=kwargs['weights'])
		else:
			width = kwargs['width']
			return sklearn.neighbors.KNeighborsClassifier(
				n_neighbors=n_neighbors,
				weights=np.vectorize(lambda d: math.exp(-d * d / (width * width))))


class RandomForestHyperClassifier(HyperClassifier):
	'''
	TODO DOCUMENTATION
	'''

	def __init__(self):
		'''
		TODO DOCUMENTATION
		'''
		super(RandomForestHyperClassifier, self).__init__('rf')

		for split in [0.5, 1, 2, 4, 8]:
			run = ClassifierRun(
				'estimators({:d})-split({:})'.format(500, split),
				{'n_estimators': 500, 'split': split})
			self.runs.append(run)

	def make_classifier(self, training_data, training_target, **kwargs):
		n_estimators = kwargs['n_estimators']
		num_of_features = training_data.shape[1]
		sqrt_num_of_features = math.sqrt(num_of_features)
		max_features = int(min(sqrt_num_of_features * kwargs['split'], num_of_features))
		print(num_of_features)
		print(sqrt_num_of_features)
		print(kwargs['split'])
		print(max_features)
		return sklearn.ensemble.RandomForestClassifier(n_estimators=n_estimators, max_features=max_features)


class LogisticRegressionHyperClassifier(HyperClassifier):
	'''
	TODO DOCUMENTATION
	'''

	def __init__(self):
		'''
		TODO DOCUMENTATION
		'''
		super(LogisticRegressionHyperClassifier, self).__init__('lr')
		
		#C_values = np.logspace(-9, 5, 15)
		C_values = np.logspace(-9, 5, 6)
		
		for C in C_values:
			run = ClassifierRun(
				'penalty(l1)-C({:e})'.format(C),
				{'penalty': 'l1', 'C': C})
			self.runs.append(run)
		
		for C in C_values:
			run = ClassifierRun(
				'penalty(l2)-C({:e})'.format(C),
				{'penalty': 'l2', 'C': C})
			self.runs.append(run)
	
	def data_preprocessing(self, data):
		print('Scaling data for logistic regression...')
		return sklearn.preprocessing.scale(data)

	def make_classifier(self, training_data, training_target, **kwargs):
		return sklearn.linear_model.LogisticRegression(**kwargs)


class PerceptronHyperClassifier(HyperClassifier):
	'''
	TODO DOCUMENTATION
	'''

	def __init__(self):
		'''
		TODO DOCUMENTATION
		'''
		super(PerceptronHyperClassifier, self).__init__('pr')
		
		#C_values = np.logspace(-9, 5, 15)
		C_values = np.logspace(-9, 5, 6)
		
		for C in C_values:
			run = ClassifierRun(
				'penalty(none)-alpha({:e})'.format(C),
				{'penalty': None, 'alpha': C})
			self.runs.append(run)
		
		for C in C_values:
			run = ClassifierRun(
				'penalty(l1)-alpha({:e})'.format(C),
				{'penalty': 'l1', 'alpha': C})
			self.runs.append(run)
		
		for C in C_values:
			run = ClassifierRun(
				'penalty(l2)-alpha({:e})'.format(C),
				{'penalty': 'l2', 'alpha': C})
			self.runs.append(run)
		
		for C in C_values:
			run = ClassifierRun(
				'penalty(elasticnet)-alpha({:e})'.format(C),
				{'penalty': 'elasticnet', 'alpha': C})
			self.runs.append(run)

	def make_classifier(self, training_data, training_target, **kwargs):
		return sklearn.linear_model.Perceptron(**kwargs)

