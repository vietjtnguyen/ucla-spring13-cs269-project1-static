#!/usr/bin/python

import os
import pickle
import pprint
import sys

assert(len(sys.argv) == 4)

data_set = sys.argv[1]
algo = sys.argv[2]
metric = sys.argv[3]
        
metric_func = max
if metric == 'ml_metric_ce' or metric == 'ml_metric_rsme':
	metric_func = min

trials = pickle.load(open('data_sets/{:}/trials.pickle'.format(data_set), 'r'))

for trial in trials:
	run_path = 'runs/{:}/{:}/{:}'.format(data_set, trial['index'], algo)

	results_list = os.listdir(run_path)
	results = []

	for results_file in results_list:
		with open('{:}/{:}'.format(run_path, results_file), 'r') as f:
			results.append(pickle.load(f))

	print('')
	print('finding best result ({:})...'.format(metric))
	best_result = results[0]
	for result in results:
		print(best_result[metric])
		try:
			if metric_func(result[metric], best_result[metric]) == result[metric]:
				best_result = result
		except KeyError:
			pass

	print('')
	pprint.pprint(best_result)

	save_path = '{:}/{:}.pickle'.format(run_path, metric)
			
	print('')
	print('saving to "{:}"'.format(save_path))
	with open(save_path, 'w') as f:
		pickle.dump(best_result, f)

