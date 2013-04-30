#!/usr/bin/python

import os
import pickle
import pprint
import sys

import numpy as np

assert(len(sys.argv) == 4)

data_set = sys.argv[1]
algo = sys.argv[2]
metric = sys.argv[3]
        
metric_func = max
if metric == 'ml_metric_ce' or metric == 'ml_metric_rsme':
	metric_func = min

trials = pickle.load(open('data_sets/{:}/trials.pickle'.format(data_set), 'r'))

results = {'max': -np.inf, 'min': np.inf, 'scores': [], 'results': []}

for trial in trials:
	run_path = 'runs/{:}/{:}/{:}'.format(data_set, trial['index'], algo)
	best_result_path = '{:}/{:}.pickle'.format(run_path, metric)
	best_result = pickle.load(open(best_result_path, 'r'))
	results['max'] = max(results['max'], best_result[metric])
	results['min'] = min(results['min'], best_result[metric])
	results['scores'].append(best_result[metric])
	results['results'].append(best_result)
results['mean'] = np.mean(results['scores'])
results['variance'] = np.std(results['scores'])

print('')
pprint.pprint(results)
        
save_path = 'runs/{:}/{:}-{:}.pickle'.format(data_set, algo, metric)

print('')
print('saving to "{:}"'.format(save_path))
with open(save_path, 'w') as f:
    pickle.dump(results, f)

