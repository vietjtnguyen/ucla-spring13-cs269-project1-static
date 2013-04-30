import pickle
import numpy as np
from sklearn.cross_validation import StratifiedKFold
X = np.load('data.npy')
y = np.load('target.npy')
trials = []
if X.shape[0] <= 5000:
	kfold = StratifiedKFold(y, n_folds=3)
	for index, indices in enumerate(kfold):
		training_indices, testing_indices = indices
		trial = {
			'index': index,
			'training': list(training_indices),
			'testing': list(testing_indices),
		}
		print(trial)
		trials.append(trial)
else:
	trial = {
		'index': 0,
		'training': xrange(0, 5000),
		'testing': xrange(5000, len(y)),
	}
	trials.append(trial)
with open('trials.pickle', 'w') as f:
	pickle.dump(trials, f)

