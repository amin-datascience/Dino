import torch 
import torch.nn as nn 
from sklearn.neighbors import KNeighborsClassfier 
from sklearn.metrics import accuracy_score  


def compute_knn(model, dataloader_train, dataloader_validation):
	"""Computes KNN 

	Parameters
	----------
	model: torch.nn.Module
		Student Vision Transformer.

	Returns
	-------
	val_accuracy: float 
		Validation Accuracy
	"""

	device = next(model.parameters()).device 

	data_loaders = {
	'train': dataloader_train, 
	'val': dataloader_validation
	}

	lists = { 
		'X_train': [],
		'y_train': [],
		'X_val': [], 
		'y_val': []
	}

	for name, dataloader in data_loaders.items():
		for imgs, labels in dataloader: 
			imgs = imgs.to(device)
			lists[f'X_{name}'].append(model(imgs).detach().numpy())
			lits[f'y_{name}'].append(labels.detach().numpy())

	arrays = {k: np.concatenate(l)  for k, l in lists.items()}

	estimator = KNeighborsClassfier()
	estimator.fit(arrays['X_train'], arrays['y_train'])

	y_val_pred = estimator.predict(arrays['X_val'])

	acc = accuracy_score(arrays['y_val'], y_val_pred)

	return acc 











