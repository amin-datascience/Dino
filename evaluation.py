import torch 
import torch.nn as nn 
from sklearn.neighbors import KNeighborsClassfier 
from sklearn.metrics import accuracy_score  

#evaludation Module

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
			lists[f'X_{name}'].append(model(imgs).detach().cpu().numpy())
			lists[f'y_{name}'].append(labels.detach().cpu().numpy())

	arrays = {k: np.concatenate(l)  for k, l in lists.items()}

	estimator = KNeighborsClassifier(n_neighbors = 20)
	estimator.fit(arrays['X_train'], arrays['y_train'])

	y_val_pred = estimator.predict(arrays['X_val'])

	acc = accuracy_score(arrays['y_val'], y_val_pred)

	return acc



def linear_evaluation(dataloader, model, head, n_classes):
	"""This function applies the Linear Evaluation Protocole.
	It takes a trained model and then trains a classifer on 
	top of the frozen features of the model.

	Parameters
	----------
	dataloader: `torch.utils.data.DataLoader`
					The data that we want to use for training
					the classifier. Typically, it is whether 
					the training dataset or the validation dataset. 
					It should be part of a data that the model was trained on.

	model: `torch.nn.Module`
		The pretrained model.

	n_calsses: int 
			The number of classes of the data.

	
	Returns
	-------
	accuracy: float
			The accuracy of the model.
	"""

	model.training = False

	for param in model.parameters():
		param.requires_grad = False

	model = model

	




if __name__ == '__main__':

	dataloader = 






